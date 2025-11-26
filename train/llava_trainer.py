import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional

# For masking ignored label positions
try:
    from llava.constants import IGNORE_INDEX
except ImportError:
    IGNORE_INDEX = -100  # Fallback to standard HF ignore index

from llava.train.embedder import PatchEmbedder


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def __init__(self, patch_embedder=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_embedder = patch_embedder if patch_embedder is not None else PatchEmbedder()
        self.patch_embedder.freeze()
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.patch_embedder.to(device_str)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            # Collect special module names
            projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
            alignment_parameters = [name for name, _ in opt_model.named_parameters() if "alignment_encoder" in name]

            use_proj_lr = self.args.mm_projector_lr is not None
            if use_proj_lr:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                alignment_parameters = [name for name, _ in opt_model.named_parameters() if "alignment_encoder" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n not in alignment_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n not in alignment_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
                # Alignment encoder groups (if present)
                if alignment_parameters:
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in alignment_parameters and p.requires_grad)],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in alignment_parameters and p.requires_grad)],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ])
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
            Use the model's native loss, with a hook for extra GranD loss.
        """
        grand_mask = inputs.pop('grand_source_mask', None)
        grand_bboxes = inputs.pop('grand_bboxes', None)
        grand_image_paths = inputs.pop('grand_image_paths', None)
        grand_dense_captions = inputs.pop('grand_dense_captions', None)
        grand_dense_labels = inputs.pop('grand_dense_labels', None)

        if getattr(self.args, 'local_rank', 0) in (-1, 0):
            logger.debug(f"[GrandAlignDebug] batch_keys={list(inputs.keys())} has_grand_mask={grand_mask is not None} has_bboxes={grand_bboxes is not None} has_paths={grand_image_paths is not None} has_captions={grand_dense_captions is not None} has_labels={grand_dense_labels is not None}")

        # Request hidden states if any grand samples present
        if grand_mask is not None and 'output_hidden_states' not in inputs:
            inputs['output_hidden_states'] = True

        labels = inputs.get('labels', None)
        outputs = model(**inputs)
        base_loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        grand_extra_loss = torch.zeros((), device=base_loss.device)
        attempted_phrase_total = 0
        matched_phrase_total = 0
        crop_total = 0
        matched_crop_total = 0
        if grand_mask is not None and labels is not None and grand_bboxes is not None and grand_image_paths is not None and grand_dense_labels is not None and grand_dense_captions is not None:
            grand_mask = grand_mask.bool()
            if getattr(self.args, 'local_rank', 0) in (-1, 0):
                try:
                    logger.debug(f"[GrandAlignDebug] grand_mask_any={grand_mask.any().item()} grand_mask_sum={grand_mask.sum().item()}")
                except Exception:
                    logger.debug("[GrandAlignDebug] grand_mask statistics unavailable")
            if grand_mask.any():
                # Obtain last hidden states
                hidden_states = None
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hidden_states = outputs.hidden_states[-1]
                elif isinstance(outputs, tuple) and len(outputs) > 2:
                    hidden_states = outputs[2]
                if getattr(self.args, 'local_rank', 0) in (-1, 0):
                    logger.debug(f"[GrandAlignDebug] hidden_states_found={hidden_states is not None}")
                if hidden_states is not None:
                    per_sample_losses = []
                    for b_idx, is_grand in enumerate(grand_mask):
                        if not is_grand:
                            continue
                        sample_bboxes = grand_bboxes[b_idx] if b_idx < len(grand_bboxes) else []
                        image_path = grand_image_paths[b_idx] if b_idx < len(grand_image_paths) else None
                        if not sample_bboxes or image_path is None:
                            if getattr(self.args, 'local_rank', 0) in (-1, 0):
                                logger.debug(f"[GrandAlignDebug] skip_sample b={b_idx} reason=no_bboxes_or_path")
                            continue
                        label_row = labels[b_idx]
                        # Mask for generated caption tokens (assistant response)
                        token_mask = (label_row != IGNORE_INDEX) & (label_row != -100)
                        if token_mask.sum() == 0:
                            if getattr(self.args, 'local_rank', 0) in (-1, 0):
                                logger.debug(f"[GrandAlignDebug] skip_sample b={b_idx} reason=empty_token_mask")
                            continue
                        generated_token_ids = label_row[token_mask].tolist()
                        generated_indices = torch.nonzero(token_mask, as_tuple=False).squeeze(-1).tolist()
                        # Load image and crop patches
                        try:
                            from PIL import Image
                            img = Image.open(image_path).convert('RGB')
                        except Exception as e:  # capture error for logging
                            if getattr(self.args, 'local_rank', 0) in (-1, 0):
                                logger.debug(f"[GrandAlignDebug] skip_sample b={b_idx} reason=image_open_fail error={repr(e)}")
                            continue
                        crops = []
                        for (l, t, r, b) in sample_bboxes:
                            try:
                                crops.append(img.crop((l, t, r, b)))
                            except Exception:
                                continue
                        if not crops:
                            if getattr(self.args, 'local_rank', 0) in (-1, 0):
                                logger.debug(f"[GrandAlignDebug] skip_sample b={b_idx} reason=no_valid_crops")
                            continue
                        crop_total += len(crops)
                        with torch.no_grad():
                            patch_embeds = self.patch_embedder(crops)  # [N, dim_patch]
                        # Alignment encoder projection
                        try:
                            base_model = model.get_model() if hasattr(model, 'get_model') else model
                            align_enc = getattr(base_model, 'alignment_encoder', None)
                        except Exception:
                            align_enc = None
                        if align_enc is None:
                            logger.warning("Alignment encoder not found in model; skipping GranD loss.")
                            continue
                        patch_embeds = patch_embeds.to(hidden_states.device)
                        aligned_vecs = align_enc(patch_embeds)  # [N, hidden]

                        crop_losses = []
                        phrases = grand_dense_labels[b_idx] if b_idx < len(grand_dense_labels) else []
                        full_caption_text = grand_dense_captions[b_idx] if b_idx < len(grand_dense_captions) else ""
                        if getattr(self.args, 'local_rank', 0) in (-1, 0):
                            logger.debug(f"[GrandAlignDebug] sample b={b_idx} bboxes={len(sample_bboxes)} crops={len(crops)} phrases={len(phrases)}")

                        for crop_i, phrase in enumerate(phrases):
                            phrase = phrase.strip()
                            if not phrase:
                                continue
                            attempted_phrase_total += 1
                            # Try multiple text variants to handle capitalization differences
                            variants = [phrase, phrase.lower(), phrase.capitalize()]
                            matched_embed = None
                            for variant in variants:
                                if self.tokenizer is None:
                                    break
                                variant_tokens = self.tokenizer(variant, add_special_tokens=False).input_ids
                                # Subsequence search
                                for start in range(len(generated_token_ids) - len(variant_tokens) + 1):
                                    if generated_token_ids[start:start+len(variant_tokens)] == variant_tokens:
                                        orig_span = generated_indices[start:start+len(variant_tokens)]
                                        span_embeds = hidden_states[b_idx][orig_span]
                                        if span_embeds.numel() > 0:
                                            matched_embed = span_embeds.mean(dim=0)
                                        break
                                if matched_embed is not None:
                                    break
                            if matched_embed is None:
                                continue
                            if crop_i >= aligned_vecs.size(0):
                                continue
                            matched_phrase_total += 1
                            matched_crop_total += 1
                            aligned_vec = aligned_vecs[crop_i]
                            sim = F.cosine_similarity(F.normalize(aligned_vec.unsqueeze(0), dim=-1),
                                                      F.normalize(matched_embed.unsqueeze(0), dim=-1)).mean()
                            crop_losses.append(1 - sim)
                        if getattr(self.args, 'local_rank', 0) in (-1, 0):
                            logger.debug(f"[GrandAlignDebug] sample b={b_idx} crop_losses_count={len(crop_losses)}")
                        if crop_losses:
                            per_sample_losses.append(torch.stack(crop_losses).mean())
                    if per_sample_losses:
                        grand_extra_loss = torch.stack(per_sample_losses).mean()
                        if getattr(self.args, 'local_rank', 0) in (-1, 0):
                            logger.debug(f"[GrandAlignDebug] per_sample_losses_count={len(per_sample_losses)} grand_extra_loss={grand_extra_loss.item():.6f}")
                    else:
                        if getattr(self.args, 'local_rank', 0) in (-1, 0):
                            logger.debug("[GrandAlignDebug] no_per_sample_losses grand_extra_loss=0")

        # Logging summary (rank 0 only to avoid spam)
        if (grand_extra_loss > 0) and (getattr(self.args, 'local_rank', 0) in (-1, 0)):
            try:
                step = getattr(self.state, 'global_step', None)
            except Exception:
                step = None
            print(
                f"[GrandAlign] step={step} base_loss={base_loss.item():.4f} align_loss={grand_extra_loss.item():.4f} "
                f"phrases_attempted={attempted_phrase_total} phrases_matched={matched_phrase_total} crops_total={crop_total} crops_matched={matched_crop_total}"
            )

        weight = getattr(self.args, 'grand_alignment_loss_weight', 0.5)
        total_loss = base_loss + (grand_extra_loss * weight)
        if return_outputs:
            return total_loss, outputs
        return total_loss