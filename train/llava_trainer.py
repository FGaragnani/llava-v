import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import bitsandbytes
import os

from torch.utils.data import Sampler
from llava.train.attn_pooler import MHAPooler

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
        if self.args.text_token_pool == 'attn':
            self.text_pooler = MHAPooler(dim=self.model.config.hidden_size, num_heads=8)

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
        model.generation_config.do_sample = True
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

        # Request hidden states if any grand samples present
        if grand_mask is not None and 'output_hidden_states' not in inputs:
            inputs['output_hidden_states'] = True

        labels = inputs.get('labels', None)
        outputs = model(**inputs)
        base_loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        # Masked unification path: always touch alignment_encoder with a masked batch to keep graph consistent.
        if os.environ.get("GRAND_FORCE_MASK", "0") == "1":
            try:
                base_model = model.get_model() if hasattr(model, 'get_model') else model
                align_enc = getattr(base_model, 'alignment_encoder', None)
                hidden_seq = getattr(outputs, 'hidden_states', None)
                hidden_last = None
                if isinstance(hidden_seq, (list, tuple)) and len(hidden_seq) > 0 and isinstance(hidden_seq[-1], torch.Tensor):
                    hidden_last = hidden_seq[-1]
                elif isinstance(outputs, (list, tuple)) and len(outputs) > 2 and isinstance(outputs[2], torch.Tensor):
                    hidden_last = outputs[2]
                if align_enc is not None and isinstance(hidden_last, torch.Tensor) and hidden_last.size(1) > 0:
                    dummy_tokens = hidden_last[:, 0, :]
                    mask_vec = grand_mask.bool() if grand_mask is not None else torch.ones(dummy_tokens.size(0), dtype=torch.bool, device=dummy_tokens.device)
                    mask_float = mask_vec.float().unsqueeze(-1)
                    masked_input = dummy_tokens * mask_float
                    # Ensure dtype match
                    try:
                        param_dtype = next(align_enc.parameters()).dtype
                        if masked_input.dtype != param_dtype:
                            masked_input = masked_input.to(param_dtype)
                    except Exception:
                        pass
                    dummy_out = align_enc(masked_input)
                    base_loss = base_loss + dummy_out.mean() * 0.0
            except Exception as e:
                logger.warning(f"[GrandAlignDebug] Masked unification error: {repr(e)}")

        grand_extra_loss = torch.zeros((), device=base_loss.device)
        weight = getattr(self.args, 'grand_alignment_loss_weight', 0.5)
        if not weight:
            total_loss = base_loss
            if return_outputs:
                return total_loss, outputs
            return total_loss
        attempted_phrase_total = 0
        matched_phrase_total = 0
        crop_total = 0
        matched_crop_total = 0
        if grand_mask is not None and labels is not None and grand_bboxes is not None and grand_image_paths is not None and grand_dense_labels is not None and grand_dense_captions is not None:
            grand_mask = grand_mask.bool()
            if grand_mask.any():
                # Obtain last hidden states
                hidden_states = None
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    if self.args.address_layer == 'first_layer':
                        hidden_states = outputs.hidden_states[1]
                    elif self.args.address_layer == 'mid_layer':
                        mid_idx = (len(outputs.hidden_states) - 1) // 2
                        hidden_states = outputs.hidden_states[mid_idx]
                    elif self.args.address_layer == 'last_layer':
                        hidden_states = outputs.hidden_states[-1]
                    else:
                        hidden_states = outputs.hidden_states[-1]
                        logger.warning(f"[GrandAlign] Unknown address_layer {self.args.address_layer}, defaulting to last_layer.")
                if hidden_states is None:
                    logger.warning("[GrandAlignDebug] hidden_states not found in outputs; skipping GranD loss.")
                elif hidden_states is not None:
                    per_sample_losses = []
                    for b_idx, is_grand in enumerate(grand_mask):
                        if not is_grand:
                            continue
                        sample_bboxes = grand_bboxes[b_idx] if b_idx < len(grand_bboxes) else []
                        image_path = grand_image_paths[b_idx] if b_idx < len(grand_image_paths) else None
                        if not sample_bboxes or image_path is None:
                            continue
                        label_row = labels[b_idx]
                        token_mask = (label_row != IGNORE_INDEX) & (label_row != -100)
                        if token_mask.sum() == 0:
                            continue
                        generated_token_ids = label_row[token_mask].tolist()
                        generated_indices = torch.nonzero(token_mask, as_tuple=False).squeeze(-1).tolist()
                        try:
                            img = Image.open(image_path).convert('RGB')
                        except Exception as e:
                            logger.warning(f"[GrandAlignDebug] skip_sample b={b_idx} path={image_path} reason=image_open_fail error={repr(e)}")
                            continue
                        crops = []
                        for (l, t, r, b) in sample_bboxes:
                            try:
                                crops.append(img.crop((l, t, r, b)))
                            except Exception:
                                continue
                        if not crops:
                            continue
                        crop_total += len(crops)
                        with torch.no_grad():
                            patch_embeds = self.patch_embedder(crops)
                        try:
                            base_model = model.get_model() if hasattr(model, 'get_model') else model
                            align_enc = getattr(base_model, 'alignment_encoder', None)
                        except Exception:
                            align_enc = None
                        if align_enc is None:
                            logger.warning("Alignment encoder not found; skipping GranD loss.")
                            continue
                        patch_embeds = patch_embeds.to(hidden_states.device)
                        crop_losses = []
                        phrases = grand_dense_labels[b_idx] if b_idx < len(grand_dense_labels) else []
                        # Batch-match phrases to text spans, then batch project with alignment encoder
                        matched_text_embeds = []
                        matched_crop_indices = []
                        for crop_i, phrase in enumerate(phrases):
                            phrase = phrase.strip()
                            if not phrase:
                                continue
                            attempted_phrase_total += 1
                            if crop_i >= patch_embeds.size(0):
                                continue
                            if self.tokenizer is None:
                                continue
                            variant_tokens = self.tokenizer(phrase, add_special_tokens=False).input_ids
                            found = False
                            # Search for phrase tokens in generated tokens
                            for start in range(len(generated_token_ids) - len(variant_tokens) + 1):
                                if generated_token_ids[start:start+len(variant_tokens)] == variant_tokens:
                                    orig_span = generated_indices[start:start+len(variant_tokens)]
                                    span_embeds = hidden_states[b_idx][orig_span]
                                    if span_embeds.numel() > 0:
                                        matched_text = None
                                        if self.args.text_token_pool == 'last':
                                            matched_text = span_embeds[-1]
                                        elif self.args.text_token_pool == 'mean':
                                            matched_text = span_embeds.mean(dim=0)
                                        elif self.args.text_token_pool == 'attn':
                                            # Ensure text_pooler exists and matches hidden_states device & dtype
                                            if not hasattr(self, 'text_pooler'):
                                                raise RuntimeError("text_pooler not initialized for attn pooling")

                                            try:
                                                self.text_pooler = self.text_pooler.to(
                                                    device=hidden_states.device,
                                                    dtype=hidden_states.dtype,
                                                )
                                            except Exception as e:
                                                logger.warning(
                                                    f"[GrandAlignDebug] Failed moving text_pooler to device/dtype: {repr(e)}"
                                                )

                                            # span_embeds inherits dtype/device from hidden_states
                                            span_embeds = span_embeds.unsqueeze(0)  # [1, L, D]
                                            pooled_embed, _ = self.text_pooler(span_embeds)  # [1, D]
                                            matched_text = pooled_embed.squeeze(0)
                                        else:
                                            matched_text = span_embeds.mean(dim=0)
                                        matched_text_embeds.append(matched_text)
                                        matched_crop_indices.append(crop_i)
                                        found = True
                                    break
                            if found:
                                matched_phrase_total += 1
                                matched_crop_total += 1

                        if matched_text_embeds:
                            text_batch = torch.stack(matched_text_embeds, dim=0).to(patch_embeds.device)
                            # Align text to image
                            try:
                                projected_text_batch = align_enc(text_batch)
                            except Exception:
                                projected_text_batch = align_enc(text_batch).squeeze(0)
                            # Compute cosine similarity batch-wise against corresponding image vectors
                            proj_norm = F.normalize(projected_text_batch, dim=-1)
                            img_vecs = patch_embeds[matched_crop_indices]
                            img_norm = F.normalize(img_vecs, dim=-1)
                            sims = (proj_norm * img_norm).sum(dim=-1)
                            crop_losses = 1 - sims  # tensor of shape [M]
                        
                        if isinstance(crop_losses, torch.Tensor) and crop_losses.numel() > 0:
                            per_sample_losses.append(crop_losses.mean())
                    if per_sample_losses:
                        grand_extra_loss = torch.stack(per_sample_losses).mean()
                        print(f"[GrandAlignDebug] grand_loss={grand_extra_loss.item():.6f}")

        total_loss = base_loss + (grand_extra_loss * weight)
        if return_outputs:
            return total_loss, outputs
        return total_loss