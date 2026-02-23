import os
import json
import pandas as pd
import argparse
from datetime import datetime, timedelta


def tabulate_results(eval_dir, experiment_csv_fname, out_pivot_fname, out_all_results_fname, days_back=None):
    exists = os.path.exists(eval_dir)
    if not exists:
        raise ValueError(f"eval_dir {eval_dir} does not exist")

    print(f"eval_dir: {eval_dir}")

    evals_order = [
        ## llava
        # 'vqav2',
        'gqa',
        'vizwiz',
        'scienceqa',
        'textvqa',
        'pope',
        'mme',
        'mmbench_en',
        'mmbench_cn',
        'seed',
        # 'llava_w',
        # 'mmvet', # submission
        ## Addtl
        'mmmu',
        'mathvista',
        'ai2d',
        'chartqa',
        # 'docvqa', # submission
        # 'infovqa', # submission
        # 'stvqa', # submission
        'ocrbench',
        'mmstar',
        'realworldqa',
        # 'qbench',
        'blink',
        'mmvp',
        'vstar',
        'ade',
        # 'omni',
        'coco'
        # 'synthdog', # seems broken?
    ]

    evals_col_overrides = {
        'scienceqa': '100x_multimodal_acc',
        'mme': "Perception",
        'mmbench_en': "100x_circular_accuracy",
        'mmbench_cn': "100x_circular_accuracy",
        'seed': "100x_accuracy",
        'mmmu': "100x_accuracy",
        'mathvista': "100x_accuracy",
        'ocrbench': "total_accuracy['accuracy']",
        'qbench': "100x_accuracy",
        'blink': "100x_accuracy",
        'ade': "100x_accuracy",
        'omni': "100x_accuracy",
        'coco': "100x_accuracy",
    }

    # gather results from each eval
    dfs = []
    missing_reasons = []
    eval_models = {}
    for eval_name in evals_order:
        results_path = os.path.join(eval_dir, eval_name, experiment_csv_fname)
        if not os.path.exists(results_path):
            print(f"Skipping {eval_name} as no results file found")
            continue

        try:
            df = pd.read_csv(results_path)
        except Exception as e:
            print(f"Error reading {results_path}: {e}")
            raise e

        # Filter by date if specified
        if days_back is not None:
            if "time" in df.columns:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                df["time_parsed"] = pd.to_datetime(df["time"])
                df = df[df["time_parsed"] >= cutoff_date]
                df = df.drop("time_parsed", axis=1)
                if len(df) == 0:
                    print(f"Skipping {eval_name} as no results within last {days_back} days")
                    continue
            else:
                missing_reasons.append(
                    f"{eval_name}: missing 'time' column so --days filter could not be applied"
                )

        if eval_name in evals_col_overrides:
            override = evals_col_overrides[eval_name]
            if override.startswith("100x_"):
                override = override[5:]
                if override in df.columns:
                    df["accuracy"] = df[override] * 100
                else:
                    missing_reasons.append(
                        f"{eval_name}: missing expected column '{override}' for 100x_ override"
                    )
            elif override == "total_accuracy['accuracy']":
                if "total_accuracy" in df.columns:
                    df["accuracy"] = df["total_accuracy"].apply(
                        lambda x: json.loads(x.replace("'", '"'))["accuracy"]
                    )
                else:
                    missing_reasons.append(
                        f"{eval_name}: missing expected column 'total_accuracy' for override"
                    )
            else:
                if override in df.columns:
                    df["accuracy"] = df[override]
                else:
                    missing_reasons.append(
                        f"{eval_name}: missing expected column '{override}' for override"
                    )

        df["eval_name"] = eval_name

        df = df.sort_values("time")
        df = df.drop_duplicates("model", keep="last")

        if "model" in df.columns:
            eval_models[eval_name] = set(df["model"].astype(str).tolist())
        else:
            missing_reasons.append(f"{eval_name}: missing 'model' column")

        accuracy_name = "accuracy" if "accuracy" in df.columns else "accurcay"
        if accuracy_name not in df.columns:
            missing_reasons.append(
                f"{eval_name}: missing accuracy column (expected 'accuracy' or 'accurcay')"
            )
            continue

        null_accuracy = df[accuracy_name].isna().sum()
        if null_accuracy > 0:
            missing_reasons.append(
                f"{eval_name}: {null_accuracy} rows have empty accuracy values"
            )

        df = df[["time", "eval_name", "model", accuracy_name]]
        dfs.append(df)

    if len(dfs) == 0:
        raise ValueError("No evaluation results were collected. Check eval_dir and filters.")

    all_results = pd.concat(dfs)
    all_results.sort_values("time").to_csv(out_all_results_fname, index=False)
    print(f"Saved all results to {out_all_results_fname}")

    pivot = all_results.pivot(index="model", columns="eval_name", values="accuracy")
    pivot = pivot[evals_order]

    # Report missing cells per eval/model.
    all_models = set(pivot.index.astype(str).tolist())
    for eval_name in evals_order:
        if eval_name not in pivot.columns:
            missing_reasons.append(f"{eval_name}: no column in pivot (no rows collected)")
            continue
        missing_models = pivot[pivot[eval_name].isna()].index.astype(str).tolist()
        if len(missing_models) > 0:
            missing_reasons.append(
                f"{eval_name}: missing values for models: {', '.join(missing_models)}"
            )

        if eval_name in eval_models:
            unseen = sorted(all_models - eval_models[eval_name])
            if len(unseen) > 0:
                missing_reasons.append(
                    f"{eval_name}: models absent from CSV: {', '.join(unseen)}"
                )

    if len(missing_reasons) > 0:
        print("\nMissing/empty value diagnostics:")
        for reason in missing_reasons:
            print(f"- {reason}")

    # if .xlsx, to_excel, else to_csv
    if out_pivot_fname.endswith(".xlsx"):
        pivot.to_excel(out_pivot_fname)
        print(f"Saved excel file pivot to {out_pivot_fname}")
    else:
        pivot.to_csv(out_pivot_fname)
        print(f"Saved csv file pivot to {out_pivot_fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabulate experiment results")
    parser.add_argument("--eval_dir", type=str, default="eval", help="Directory containing evaluation results")
    parser.add_argument("--experiment_csv", type=str, default="experiments.csv", help="Name of the CSV file containing experiment results")
    parser.add_argument("--out_pivot", type=str, default="pivot.xlsx", help="Name of the output file (Excel or CSV)")
    parser.add_argument("--out_all_results", type=str, default="all_results.csv", help="Name of the CSV file to save all results")
    parser.add_argument("--days", type=int, default=4, help="Only include results from the last N days (default: all)")

    args = parser.parse_args()

    tabulate_results(args.eval_dir, args.experiment_csv, args.out_pivot, args.out_all_results, args.days)