import argparse
import glob
import json
import os
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


DEFAULT_METHODS = ["noaug", "manual", "randaugment", "rl"]
DEFAULT_METRICS = ["test_acc", "test_ece", "test_min_class_acc"]
BASE_COLUMNS = [
    "summary_path",
    "run_name",
    "dataset",
    "model",
    "seed",
    "method",
    "best_val_acc",
    "test_acc",
    "test_top5",
    "test_ece",
    "test_min_class_acc",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run statistical significance tests for experiment summaries")
    parser.add_argument("--outputs_root", type=str, default="./outputs")
    parser.add_argument("--summary_csv", type=str, default="", help="Optional: use an existing summary table CSV")
    parser.add_argument("--report_dir", type=str, default="./outputs/reports")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--methods", type=str, default="noaug,manual,randaugment,rl")
    parser.add_argument("--metrics", type=str, default="test_acc,test_ece,test_min_class_acc")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bonferroni", action="store_true")
    return parser.parse_args()


def infer_method(path: str, run_name: str) -> str:
    norm = path.replace("\\", "/").lower()
    if "/rl/" in norm:
        return "rl"
    if "/transfer/" in norm:
        return "transfer"
    rn = run_name.lower()
    for method in DEFAULT_METHODS:
        if method in rn:
            return method
    return "baseline"


def extract_model(run_name: str) -> str:
    for token in str(run_name).lower().split("_"):
        if token in ["resnet18", "resnet50"]:
            return token
    return "unknown"


def extract_seed(run_name: str) -> Optional[int]:
    text = str(run_name).lower()
    idx = text.rfind("seed")
    if idx < 0:
        return None
    suffix = text[idx + 4 :]
    digits = ""
    for ch in suffix:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None


def find_summary_files(outputs_root: str) -> List[str]:
    files = glob.glob(os.path.join(outputs_root, "**", "summary.json"), recursive=True)
    return sorted(set(files))


def build_summary_table(outputs_root: str) -> pd.DataFrame:
    rows = []
    for fp in find_summary_files(outputs_root):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        run_name = data.get("run_name", os.path.basename(os.path.dirname(fp)))
        rows.append(
            {
                "summary_path": fp,
                "run_name": run_name,
                "dataset": str(run_name).split("_")[0] if run_name else "",
                "model": extract_model(run_name),
                "seed": extract_seed(run_name),
                "method": infer_method(fp, run_name),
                "best_val_acc": data.get("best_val_acc"),
                "test_acc": data.get("test_acc"),
                "test_top5": data.get("test_top5"),
                "test_ece": data.get("test_ece"),
                "test_min_class_acc": data.get("test_min_class_acc"),
            }
        )
    df = pd.DataFrame.from_records(rows)
    return ensure_schema(df)


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        df = pd.DataFrame()

    if "run_name" not in df.columns:
        if "summary_path" in df.columns:
            df["run_name"] = df["summary_path"].apply(lambda x: os.path.basename(os.path.dirname(str(x))))
        else:
            df["run_name"] = ""

    if "summary_path" not in df.columns:
        df["summary_path"] = ""

    if "dataset" not in df.columns:
        df["dataset"] = df["run_name"].apply(lambda x: str(x).split("_")[0] if str(x) else "")
    else:
        missing = df["dataset"].isna() | (df["dataset"].astype(str).str.len() == 0)
        df.loc[missing, "dataset"] = df.loc[missing, "run_name"].apply(lambda x: str(x).split("_")[0] if str(x) else "")

    if "model" not in df.columns:
        df["model"] = df["run_name"].apply(extract_model)
    else:
        missing = df["model"].isna() | (df["model"].astype(str).str.len() == 0)
        df.loc[missing, "model"] = df.loc[missing, "run_name"].apply(extract_model)

    if "seed" not in df.columns:
        df["seed"] = df["run_name"].apply(extract_seed)

    if "method" not in df.columns:
        df["method"] = df.apply(lambda r: infer_method(str(r.get("summary_path", "")), str(r.get("run_name", ""))), axis=1)
    else:
        missing = df["method"].isna() | (df["method"].astype(str).str.len() == 0)
        df.loc[missing, "method"] = df.loc[missing].apply(
            lambda r: infer_method(str(r.get("summary_path", "")), str(r.get("run_name", ""))), axis=1
        )

    for c in ["best_val_acc", "test_acc", "test_top5", "test_ece", "test_min_class_acc"]:
        if c not in df.columns:
            df[c] = np.nan

    return df.reindex(columns=BASE_COLUMNS)


def mean_ci95(values: np.ndarray) -> Tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if n == 1:
        return mean, mean, mean
    std = float(np.std(values, ddof=1))
    sem = std / np.sqrt(n)
    margin = 1.96 * sem
    return mean, mean - margin, mean + margin


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / max((len(a) + len(b) - 2), 1)
    if pooled <= 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def descriptive_stats(df: pd.DataFrame, methods: List[str], metrics: List[str], dataset: str) -> pd.DataFrame:
    out = []
    required = {"dataset", "method", "model"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    valid_metrics = [m for m in metrics if m in df.columns]
    if not valid_metrics:
        return pd.DataFrame()

    filtered = df[(df["dataset"] == dataset) & (df["method"].isin(methods))].copy()
    if filtered.empty:
        return pd.DataFrame()

    for model in sorted(filtered["model"].dropna().unique()):
        model_df = filtered[filtered["model"] == model]
        for method in methods:
            method_df = model_df[model_df["method"] == method]
            for metric in valid_metrics:
                vals = method_df[metric].dropna().to_numpy(dtype=float)
                mean, ci_low, ci_high = mean_ci95(vals)
                std = float(np.std(vals, ddof=1)) if len(vals) >= 2 else float(0.0 if len(vals) == 1 else np.nan)
                out.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "metric": metric,
                        "n": int(len(vals)),
                        "mean": mean,
                        "std": std,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                    }
                )
    return pd.DataFrame(out)


def pairwise_tests(df: pd.DataFrame, methods: List[str], metrics: List[str], dataset: str, alpha: float, bonferroni: bool) -> pd.DataFrame:
    out = []
    required = {"dataset", "method", "model"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    valid_metrics = [m for m in metrics if m in df.columns]
    if not valid_metrics:
        return pd.DataFrame()

    filtered = df[(df["dataset"] == dataset) & (df["method"].isin(methods))].copy()
    if filtered.empty:
        return pd.DataFrame()

    all_pairs = list(combinations(methods, 2))
    correction = len(all_pairs) if bonferroni else 1

    for model in sorted(filtered["model"].dropna().unique()):
        model_df = filtered[filtered["model"] == model]
        for metric in valid_metrics:
            for m1, m2 in all_pairs:
                a = model_df[model_df["method"] == m1][metric].dropna().to_numpy(dtype=float)
                b = model_df[model_df["method"] == m2][metric].dropna().to_numpy(dtype=float)

                if len(a) < 2 or len(b) < 2:
                    out.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "metric": metric,
                            "method_a": m1,
                            "method_b": m2,
                            "n_a": int(len(a)),
                            "n_b": int(len(b)),
                            "mean_a": float(np.mean(a)) if len(a) else np.nan,
                            "mean_b": float(np.mean(b)) if len(b) else np.nan,
                            "delta_a_minus_b": (float(np.mean(a)) - float(np.mean(b))) if len(a) and len(b) else np.nan,
                            "t_stat": np.nan,
                            "p_value": np.nan,
                            "p_value_corrected": np.nan,
                            "cohen_d": np.nan,
                            "significant": False,
                            "note": "insufficient_samples",
                        }
                    )
                    continue

                test = ttest_ind(a, b, equal_var=False)
                p_value = float(test.pvalue)
                p_corr = min(1.0, p_value * correction)
                effect = cohen_d(a, b)
                out.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "metric": metric,
                        "method_a": m1,
                        "method_b": m2,
                        "n_a": int(len(a)),
                        "n_b": int(len(b)),
                        "mean_a": float(np.mean(a)),
                        "mean_b": float(np.mean(b)),
                        "delta_a_minus_b": float(np.mean(a) - np.mean(b)),
                        "t_stat": float(test.statistic),
                        "p_value": p_value,
                        "p_value_corrected": p_corr,
                        "cohen_d": effect,
                        "significant": bool(p_corr < alpha),
                        "note": "",
                    }
                )

    return pd.DataFrame(out)


def build_markdown_report(desc_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float, bonferroni: bool) -> str:
    lines = []
    lines.append("# Statistical Significance Report")
    lines.append("")
    lines.append(f"- alpha: {alpha}")
    lines.append(f"- correction: {'bonferroni' if bonferroni else 'none'}")
    lines.append("")

    if desc_df.empty:
        lines.append("No descriptive statistics available.")
        return "\n".join(lines)

    lines.append("## Descriptive Statistics")
    lines.append("")
    lines.append("| dataset | model | method | metric | n | mean | std | ci95_low | ci95_high |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for r in desc_df.itertuples(index=False):
        lines.append(
            f"| {r.dataset} | {r.model} | {r.method} | {r.metric} | {int(r.n)} | {r.mean:.6f} | {r.std:.6f} | {r.ci95_low:.6f} | {r.ci95_high:.6f} |"
        )

    lines.append("")
    lines.append("## Pairwise Welch t-tests")
    lines.append("")
    lines.append("| dataset | model | metric | method_a | method_b | n_a | n_b | delta | p | p_corrected | cohen_d | significant | note |")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    for r in test_df.itertuples(index=False):
        p = "nan" if pd.isna(r.p_value) else f"{r.p_value:.6g}"
        pc = "nan" if pd.isna(r.p_value_corrected) else f"{r.p_value_corrected:.6g}"
        d = "nan" if pd.isna(r.cohen_d) else f"{r.cohen_d:.6f}"
        delta = "nan" if pd.isna(r.delta_a_minus_b) else f"{r.delta_a_minus_b:.6f}"
        lines.append(
            f"| {r.dataset} | {r.model} | {r.metric} | {r.method_a} | {r.method_b} | {int(r.n_a)} | {int(r.n_b)} | {delta} | {p} | {pc} | {d} | {bool(r.significant)} | {r.note} |"
        )

    return "\n".join(lines)


def main():
    args = parse_args()
    os.makedirs(args.report_dir, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    desc_path = os.path.join(args.report_dir, "significance_descriptive.csv")
    test_path = os.path.join(args.report_dir, "significance_tests.csv")
    report_md_path = os.path.join(args.report_dir, "significance_report.md")

    error_message = ""
    summary_df = pd.DataFrame()
    desc_df = pd.DataFrame()
    test_df = pd.DataFrame()

    try:
        if args.summary_csv and os.path.exists(args.summary_csv):
            summary_df = ensure_schema(pd.read_csv(args.summary_csv))
        else:
            summary_df = build_summary_table(args.outputs_root)

        desc_df = descriptive_stats(summary_df, methods=methods, metrics=metrics, dataset=args.dataset)
        test_df = pairwise_tests(
            summary_df,
            methods=methods,
            metrics=metrics,
            dataset=args.dataset,
            alpha=args.alpha,
            bonferroni=args.bonferroni,
        )
    except Exception as exc:
        error_message = str(exc)

    desc_df.to_csv(desc_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    md = build_markdown_report(desc_df, test_df, alpha=args.alpha, bonferroni=args.bonferroni)
    if error_message:
        md += "\n\n## Runtime Warning\n\n"
        md += f"Script handled an exception and returned empty/partial outputs: `{error_message}`\n"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md)

    summary = {
        "script_path": os.path.abspath(__file__),
        "rows_summary": int(len(summary_df)),
        "rows_descriptive": int(len(desc_df)),
        "rows_pairwise": int(len(test_df)),
        "descriptive_csv": desc_path,
        "pairwise_csv": test_path,
        "report_md": report_md_path,
        "error": error_message,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
