import argparse
import glob
import json
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_COLUMNS = [
    'summary_path',
    'run_name',
    'dataset',
    'model',
    'method',
    'best_val_acc',
    'test_acc',
    'test_top5',
    'test_ece',
    'test_min_class_acc',
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot experiment results and RL diagnostics")
    parser.add_argument("--outputs_root", type=str, default="./outputs")
    parser.add_argument("--fig_dir", type=str, default="./outputs/figures")
    parser.add_argument("--rl_history", type=str, default="", help="Optional: explicit rl_history.csv path")
    parser.add_argument("--top_n_actions", type=int, default=12)
    parser.add_argument("--smooth_window", type=int, default=7)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.facecolor": "#f7f8fa",
            "axes.facecolor": "#fbfcff",
        }
    )


def find_summary_files(outputs_root: str) -> List[str]:
    patterns = [
        os.path.join(outputs_root, "baselines", "**", "summary.json"),
        os.path.join(outputs_root, "rl", "**", "summary.json"),
        os.path.join(outputs_root, "transfer", "**", "summary.json"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    if not files:
        files.extend(glob.glob(os.path.join(outputs_root, "**", "summary.json"), recursive=True))
    return sorted(set(files))


def infer_method(path: str) -> str:
    norm = path.replace('\\', '/').lower()
    if '/rl/' in norm:
        return 'rl'
    if '/transfer/' in norm:
        return 'transfer'
    run_name = os.path.basename(os.path.dirname(path)).lower()
    for method in ['noaug', 'manual', 'randaugment']:
        if method in run_name:
            return method
    return 'baseline'


def load_summary_table(outputs_root: str) -> pd.DataFrame:
    records = []
    for fp in find_summary_files(outputs_root):
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        run_name = data.get('run_name', os.path.basename(os.path.dirname(fp)))
        parts = run_name.split('_')
        dataset = parts[0] if parts else ''
        model = ''
        for p in parts:
            if p in ['resnet18', 'resnet50']:
                model = p
                break
        records.append(
            {
                'summary_path': fp,
                'run_name': run_name,
                'dataset': dataset,
                'model': model,
                'method': infer_method(fp),
                'best_val_acc': data.get('best_val_acc'),
                'test_acc': data.get('test_acc'),
                'test_top5': data.get('test_top5'),
                'test_ece': data.get('test_ece'),
                'test_min_class_acc': data.get('test_min_class_acc'),
            }
        )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        fallback_csv = os.path.join(outputs_root, 'reports', 'all_summaries.csv')
        if os.path.exists(fallback_csv):
            df = pd.read_csv(fallback_csv)

    for c in BASE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    if 'run_name' in df.columns:
        missing_dataset = df['dataset'].isna() | (df['dataset'].astype(str).str.len() == 0)
        df.loc[missing_dataset, 'dataset'] = df.loc[missing_dataset, 'run_name'].apply(
            lambda x: str(x).split('_')[0] if str(x) else ''
        )

    if 'summary_path' in df.columns and 'run_name' in df.columns:
        missing_method = df['method'].isna() | (df['method'].astype(str).str.len() == 0)
        df.loc[missing_method, 'method'] = df.loc[missing_method].apply(
            lambda r: infer_method(str(r.get('summary_path', ''))), axis=1
        )

    return df[BASE_COLUMNS]


def _extract_seed(run_name: str) -> Optional[int]:
    lower = str(run_name).lower()
    marker = "seed"
    idx = lower.rfind(marker)
    if idx < 0:
        return None
    tail = lower[idx + len(marker):]
    digits = ""
    for ch in tail:
        if ch.isdigit():
            digits += ch
        else:
            break
    if not digits:
        return None
    return int(digits)


def plot_main_comparison(df: pd.DataFrame, fig_dir: str) -> Optional[str]:
    if df.empty:
        return None

    main_df = df[df['method'].isin(['noaug', 'manual', 'randaugment', 'rl'])].copy()
    if main_df.empty:
        return None

    main_df = main_df.dropna(subset=['test_acc'])
    if main_df.empty:
        return None

    main_df['model'] = main_df['model'].fillna('unknown')
    main_df['seed'] = main_df['run_name'].apply(_extract_seed)

    methods_order = ['noaug', 'manual', 'randaugment', 'rl']
    main_df['method'] = pd.Categorical(main_df['method'], categories=methods_order, ordered=True)
    grouped = (
        main_df.groupby(['model', 'method'], as_index=False)
        .agg(mean_acc=('test_acc', 'mean'), std_acc=('test_acc', 'std'), n=('test_acc', 'count'))
    )
    grouped['std_acc'] = grouped['std_acc'].fillna(0.0)
    grouped = grouped.sort_values(['model', 'method'])

    models = sorted(grouped['model'].unique())
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6.5 * n_models, 5.3), squeeze=False)
    palette = {
        'noaug': '#7d8597',
        'manual': '#2a9d8f',
        'randaugment': '#f4a261',
        'rl': '#e76f51',
    }

    for idx, model in enumerate(models):
        ax = axes[0][idx]
        part = grouped[grouped['model'] == model].copy()
        x = np.arange(len(part))
        colors = [palette.get(str(m), '#4c78a8') for m in part['method']]

        ax.bar(
            x,
            part['mean_acc'],
            yerr=part['std_acc'],
            color=colors,
            alpha=0.92,
            linewidth=0,
            capsize=5,
            error_kw={'elinewidth': 1.2, 'ecolor': '#4a4e69'},
        )

        raw_part = main_df[main_df['model'] == model].copy()
        for i, method in enumerate(part['method']):
            scatter_vals = raw_part[raw_part['method'] == method]['test_acc'].values
            if len(scatter_vals) == 0:
                continue
            jitter = np.random.uniform(-0.12, 0.12, size=len(scatter_vals))
            ax.scatter(np.full(len(scatter_vals), i) + jitter, scatter_vals, color='#1d3557', alpha=0.65, s=18, zorder=3)

        ax.set_title(f'{model.upper()} test accuracy', fontweight='bold')
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel('accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(part['method'])
        ax.axhline(y=0.5, color='#adb5bd', linewidth=0.8, linestyle=':')

        for i, row in enumerate(part.itertuples(index=False)):
            ax.text(i, float(row.mean_acc) + 0.02, f"{row.mean_acc:.3f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(fig_dir, 'main_comparison_test_acc.png')
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _pick_rl_history(outputs_root: str, explicit_path: str) -> Optional[str]:
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    candidates = glob.glob(os.path.join(outputs_root, 'rl', '**', 'rl_history.csv'), recursive=True)
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _rolling_mean(values: pd.Series, window: int) -> pd.Series:
    win = max(1, int(window))
    return values.rolling(win, min_periods=1).mean()


def plot_rl_diagnostics(outputs_root: str, fig_dir: str, explicit_path: str, top_n_actions: int, smooth_window: int) -> List[str]:
    rl_path = _pick_rl_history(outputs_root, explicit_path)
    if rl_path is None:
        return []

    df = pd.read_csv(rl_path)
    if df.empty:
        return []

    saved = []

    fig, ax1 = plt.subplots(figsize=(9, 5.4))
    ax1.plot(df['episode'], df['val_acc'], label='val_acc (raw)', color='#2a9d8f', alpha=0.28, linewidth=1.1)
    ax1.plot(df['episode'], _rolling_mean(df['val_acc'], smooth_window), label='val_acc (smooth)', color='#2a9d8f', linewidth=2.0)
    ax1.plot(df['episode'], df['reward'], label='reward (raw)', color='#e76f51', alpha=0.24, linewidth=1.0)
    ax1.plot(df['episode'], _rolling_mean(df['reward'], smooth_window), label='reward (smooth)', color='#e76f51', linewidth=2.0)
    ax1.set_xlabel('episode')
    ax1.set_ylabel('value')
    ax1.set_title('RL training progress', fontweight='bold')
    ax1.legend(ncol=2)
    ax1.axhline(y=0.0, color='#6c757d', linewidth=0.8, linestyle=':')
    plt.tight_layout()
    out1 = os.path.join(fig_dir, 'rl_progress_curve.png')
    plt.savefig(out1, dpi=180)
    plt.close(fig)
    saved.append(out1)

    action_counts = df['action'].value_counts().head(top_n_actions)
    fig, ax2 = plt.subplots(figsize=(9, 5.8))
    action_counts = action_counts.sort_values(ascending=True)
    y_pos = np.arange(len(action_counts))
    bars = ax2.barh(y_pos, action_counts.values, color='#457b9d', alpha=0.9)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(action_counts.index)
    ax2.set_title('Action selection frequency', fontweight='bold')
    ax2.set_xlabel('count')
    for b in bars:
        w = b.get_width()
        ax2.text(w + 0.2, b.get_y() + b.get_height() / 2, f"{int(w)}", va='center', fontsize=9)
    plt.tight_layout()
    out2 = os.path.join(fig_dir, 'rl_action_distribution.png')
    plt.savefig(out2, dpi=180)
    plt.close(fig)
    saved.append(out2)

    reward_cols = [c for c in ['r_acc', 'r_loss', 'r_complex', 'r_robust', 'r_class_balance'] if c in df.columns]
    if reward_cols:
        fig, ax3 = plt.subplots(figsize=(9.5, 5.5))
        colors = ['#2a9d8f', '#e76f51', '#264653', '#8ab17d', '#6d597a']
        for c in reward_cols:
            color = colors[reward_cols.index(c) % len(colors)]
            ax3.plot(df['episode'], _rolling_mean(df[c], smooth_window), label=f"{c} (smooth)", color=color, linewidth=1.9)
        ax3.set_title('Reward components by episode', fontweight='bold')
        ax3.set_xlabel('episode')
        ax3.set_ylabel('component value')
        ax3.axhline(y=0.0, color='#6c757d', linewidth=0.8, linestyle=':')
        ax3.legend()
        plt.tight_layout()
        out3 = os.path.join(fig_dir, 'rl_reward_components.png')
        plt.savefig(out3, dpi=180)
        plt.close(fig)
        saved.append(out3)

    return saved


def main():
    args = parse_args()
    setup_plot_style()
    ensure_dir(args.fig_dir)

    summary_df = load_summary_table(args.outputs_root)
    summary_csv = os.path.join(args.fig_dir, 'summary_table.csv')
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')

    saved_plots = []
    comp_plot = plot_main_comparison(summary_df, args.fig_dir)
    if comp_plot:
        saved_plots.append(comp_plot)

    saved_plots.extend(plot_rl_diagnostics(args.outputs_root, args.fig_dir, args.rl_history, args.top_n_actions, args.smooth_window))

    report = {
        'summary_csv': summary_csv,
        'saved_plots': saved_plots,
        'num_runs': int(len(summary_df)),
        'message': 'No summary.json found under outputs_root' if summary_df.empty else '',
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
