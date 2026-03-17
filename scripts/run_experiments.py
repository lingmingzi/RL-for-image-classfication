import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def to_project_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(PROJECT_ROOT, path_value))


def script_path(name: str) -> str:
    return os.path.join(SCRIPT_DIR, name)


def parse_args():
    parser = argparse.ArgumentParser(description="One-click experiment launcher for CIFAR augmentation study")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--outputs_root", type=str, default="./outputs")
    parser.add_argument("--models", type=str, default="resnet18,resnet50")
    parser.add_argument("--seeds", type=str, default="42,52,62")
    parser.add_argument("--baseline_policies", type=str, default="noaug,manual,randaugment")

    parser.add_argument("--baseline_epochs", type=int, default=100)
    parser.add_argument("--rl_epochs", type=int, default=120)
    parser.add_argument("--episode_epochs", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val_size", type=int, default=5000)

    parser.add_argument("--reward_w_acc", type=float, default=1.0)
    parser.add_argument("--reward_w_loss", type=float, default=0.5)
    parser.add_argument("--reward_w_complexity", type=float, default=0.1)

    parser.add_argument("--run_baselines", action="store_true")
    parser.add_argument("--run_rl", action="store_true")
    parser.add_argument("--run_transfer", action="store_true")
    parser.add_argument("--run_plot", action="store_true")
    parser.add_argument("--run_stats", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def split_csv_int(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def split_csv_str(s: str) -> List[str]:
    return [x.strip() for x in s.split(',') if x.strip()]


def run_cmd(cmd: List[str], dry_run: bool = False) -> int:
    print(' '.join(cmd))
    if dry_run:
        return 0
    env = os.environ.copy()
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = PROJECT_ROOT if not existing else PROJECT_ROOT + os.pathsep + existing
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    return int(proc.returncode)


def find_rl_summary(outputs_root: str, model: str, seed: int) -> str:
    pattern = os.path.join(outputs_root, 'rl', f'cifar10_{model}_rl_seed{seed}', 'summary.json')
    if os.path.exists(pattern):
        return pattern
    fallback = glob.glob(os.path.join(outputs_root, 'rl', '**', 'summary.json'), recursive=True)
    fallback = sorted(fallback, key=lambda p: os.path.getmtime(p), reverse=True)
    return fallback[0] if fallback else ''


def extract_model(run_name: str) -> str:
    for token in str(run_name).lower().split('_'):
        if token in ['resnet18', 'resnet50']:
            return token
    return 'unknown'


def extract_seed(run_name: str):
    text = str(run_name).lower()
    idx = text.rfind('seed')
    if idx < 0:
        return ''
    suffix = text[idx + 4:]
    digits = ''
    for ch in suffix:
        if ch.isdigit():
            digits += ch
        else:
            break
    return digits if digits else ''


def infer_method(path: str, run_name: str) -> str:
    norm = str(path).replace('\\', '/').lower()
    if '/rl/' in norm:
        return 'rl'
    if '/transfer/' in norm:
        return 'transfer'
    rn = str(run_name).lower()
    for method in ['noaug', 'manual', 'randaugment']:
        if method in rn:
            return method
    return 'baseline'


def aggregate_summaries(outputs_root: str, report_dir: str) -> str:
    summary_paths = glob.glob(os.path.join(outputs_root, '**', 'summary.json'), recursive=True)
    rows = []
    for path in summary_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        run_name = data.get('run_name', os.path.basename(os.path.dirname(path)))
        dataset = str(run_name).split('_')[0] if run_name else ''
        rows.append(
            {
                'summary_path': path,
                'run_name': run_name,
            'dataset': dataset,
            'model': extract_model(run_name),
            'seed': extract_seed(run_name),
            'method': infer_method(path, run_name),
                'best_val_acc': data.get('best_val_acc'),
                'test_acc': data.get('test_acc'),
                'test_top5': data.get('test_top5'),
                'test_ece': data.get('test_ece'),
                'test_min_class_acc': data.get('test_min_class_acc'),
                'best_policy': data.get('best_policy', data.get('source_policy', '')),
            }
        )

    os.makedirs(report_dir, exist_ok=True)
    out_csv = os.path.join(report_dir, 'all_summaries.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'summary_path',
                'run_name',
                'dataset',
                'model',
                'seed',
                'method',
                'best_val_acc',
                'test_acc',
                'test_top5',
                'test_ece',
                'test_min_class_acc',
                'best_policy',
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return out_csv


def main():
    args = parse_args()

    args.data_dir = to_project_path(args.data_dir)
    args.outputs_root = to_project_path(args.outputs_root)

    models = split_csv_str(args.models)
    seeds = split_csv_int(args.seeds)
    policies = split_csv_str(args.baseline_policies)

    run_baselines = args.run_baselines or (not args.run_rl and not args.run_transfer and not args.run_plot and not args.run_stats)
    run_rl = args.run_rl or (not args.run_baselines and not args.run_transfer and not args.run_plot and not args.run_stats)

    failures: List[Dict[str, str]] = []

    if run_baselines:
        for model in models:
            for seed in seeds:
                for policy in policies:
                    cmd = [
                        sys.executable,
                        script_path('train_baselines.py'),
                        '--dataset',
                        'cifar10',
                        '--model',
                        model,
                        '--policy',
                        policy,
                        '--data_dir',
                        args.data_dir,
                        '--output_dir',
                        os.path.join(args.outputs_root, 'baselines'),
                        '--epochs',
                        str(args.baseline_epochs),
                        '--batch_size',
                        str(args.batch_size),
                        '--num_workers',
                        str(args.num_workers),
                        '--seed',
                        str(seed),
                        '--device',
                        args.device,
                        '--val_size',
                        str(args.val_size),
                    ]
                    code = run_cmd(cmd, dry_run=args.dry_run)
                    if code != 0:
                        failures.append({'stage': 'baseline', 'model': model, 'seed': str(seed), 'policy': policy})

    if run_rl:
        for model in models:
            for seed in seeds:
                cmd = [
                    sys.executable,
                    script_path('train_rl.py'),
                    '--model',
                    model,
                    '--data_dir',
                    args.data_dir,
                    '--output_dir',
                    os.path.join(args.outputs_root, 'rl'),
                    '--epochs',
                    str(args.rl_epochs),
                    '--episode_epochs',
                    str(args.episode_epochs),
                    '--batch_size',
                    str(args.batch_size),
                    '--num_workers',
                    str(args.num_workers),
                    '--seed',
                    str(seed),
                    '--device',
                    args.device,
                    '--reward_w_acc',
                    str(args.reward_w_acc),
                    '--reward_w_loss',
                    str(args.reward_w_loss),
                    '--reward_w_complexity',
                    str(args.reward_w_complexity),
                    '--val_size',
                    str(args.val_size),
                ]
                code = run_cmd(cmd, dry_run=args.dry_run)
                if code != 0:
                    failures.append({'stage': 'rl', 'model': model, 'seed': str(seed), 'policy': ''})

    if args.run_transfer:
        for model in models:
            for seed in seeds:
                policy_summary = find_rl_summary(args.outputs_root, model, seed)
                if not policy_summary:
                    failures.append({'stage': 'transfer', 'model': model, 'seed': str(seed), 'policy': 'missing_rl_summary'})
                    continue
                cmd = [
                    sys.executable,
                    script_path('transfer_cifar100.py'),
                    '--policy_summary',
                    policy_summary,
                    '--model',
                    model,
                    '--data_dir',
                    args.data_dir,
                    '--output_dir',
                    os.path.join(args.outputs_root, 'transfer'),
                    '--epochs',
                    str(args.rl_epochs),
                    '--batch_size',
                    str(args.batch_size),
                    '--num_workers',
                    str(args.num_workers),
                    '--seed',
                    str(seed),
                    '--device',
                    args.device,
                    '--val_size',
                    str(args.val_size),
                ]
                code = run_cmd(cmd, dry_run=args.dry_run)
                if code != 0:
                    failures.append({'stage': 'transfer', 'model': model, 'seed': str(seed), 'policy': ''})

    report_dir = os.path.join(args.outputs_root, 'reports')
    out_csv = ''
    if not args.dry_run:
        out_csv = aggregate_summaries(args.outputs_root, report_dir)

    if args.run_plot:
        cmd = [
            sys.executable,
            script_path('plot_results.py'),
            '--outputs_root',
            args.outputs_root,
            '--fig_dir',
            os.path.join(args.outputs_root, 'figures'),
        ]
        code = run_cmd(cmd, dry_run=args.dry_run)
        if code != 0:
            failures.append({'stage': 'plot', 'model': '', 'seed': '', 'policy': ''})

    if args.run_stats:
        cmd = [
            sys.executable,
            script_path('stat_significance.py'),
            '--outputs_root',
            args.outputs_root,
            '--report_dir',
            os.path.join(args.outputs_root, 'reports'),
            '--dataset',
            'cifar10',
        ]
        if out_csv:
            cmd.extend(['--summary_csv', out_csv])
        code = run_cmd(cmd, dry_run=args.dry_run)
        if code != 0:
            failures.append({'stage': 'stats', 'model': '', 'seed': '', 'policy': ''})

    final_report = {
        'launcher_path': os.path.abspath(__file__),
        'project_root': PROJECT_ROOT,
        'warning': 'Running from .Trash path may use stale files; clone a fresh copy if errors persist.' if '.Trash-0' in PROJECT_ROOT else '',
        'timestamp': datetime.now().isoformat(),
        'dry_run': args.dry_run,
        'failures': failures,
        'summary_csv': out_csv,
        'num_failures': len(failures),
    }

    report_path = os.path.join(report_dir, 'run_report.json')
    if not args.dry_run:
        os.makedirs(report_dir, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(json.dumps(final_report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
