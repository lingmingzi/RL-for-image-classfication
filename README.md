# RL-Driven Data Augmentation for CIFAR Classification

本项目实现了一个可直接复现实验框架：用 PPO 控制器动态选择 CIFAR 图像增强子策略，并与 `NoAug / Manual / RandAugment` 做对照。

## 1. 实验覆盖内容

- 数据集：`CIFAR-10`（主实验）
- 对照基线：`noaug`、`manual`、`randaugment`
- RL 方法：离散 sub-policy 选择（12 个策略），每个 episode 动态调度
- 奖励函数（默认）：
  - `R = 1.0 * r_acc + 0.5 * r_loss + 0.1 * r_complex`
  - 支持可选项：`r_robust`（CIFAR-C）与 `r_class_balance`
- 扩展评测：
  - `CIFAR-C` 鲁棒性
  - 将学到策略迁移到 `CIFAR-100`

## 2. 环境安装

```bash
pip install -r requirements.txt
```

## 3. 运行基线实验

### 3.1 NoAug

```bash
python scripts/train_baselines.py --dataset cifar10 --model resnet18 --policy noaug --epochs 100 --seed 42
```

### 3.2 Manual Aug

```bash
python scripts/train_baselines.py --dataset cifar10 --model resnet18 --policy manual --epochs 100 --seed 42
```

### 3.3 RandAugment

```bash
python scripts/train_baselines.py --dataset cifar10 --model resnet18 --policy randaugment --epochs 100 --seed 42
```

## 4. 运行 RL 动态增强实验

```bash
python scripts/train_rl.py \
  --model resnet18 \
  --epochs 120 \
  --episode_epochs 1 \
  --ppo_update_every 8 \
  --reward_w_acc 1.0 \
  --reward_w_loss 0.5 \
  --reward_w_complexity 0.1 \
  --seed 42
```

输出目录示例：`outputs/rl/cifar10_resnet18_rl_seed42/`

关键文件：

- `rl_history.csv`：每个 episode 的动作、奖励拆分、验证指标
- `summary.json`：最终 test 指标、最佳策略、动作分布
- `best_model.pt`：模型与 controller 检查点

## 5. 鲁棒性评测（CIFAR-C）

先准备 CIFAR-C（目录中至少包含 `labels.npy` 与各 corruption 文件）。

```bash
python scripts/eval_cifar_c.py \
  --ckpt outputs/rl/cifar10_resnet18_rl_seed42/best_model.pt \
  --model resnet18 \
  --cifar_c_dir /path/to/CIFAR-10-C
```

可在 RL 训练时直接加入鲁棒性奖励：

```bash
python scripts/train_rl.py \
  --reward_w_robust 0.2 \
  --cifar_c_dir /path/to/CIFAR-10-C \
  --cifar_c_corruption gaussian_noise.npy \
  --cifar_c_eval_freq 5
```

## 6. 迁移到 CIFAR-100

将 CIFAR-10 学到的最佳策略直接迁移：

```bash
python scripts/transfer_cifar100.py \
  --policy_summary outputs/rl/cifar10_resnet18_rl_seed42/summary.json \
  --model resnet18 \
  --epochs 120
```

## 7. 建议的最小实验矩阵

- 模型：`resnet18`、`resnet50`
- 方法：`noaug`、`manual`、`randaugment`、`rl`
- 随机种子：`42, 52, 62`（至少 3 个）

推荐统计表字段：

- `test_acc`
- `CIFAR-C mean corruption error`
- `ECE`
- `min_class_acc`
- `训练总时长 / GPU 小时`

## 8. 消融建议

- Reward 消融：
  - 去掉 `r_loss`（`--reward_w_loss 0`）
  - 去掉 `r_complexity`（`--reward_w_complexity 0`）
  - 开启/关闭 `r_robust`
- 控制器更新频率：`--ppo_update_every {4, 8, 16}`
- Episode 长度：`--episode_epochs {1, 2, 3}`
- 模型迁移：`resnet18` 训练策略用于 `resnet50`

## 9. 可复现性说明

- 已内置随机种子设置（Python / NumPy / PyTorch）
- 每次运行都会保存完整配置与训练轨迹
- 建议每个实验独立输出目录，避免覆盖

## 10. 一键批量实验

新增脚本：`scripts/run_experiments.py`

默认行为：同时跑基线和 RL（模型 `resnet18,resnet50`，种子 `42,52,62`）。

```bash
python scripts/run_experiments.py --data_dir ./data --outputs_root ./outputs
```

常见用法：

- 仅跑基线：

```bash
python scripts/run_experiments.py --run_baselines --models resnet18 --seeds 42,52,62
```

- 仅跑 RL：

```bash
python scripts/run_experiments.py --run_rl --models resnet18 --seeds 42,52,62
```

- 跑迁移（要求已有 RL summary）：

```bash
python scripts/run_experiments.py --run_transfer --models resnet18 --seeds 42,52,62
```

- 只查看将执行命令（不实际运行）：

```bash
python scripts/run_experiments.py --dry_run
```

脚本会输出：

- `outputs/reports/all_summaries.csv`
- `outputs/reports/run_report.json`

## 11. 绘图脚本

新增脚本：`scripts/plot_results.py`

```bash
python scripts/plot_results.py --outputs_root ./outputs --fig_dir ./outputs/figures
```

可选参数（美观与可读性）：

- `--smooth_window 7`：RL 曲线平滑窗口
- `--top_n_actions 12`：动作分布图显示前 N 个动作

输出内容：

- `summary_table.csv`：汇总所有 summary
- `main_comparison_test_acc.png`：主方法对比柱状图
- `rl_progress_curve.png`：RL 奖励与验证精度曲线
- `rl_action_distribution.png`：策略选择频率
- `rl_reward_components.png`：奖励分量曲线

## 12. 统计显著性脚本

新增脚本：`scripts/stat_significance.py`

```bash
python scripts/stat_significance.py \
  --outputs_root ./outputs \
  --report_dir ./outputs/reports \
  --dataset cifar10 \
  --methods noaug,manual,randaugment,rl \
  --metrics test_acc,test_ece,test_min_class_acc
```

输出文件：

- `significance_descriptive.csv`：各方法描述统计（均值、标准差、95%CI）
- `significance_tests.csv`：两两 Welch t-test、效应量 Cohen's d
- `significance_report.md`：可直接放入实验报告的 markdown 表格

可选：Bonferroni 校正

```bash
python scripts/stat_significance.py --outputs_root ./outputs --bonferroni
```

## 13. 一键脚本新增统计阶段

一键脚本已支持 `--run_stats`：

```bash
python scripts/run_experiments.py --run_stats --outputs_root ./outputs
```

也可以串起来：

```bash
python scripts/run_experiments.py --run_plot --run_stats --outputs_root ./outputs
```

## 14. 新增实用参数

为了方便批量实验，以下脚本已新增参数：

- `scripts/train_baselines.py`
  - `--val_size`
  - `--run_name`
- `scripts/train_rl.py`
  - `--val_size`
  - `--run_name`
  - RL episode 数改为 `ceil(epochs / episode_epochs)`，避免末尾 epoch 被跳过
  - 额外保存 `action_distribution.csv`
- `scripts/transfer_cifar100.py`
  - `--val_size`
  - `--run_name`
