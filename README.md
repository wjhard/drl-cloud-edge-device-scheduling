# 基于深度强化学习的云-边-端异构计算资源管理调度方法

## 评审快速入口

四项核心评审材料已经按顺序放在仓库根目录，无需进入任何子文件夹：

| 顺序 | 材料 | 根目录文件 |
|---:|---|---|
| 01 | 原创承诺书 | [01_原创承诺书.pdf](01_原创承诺书.pdf) |
| 02 | 作品介绍 PPT（现有版，决赛版待更新） | [02_作品介绍PPT.pptx](02_作品介绍PPT.pptx) |
| 03 | 项目说明书 | [03_项目说明书.docx](03_项目说明书.docx) |
| 04 | 演示视频 | [04_演示视频.mp4](04_演示视频.mp4) |

本次材料更新（2026-09-17）：项目说明书为 2026-09-06 生成的 63 页决赛优化版；演示视频为 2026-09-08 录制的 1920×1080 高清版，时长约 3 分 57 秒、大小约 12.02 MB，含 AI 配音和字幕，展示真实 openEuler Docker 验证、单场景调度、任务 DAG、资源甘特图与评测结果。PPT 暂保留现有版，决赛版完成后更新同一路径。最新算法数字以项目说明书和下方“核心结果”为准。

根目录项目说明书、演示视频与 `docs/` 中对应正式副本保持一致。完整开发过程记录、参考文献核验记录和自主探索日志见 [`docs/`](docs/)；正式历史实验数据、消融实验和跨系统验证证据见 [`evaluation/results/`](evaluation/results/)；已归档的过程产物见 [`artifacts/`](artifacts/)。项目源代码模块 `env/`、`baselines/`、`policies/`、`training/`、`evaluation/`、`tests/` 和 `scripts/` 均位于仓库根目录。

本项目面向中国研究生操作系统开源创新大赛第 16 题，研究带依赖约束的任务 DAG 在云、边、端异构计算资源上的调度问题。目标是在满足任务前驱关系、通信时延和资源互斥约束的前提下，最小化整体完成时间（makespan）。

项目提供完整的环境建模、HEFT/MILP 基线、MaskablePPO 训练、残差式任务排序、Best-of-N 推理、局部搜索精修、统计检验及 openEuler/openKylin/Anolis Docker 验证入口。

## 最终方案

最终提交方案由三个阶段组成：

1. **Residual Scheduling**：以 HEFT upward rank 作为零初始化策略的确定性锚点，神经网络只学习有界排序修正量。
2. **Best-of-64**：随机生成 64 个完整合法调度，保留 makespan 最小的初解。
3. **拓扑序重定位与 best-only LNS**：在保持 DAG 拓扑合法性的前提下执行局部重定位和破坏-修复搜索，只接受严格改善 makespan 的候选。

为应对决赛中可能出现的宽并行 DAG，仓库同时提供自适应候选集：

4. **Adaptive portfolio**：保留 Residual/LNS 主路径，并加入 HEFT 锚点、HEFT+LNS、宽度感知拓扑序和成本优先拓扑序候选；所有候选使用同一 EFT 重放器和 best-only LNS，最终选择 makespan 最小者。该策略对 HEFT 具有硬性非回归保护，宽并行图不再依赖单一训练分布。

```mermaid
flowchart LR
    A[任务 DAG 与异构资源] --> B[Residual Policy]
    B --> C[Best-of-64 候选初解]
    C --> D[合法拓扑序重定位]
    D --> E[best-only LNS]
    E --> F[最终调度方案]
```

## 核心结果

指标定义为 `mean_ratio = RL/LNS makespan ÷ HEFT makespan`，小于 1 表示优于 HEFT。

| 方法 | mean_ratio（5 次均值 ± 样本标准差） | 反超 HEFT | 证据 |
|---|---:|---:|---|
| Residual Best-of-64 | 0.950814 ± 0.003358 | 平均 18.2/20 | `evaluation/results/autonomous_exploration/direction2_lns/direct_vs_residual_paired_summary.json` |
| 计算量对齐 Best-of-128 | 0.942048 ± 0.002063 | 平均 18.6/20 | `evaluation/results/autonomous_exploration/compute_matched_sampling/paired_comparison_summary.json` |
| Residual Best-of-64 + 重定位 + LNS | 0.920890 ± 0.001729 | 20/20 | `evaluation/results/autonomous_exploration/direction2_lns/direct_vs_residual_paired_summary.json` |
| **Adaptive portfolio + 重定位 + LNS** | **0.916441 ± 0.000984** | **20/20** | `evaluation/results/final_pipeline_lns_summary.json` |

决赛优化版最终方案相对配对的 Residual Best-of-64 平均降低 `0.034373`，双侧配对 t 检验 `p=3.121263×10^-5`。旧版 LNS 与耗时相近的纯 Best-of-128 相比平均降低 `0.021158`，`p=2.072081×10^-6`，用于排除“效果仅来自增加采样次数”的混淆因素。

MILP 精确求解在 7 个可证明最优的小规模场景上给出的平均距离为：HEFT/Optimal=`1.106807`，Residual/Optimal=`1.044188`。原始数据见 `evaluation/results/milp_optimal_comparison.json`。

### 30～60 任务规模外验证与运行开销

最终 Adaptive 调度器现在显式区分模型容量与系统容量：30 个任务以内使用
Residual 候选；超过模型 `max_tasks_padding=30` 时不会把越界输入送入神经网络，
而是自动切换到 HEFT 锚点与合法拓扑序 LNS。最终结果始终与 HEFT 锚点比较，
因此在同一调度物理规则下具有硬性非回归保证。

固定 20 个大规模场景（30/40/50/60 各 5 个）的正式结果如下：

| 协议 | mean_ratio ± 样本标准差 | 反超 HEFT | 平均耗时 | P95 耗时 | 峰值进程内存 |
|---|---:|---:|---:|---:|---:|
| 质量档，Best-of-64/3 轮重定位/64 轮 LNS，每场景 10 秒软预算 | 0.941627 ± 0.026569 | 20/20 | 10.002 秒 | 10.003 秒 | 358.1 MiB |
| 固定工作量，Best-of-8/1 轮重定位/8 轮 LNS，无时间截止 | 0.949723 ± 0.023441 | 20/20 | 6.633 秒 | 13.284 秒 | 359.1 MiB |

10 秒质量档相对 HEFT 平均降低 makespan `5.8373%`，配对 t 检验
`p=6.985758×10^-9`。固定工作量协议用于严格复现和观察规模趋势；其平均耗时
从 30 任务的 `3.641` 秒增长到 60 任务的 `13.188` 秒。机器可读证据分别见
`evaluation/results/final_adaptive_large_scale.json` 和
`evaluation/results/final_adaptive_large_scale_fixed.json`。这项结果证明的是完整
Adaptive 系统在 30～60 任务上的安全扩展，不等价于宣称 Residual 神经网络本身
已在 40～60 任务上泛化；这些越界场景会被明确记录为跳过模型。

## 技术报告

- [技术报告（Word）](docs/技术报告.docx)
- [项目说明书（PDF，与当前 Word 同批生成的 63 页版本）](docs/技术报告.pdf)
- [技术报告源文件（Markdown）](docs/技术报告.md)
- [自主探索日志](docs/自主探索日志.md)
- [作品介绍 PPT 制作研究记录](docs/PPT制作研究记录.md)
- [参考文献核验记录](docs/参考文献核验记录.md)

Word 报告包含原生数学公式、学术三线表、自动目录、页码和完整证据路径。报告审计工具会核对引用文件、实验数字、参考文献与隐私信息。

## 快速开始

推荐 Python 3.12。首次使用时安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行完整测试：

```bash
python -m pytest tests/ -v
```

仓库已包含最终模型 `training/checkpoints/ppo_mlp_residual.zip`，可直接评测，无需重新训练。

### 单场景调度 CLI

评委或用户可以直接输入一个 NetworkX node-link 格式的 DAG JSON，得到任务到资源
的完整映射、开始/结束时间、makespan、相对 HEFT 比值、耗时、内存和合法性校验：

```bash
python scripts/schedule.py \
  --input evaluation/scenarios/scenario_10_0.json \
  --output evaluation/results/schedule_output.json \
  --preset balanced \
  --time-budget-seconds 10
```

`fast`、`balanced`、`quality` 三档分别使用 8/32/64 个 Residual 候选和
8/32/64 轮 LNS；`--num-samples`、`--local-max-passes`、`--lns-iterations`
可以单独覆盖。时间预算是软墙钟限制，会在每次 rollout 和邻域评估之间检查；
为保证始终返回合法结果，已经开始的一次评估和 HEFT 锚点允许完成。输入超过模型
容量时，输出 JSON 的 `diagnostics.residual_skipped_reason` 会记录自动降级原因。

批量复现大规模质量与开销测试：

```bash
python evaluation/benchmark_scalability.py \
  --num-samples 64 \
  --local-max-passes 3 \
  --lns-iterations 64 \
  --time-budget-seconds 10
```

### 一键最终流水线

该脚本检查依赖和 checkpoint，必要时训练模型，生成固定验证场景，并使用报告中的 5 个规范种子依次运行 **Adaptive portfolio + Residual Best-of-64 + 合法拓扑序重定位 + best-only LNS**。最后自动执行配对统计，输出决赛优化版对应的 `mean_ratio=0.916441 ± 0.000984`。使用仓库内 checkpoint 时，完整评测通常需要约 10 至 12 分钟。

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_final_pipeline.ps1
```

Linux/openEuler：

```bash
bash scripts/run_final_pipeline.sh
```

输出汇总保存在 `evaluation/results/final_pipeline_lns_summary.json`，五次逐轮结果保存在 `evaluation/results/final_pipeline_lns_repeats/`。

### 决赛前推荐的一键复现

默认 smoke 流程会执行字节码编译、完整 pytest、结构化 DAG 生成；`wide`
流程还会对宽并行场景比较原始 Residual/LNS 与自适应候选集；`final`
流程再执行五次正式评测。

```bash
python scripts/reproduce.py --profile smoke --write-manifest
python scripts/reproduce.py --profile wide --num-samples 16 --lns-iterations 16 --write-manifest
python scripts/reproduce.py --profile final --write-manifest
python scripts/reproduce.py --profile smoke --with-os-matrix --write-manifest
```

一键宽并行 smoke 结果写入 `evaluation/results/wide_parallel_adaptive_reproduce.json`；正式 64/64 宽并行专项结果写入 `evaluation/results/wide_parallel_adaptive.json`，其中会记录
图形状、候选来源、每个候选 makespan 及配对改进量。

`--with-os-matrix` 会先等待 Docker Desktop 引擎就绪（默认 120 秒），再依次构建
openEuler 24.03 LTS-SP4、openKylin 2.0 SP1 和 Anolis OS 23 镜像并执行 smoke 检查；等待超时或本机
没有 Docker 时会将原因写入结果 JSON，而不会伪造通过结果。需要严格把环境不可用
视为失败时，可直接运行：

```bash
python scripts/run_os_matrix.py --pull --strict --wait-seconds 120
```

### 单次调试 LNS 精修方案

```bash
python evaluation/evaluate_residual_lns.py \
  --config training/configs/ppo_mlp_residual.yaml \
  --model-path training/checkpoints/ppo_mlp_residual \
  --results-path evaluation/results/final_lns_run.json \
  --num-samples 64 \
  --local-max-passes 3 \
  --lns-iterations 64
```

未指定 `--sampling-seed` 时使用系统熵种子，因此单次结果会在统计范围内波动。一键脚本固定使用报告中最初由系统熵产生的 5 个规范种子，使任何克隆环境都能复现正式统计，而不是挑选单次最好结果。

### 重新训练 Residual 模型

```bash
python training/train_ppo.py --config training/configs/ppo_mlp_residual.yaml
```

训练配置为 200000 timesteps，归一化观测、`relative_heft` 奖励和 residual 调度模式均由 YAML 显式控制。

## 项目结构

```text
baselines/      HEFT、MILP 精确解与 Hybrid 调度器
configs/        异构资源配置
env/            DAG 环境、插入式调度、归一化与 ranked/residual 环境
policies/       RL 调度器、GAT 特征提取器、残差策略与 LNS 精修器
training/       数据集生成、BC/PPO 训练、配置及最终 checkpoint
evaluation/     验证场景、评测脚本、统计检验和原始结果
diagnostics/    数据集、动作分布、报告真实性与结构审计工具
tests/          环境、调度一致性、MILP、Residual 和 LNS 回归测试
artifacts/      经归类保留的诊断日志和中间实验依据
docs/           技术报告、参考资料、截图及审计结果
scripts/        一键复现、Word 报告生成和终端截图工具
```

## 关键设计

- **调度物理一致性**：环境与 HEFT 共用 `find_earliest_slot` 插入式资源时间线规则。
- **动作空间拆分**：策略只排序当前就绪任务，资源分配由 EFT 确定性完成。
- **残差锚定**：策略初始行为等价于 HEFT，训练只学习 upward rank 的修正量。
- **合法动作约束**：MaskablePPO 与 DAG 就绪掩码保证每一步动作合法。
- **统计严谨性**：所有正式结论来自重复评测、配对设计和显著性检验。
- **精确基准**：PuLP/CBC MILP 在可求解规模上提供全局最优参照。

## 证据与审计

```bash
python diagnostics/technical_report_result_audit.py --report docs/技术报告.md
python diagnostics/technical_report_structure_audit.py --report docs/技术报告.docx
```

当前审计结果：66 个报告引用路径全部存在，全部正式数字核对通过；Word 报告包含 6 个原生公式对象和 15 张合规三线数据表。

完整实验结果索引见 [evaluation/results/README.md](evaluation/results/README.md)，过程性产物说明见 [artifacts/README.md](artifacts/README.md)。

## 跨平台验证

项目提供 Windows 本地验证、历史 openEuler 24.03 LTS-SP4 容器证据，以及
openEuler 24.03 LTS-SP4 / openKylin 2.0 SP1 / Anolis OS 23 Docker 矩阵入口：

```bash
python scripts/run_os_matrix.py
```

2026-08-31 已执行 `--pull --strict` 严格矩阵：三套镜像均构建成功、容器运行
退出码均为 0，分别使用 Python 3.11.6、3.12.2、3.10.12，统一安装
`torch=2.12.1+cpu` 且 `cuda_available=False`。三套环境均完成全项目字节码
编译、完整测试（每套均为 `38 passed, 17 warnings`）和 20 个结构泛化场景
生成。机器可读原始输出见 `evaluation/results/os_matrix.json`；同日本地一键
smoke 同样为 `38 passed, 17 warnings`。

openEuler 环境信息、依赖快照、pytest 输出及推理日志保存在：

- `evaluation/results/openeuler_validation/`
- `evaluation/results/openEuler_pytest_structural_final.log`

Docker 构建定义和边界说明见 `docker/README.md`。OpenHarmony 不是通用 Linux
服务器运行环境；在没有可公开验证的标准 Python 运行容器前，仓库不宣称已在
OpenHarmony 原生设备系统上运行。

## 已知边界

- Residual 模型训练规模为 8 至 25 个任务、输入容量为 30；40～60 任务由系统显式
  跳过越界模型并使用 HEFT/LNS 安全路径。大规模整体调度结果已经验证，但不能把它
  表述为神经网络本身完成了任意规模泛化。
- 原始 Residual 模型在历史宽并行 DAG 泛化集上弱于原始分布；该负面结果保留在
  `evaluation/results/structural_generalization/`。决赛优化版自适应候选集通过
  HEFT 锚点、HEFT+LNS、宽度感知和成本优先候选提供运行时非回归保护；正式宽并行
  64/64 评测达到 `mean_ratio=0.929848`、5/5 不劣于 HEFT，其中 1 个打平场景由
  MILP 证明已达到全局最优，证据见 `evaluation/results/wide_parallel_adaptive.json`。
- LNS 通过增加推理阶段搜索换取更优 makespan，不属于零额外成本改进。
- 当前结论针对静态已知 DAG，不自动覆盖动态任务到达、抢占或实时硬截止期场景。
