# 基于深度强化学习的云—边—端异构计算资源管理调度方法

队伍：操作系统创新小分队。赛题：第 16 题。当前版本按组委会决赛目录规范整理，正式材料更新于 2026-10-05。

**团队手抄声明已补充：** [人工智能及第三方工具使用说明](docs/人工智能及第三方工具使用说明.pdf) 已包含团队提供的亲笔手抄照片及工具、来源和许可证附录，并提供对应 Word 文件。其他 AI 工具按团队确认填写为“无”；本次 Codex 的代码、文档和演示材料整理如实列入声明。

## 决赛材料入口

| 材料 | 文件 | 说明 |
|---|---|---|
| 项目说明书 | [PDF](docs/项目说明书.pdf) · [可编辑 Word](docs/项目说明书.docx) | 32 页，沿用组委会模板，保留目录、公式及表格 |
| 项目创新说明 | [PDF](docs/项目创新说明.pdf) · [Word](docs/项目创新说明.docx) | 三项创新，区分已有方法、项目贡献与验证证据 |
| 成员分工及主要贡献说明 | [PDF](docs/成员分工及主要贡献说明.pdf) · [Word](docs/成员分工及主要贡献说明.docx) | 成员 A（队长）、B、C，保留团队提供的职责 |
| 人工智能及第三方工具使用说明 | [PDF](docs/人工智能及第三方工具使用说明.pdf) · [Word](docs/人工智能及第三方工具使用说明.docx) | 官方表格、团队手抄声明照片与工具附录 |
| 作品原创承诺书 | [PDF](作品原创承诺书.pdf) | 团队提供的手抄扫描件，保留补充日期 |
| 决赛现场演示 PPT | [PPTX](presentation/决赛现场演示PPT.pptx) | 22 页，含嵌入式视频 |
| 决赛演示视频 | [MP4](presentation/决赛演示视频.mp4) | 约 1 分 58 秒，小于 100 MB |
| 测试及验证材料 | [测试](tests/) · [本次验证](tests/validation/finals_20261005/README.md) · [历史实验索引](evaluation/results/README.md) | 测试代码、原始日志及机器可读结果 |
| 可验证交付物 | [演示说明](demo/README.md) · [运行入口](demo/run_demo.py) | 使用提供的模型生成完整调度及 DAG／甘特图 |

依据 [组委会决赛提交说明](https://cpipc.acge.org.cn/cw/contestNews/detail/2c9080178c7c917b018d1b1a0af61cd6/2c908017a0a728fb01a0ad185d28082f?page=0)（[官网通知列表](https://cpipc.acge.org.cn/cw/contestNews/list/2c9080178c7c917b018d1b1a0af61cd6/1)中也可查阅），最终材料须进入初赛 fork 的 GitLink 仓库 **master** 分支。GitHub 与原比赛 GitLink 仓库的 master 分支已同步更新；报名系统还须更新项目名称、简介及至少一份图片／视频／PDF 材料。官方截止时间为 **2026-10-07 24:00**，以组委会公告为准。

## 项目简介、场景和功能

本项目研究带前驱依赖的任务 DAG 在云、边、端异构计算资源上的调度，适用于任务图和资源参数已知的批处理工作流、边缘计算卸载及异构任务编排。在满足前驱关系、通信时延、资源互斥约束的前提下，降低整体完成时间（makespan），输出任务资源映射、开始与结束时间、资源时间线及合法性校验。

主要功能包括 DAG／资源建模、HEFT 与 MILP 基线、MaskablePPO 残差排序训练、Best-of-N 推理、合法拓扑序重定位和 best-only LNS、结构自适应候选集、重复评测与配对统计、单场景 CLI、图形展示和国产 Linux 容器验证入口。

最终调度器保留 HEFT 锚点，使用统一 EFT 插入式重放规则比较 Residual、HEFT+LNS、宽度感知和成本优先等候选。网络修正任务优先级，EFT 完成资源分配；重定位与 LNS 只接受合法且严格改善的候选。因此返回结果在同一物理模型下不劣于 HEFT。网络输入容量为 30，超过该容量时显式跳过 Residual，使用 HEFT／拓扑序搜索路径。

## 目录结构

```text
项目根目录/
├── README.md
├── src/                         项目源代码
│   ├── env/                     DAG、资源、环境与统一 EFT 规则
│   ├── baselines/               HEFT、MILP、Hybrid
│   ├── policies/                Residual、候选集、重定位与 LNS
│   ├── training/                BC/PPO 训练代码
│   ├── evaluation/              评测、统计和规模验证代码
│   ├── diagnostics/             诊断和历史报告审计
│   └── scheduler_interface.py
├── docs/                        四项必需 PDF、Word 和过程记录
├── tests/                       测试及 validation/ 验证材料
├── demo/                        可运行演示及输出示例
├── presentation/                决赛 PPT 和视频
├── 作品原创承诺书.pdf
├── scripts/                     CLI、复现和系统矩阵入口
├── configs/                     异构资源配置
├── training/configs/            训练配置
├── training/checkpoints/        最终模型
├── evaluation/scenarios*/       固定场景
├── evaluation/results/          历史实验和跨系统证据
├── docker/                      国产 Linux 容器定义
├── requirements.txt
└── pyproject.toml               editable 安装及 pytest 路径
```

Python 导入名保持 `env`、`policies` 等原名，源码统一置于 `src/`；配置、模型及结果保留在根目录。旧版报告生成脚本和研究记录属于过程资料，当前正式说明书以本页链接的 32 页 Word／PDF 为准。

仓库已补齐本地保存的 20 个主验证 DAG（10／15／20／25 任务各 5 个），与五轮正式结果中的场景名、任务数和边数逐项一致；来源及哈希见 [shipped_scenarios.json](tests/validation/finals_20261005/shipped_scenarios.json)。本次补入这些持久化输入，没有重新生成它们或改动历史性能结果。

## 环境、安装和构建

推荐 Python 3.12，目标依赖支持 Python 3.10–3.12。固定依赖见 [requirements.txt](requirements.txt)，容器使用 `torch==2.12.1+cpu`。从仓库根目录运行：

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.12.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Python 源码无需 C/C++ 编译。推荐完整仓库配合 editable 安装，保留根目录数据与模型；本次没有将普通 wheel 作为独立部署交付验证。提供的 `training/checkpoints/ppo_mlp_residual.zip` 可直接推理，无需重新训练。

## 运行

生成 HEFT／Adaptive 调度、合法性结果、任务 DAG 和资源甘特图：

```bash
python demo/run_demo.py
```

Windows 可运行 `demo.cmd`，Linux 可运行 `bash demo/run_demo.sh`。默认输出到 `demo/output/`，支持 `--output-dir`、`--skip-plot` 和 `--preset fast`，详见 [demo/README.md](demo/README.md)。

单场景 CLI 接收 NetworkX node-link 格式 DAG JSON：

```bash
python scripts/schedule.py --input evaluation/scenarios/scenario_10_0.json --method adaptive --preset quality --time-budget-seconds 8 --output demo/output/adaptive.json
```

`fast`、`balanced`、`quality` 默认采样和 LNS 次数为 8、32、64，可用参数覆盖。时间预算是软墙钟限制，模型加载、Python 启动、HEFT 锚点及已经开始的一次评估可能增加实际耗时。截止提前触发时，候选数可能少于上限；输出会记录候选来源和模型跳过原因。

训练和批量评测入口：

```bash
python src/training/train_ppo.py --config training/configs/ppo_mlp_residual.yaml
python src/evaluation/evaluate_residual_lns.py --config training/configs/ppo_mlp_residual.yaml --model-path training/checkpoints/ppo_mlp_residual --results-path demo/output/lns_run.json --num-samples 64 --local-max-passes 3 --lns-iterations 64
python src/evaluation/benchmark_scalability.py --num-samples 64 --local-max-passes 3 --lns-iterations 64 --time-budget-seconds 10 --results-path demo/output/large_scale_run.json
```

同一模型的五次规范推理种子完整流程可运行 `powershell -ExecutionPolicy Bypass -File scripts/run_final_pipeline.ps1` 或 `bash scripts/run_final_pipeline.sh`。该入口默认写入 `evaluation/results/`，如需保留现有记录，请先复制仓库或备份输出。硬件、依赖和预算会影响运行时间与选中候选，不能承诺每次恰好得到历史汇总值。

## 测试及验证

```bash
python -m pytest tests -q
python scripts/reproduce.py --profile smoke --write-manifest --output-dir demo/output/smoke
```

2026-10-05 源码迁移后的 Windows 原生验证为 **44 passed、17 warnings**，编译、训练／评测 CLI、隔离 smoke、editable 安装和 checkpoint 推理通过。单个 10 任务演示通过合法性校验，makespan 从 HEFT 的 `0.455228` 降至 `0.390746`，比值 `0.858350`。这是一场景演示，不能替代批量统计。

本次实际环境为 Python 3.12.4、PyTorch 2.13.0+cpu，部分依赖高于固定目标版本，详情见 [validation.json](tests/validation/finals_20261005/validation.json)。本次验证确认 **158 个既有实验 JSON 的 SHA-256 未变化**，未重跑完整训练、五次批量性能评测或国产系统镜像构建。Dockerfile 与 checkpoint 包含规则完成静态检查，本机 Docker Linux daemon 未运行。

历史 2026-08-31 严格矩阵在 openEuler 24.03 LTS-SP4、openKylin 2.0 SP1、Anolis OS 23 均完成构建、编译和测试，每套 38 passed、17 warnings，见 [os_matrix.json](evaluation/results/os_matrix.json)。2026-09-22 openEuler 另有 44 项测试及实际调度日志，见 [openEuler 测试日志](evaluation/results/openEuler_pytest_structural_final.log)。这些证据验证相应容器用户态环境，不能视为原生设备内核或 OpenHarmony 支持。矩阵入口和运行说明见 [docker/README.md](docker/README.md)。

## 当前实验结果

指标为同一场景下 `makespan / HEFT makespan`，小于 1 表示优于 HEFT。五次重复使用**同一个已训练 checkpoint 的不同推理采样种子**，不属于五次独立重新训练。

| 方法／协议 | 比值均值 ± 样本标准差 | 证据 |
|---|---:|---|
| Residual Best-of-64，固定 20 场景、五次推理重复 | 0.950814 ± 0.003358 | [配对结果](evaluation/results/autonomous_exploration/direction2_lns/direct_vs_residual_paired_summary.json) |
| Residual Best-of-64 + 重定位 + LNS，同上 | 0.920890 ± 0.001729 | [配对结果](evaluation/results/autonomous_exploration/direction2_lns/direct_vs_residual_paired_summary.json) |
| **Adaptive portfolio + 重定位 + LNS，同上** | **0.916441 ± 0.0009845** | [最终汇总](evaluation/results/final_pipeline_lns_summary.json) |
| 30／40／50／60 任务各 5 个，质量档、10 秒软预算 | 0.941627 ± 0.026569 | [规模质量结果](evaluation/results/final_adaptive_large_scale.json) |
| 同一大规模场景集，固定 8 次采样／1 轮重定位／8 轮 LNS | 0.949723 ± 0.023441 | [固定工作量结果](evaluation/results/final_adaptive_large_scale_fixed.json) |

最终 Adaptive 每次重复均 20/20 优于 HEFT，平均降低 makespan **8.3559%**；相对配对 Residual Best-of-64 的比值平均降低 `0.034373`（相对降幅约 3.6151%），双侧配对 t 检验 `p=3.121263×10^-5`。

历史计算量对齐实验比较旧版 LNS 与 Residual Best-of-128：比值为 `0.920890` 和 `0.942048`，平均耗时约 `54.402` 和 `56.802` 秒，`p=2.072081×10^-6`，见 [证据](evaluation/results/autonomous_exploration/compute_matched_sampling/paired_comparison_summary.json)。这个等耗时结论针对旧版 LNS，不能扩展为最终 Adaptive 的等耗时比较。

大规模质量档 20/20 优于 HEFT，平均耗时 `10.002` 秒、P95 `10.003` 秒、峰值进程内存 `358.1 MiB`；固定工作量档平均 `6.633` 秒、P95 `13.284` 秒、峰值 `359.1 MiB`。超过 30 个任务的结果验证系统 HEFT／LNS 路径，不是神经网络越界泛化。

宽并行 64/64 搜索比值为 `0.929848`，5 场景中 4 个改善、1 个打平；打平场景由 MILP 证明已达最优，见 [专项证据](evaluation/results/wide_parallel_adaptive.json)。历史 MILP 的 10 场景中有 7 个获得最优证明，对这 7 个场景 HEFT/Optimal=`1.106807`、Residual Best-of-16/Optimal=`1.044188`，见 [精确基准](evaluation/results/milp_optimal_comparison.json)；这不能直接证明最终 Adaptive 的最优性。

## 创新、开源来源和边界

本项目使用 HEFT、PPO／MaskablePPO、MILP 和局部搜索等已有方法，创新主张集中于统一物理规则上的残差排序、合法拓扑序精修、结构自适应候选集和容量安全处理。贡献、现有能力及代码／证据映射见 [项目创新说明](docs/项目创新说明.pdf)。依赖、许可来源、工具范围和待核查项见 [工具使用说明](docs/人工智能及第三方工具使用说明.pdf)。本次没有完成全部源码逐行溯源或所有演示素材的授权核验。

当前结论针对静态已知 DAG，不覆盖动态到达、抢占、实时硬截止期及任意规模神经网络泛化。搜索需要额外推理计算。历史不利的结构泛化结果保留在 [structural_generalization](evaluation/results/structural_generalization/)，过程记录见 [自主探索日志](docs/自主探索日志.md)、[决赛优化记录](docs/决赛优化记录.md)和 [artifacts](artifacts/README.md)。
