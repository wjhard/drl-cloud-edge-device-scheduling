# 基于深度强化学习的云-边-端异构计算资源管理调度方法

本项目面向中国研究生操作系统开源创新大赛第 16 题，研究带依赖约束的任务 DAG 在云、边、端异构计算资源上的调度问题。目标是在满足任务前驱关系、通信时延和资源互斥约束的前提下，最小化整体完成时间（makespan）。

项目提供完整的环境建模、HEFT/MILP 基线、MaskablePPO 训练、残差式任务排序、Best-of-N 推理、局部搜索精修、统计检验及 openEuler 跨平台验证流程。

## 最终方案

最终提交方案由三个阶段组成：

1. **Residual Scheduling**：以 HEFT upward rank 作为零初始化策略的确定性锚点，神经网络只学习有界排序修正量。
2. **Best-of-64**：随机生成 64 个完整合法调度，保留 makespan 最小的初解。
3. **拓扑序重定位与 best-only LNS**：在保持 DAG 拓扑合法性的前提下执行局部重定位和破坏-修复搜索，只接受严格改善 makespan 的候选。

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
| **Residual Best-of-64 + 重定位 + LNS** | **0.920890 ± 0.001729** | **20/20** | `evaluation/results/autonomous_exploration/direction2_lns/direct_vs_residual_paired_summary.json` |

最终方案相对配对的 Residual Best-of-64 平均降低 `0.029925`，双侧配对 t 检验 `p=2.656851×10^-5`。与耗时相近的纯 Best-of-128 相比，最终方案平均降低 `0.021158`，`p=2.072081×10^-6`，用于排除“效果仅来自增加采样次数”的混淆因素。

MILP 精确求解在 7 个可证明最优的小规模场景上给出的平均距离为：HEFT/Optimal=`1.106807`，Residual/Optimal=`1.044188`。原始数据见 `evaluation/results/milp_optimal_comparison.json`。

## 技术报告

- [技术报告（Word）](docs/技术报告.docx)
- [技术报告（PDF）](docs/技术报告.pdf)
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

### 一键基础流水线

该脚本检查依赖和 checkpoint，必要时训练模型，并完成固定验证集上的 Residual Best-of-64 评测。

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_final_pipeline.ps1
```

Linux/openEuler：

```bash
bash scripts/run_final_pipeline.sh
```

### 运行最终 LNS 精修方案

```bash
python evaluation/evaluate_residual_lns.py \
  --config training/configs/ppo_mlp_residual.yaml \
  --model-path training/checkpoints/ppo_mlp_residual \
  --results-path evaluation/results/final_lns_run.json \
  --num-samples 64 \
  --local-max-passes 3 \
  --lns-iterations 64
```

未指定 `--sampling-seed` 时使用系统熵种子，因此单次结果会在统计范围内波动。正式报告数字来自 5 次独立、配对的重复评测，而非挑选单次最好结果。

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

当前审计结果：62 个报告引用路径全部存在，全部正式数字核对通过；Word 报告包含 6 个原生公式对象和 14 张合规三线数据表。

完整实验结果索引见 [evaluation/results/README.md](evaluation/results/README.md)，过程性产物说明见 [artifacts/README.md](artifacts/README.md)。

## 跨平台验证

项目已在 Windows 和 openEuler 24.03 LTS-SP4 Docker 容器中验证。openEuler 环境信息、依赖快照、pytest 输出及推理日志保存在：

- `evaluation/results/openeuler_validation/`
- `evaluation/results/openEuler_pytest_structural_final.log`

## 已知边界

- 模型训练规模为 8 至 25 个任务，不能直接外推为任意大规模 DAG 的性能保证。
- 宽并行 DAG 上的泛化弱于原始分布；该负面结果保留在 `evaluation/results/structural_generalization/`。
- LNS 通过增加推理阶段搜索换取更优 makespan，不属于零额外成本改进。
- 当前结论针对静态已知 DAG，不自动覆盖动态任务到达、抢占或实时硬截止期场景。
