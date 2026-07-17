# 实验产物归档

本目录保存具有复核价值、但不属于源代码、运行配置或正式文档的实验产物。

## `bestof_sweep_logs/`

- `bestof_sweep_serial_times.tsv`：Residual/Ranked 策略 Best-of-N 串行推理扫描中，N=1、4、8、16、32、64、128 的总评测耗时记录。数值与同批 `bestof_sweep_n*_serial_output.log` 运行日志对应。

## `ablation_scan_results/`

- `scan_combined_times.tsv`：BC 热启动、HEFT rank 特征和课程学习叠加实验的训练及评测耗时。
- `scan_eval_times.tsv`：ranked 系列消融实验的确定性与 Best-of-8 评测耗时。
- `scan_final_summary_table.tsv`：七阶段消融扫描的最终汇总表。
- `scan_step4_quick_times.tsv`：ranked 环境学习率与网络规模候选配置的 50,000 步快速筛选耗时。
- `scan_step4_full_times.tsv`：快速筛选后入选配置的 200,000 步完整训练与评测耗时。
- `scan_step6_times.tsv`：400,000 步长训练实验的训练与评测耗时。
- `scan_step7_seed_times.tsv`：多随机种子训练、确定性评测与 Best-of-8 评测耗时。

## `milp_manual_solutions/`

这些文件的内容头和变量格式表明它们是 CBC 求解器在 MILP 诊断期间写出的解文件；`baselines/milp_optimal_scheduler.py` 在 Windows 上启用 `keepFiles=True`，会保留 CBC/PuLP 中间文件。

- `manual_lp.sol`：文件头为 `Optimal`，且包含分数指派变量，与 LP 松弛诊断结果一致。
- `manual_root.sol`：文件名将其标记为 root 诊断；文件头可确认它因迭代限制停止、未得到整数解并采用连续解。
- `manual_nothreads.sol`：文件名将其标记为 no-threads 诊断；文件头可确认它因超时停止并保存了当时的整数可行解。

现有仓库没有保存生成这三个 `manual_*.sol` 文件的完整命令或场景编号，因此不能进一步确认其具体命令行来源，也不能仅凭文件名独立验证求解参数。它们作为 MILP/CBC 排查证据保留；自动生成的 `*-pulp.mps`、`*-pulp.mst` 和 `*-pulp.sol` 临时文件由 `.gitignore` 排除。
