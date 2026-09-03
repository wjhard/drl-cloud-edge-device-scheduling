# 评测证据

本目录保存技术报告引用的评测汇总、逐场景结果、统计检验和跨平台验证日志。与缓存或临时输出不同，这些文件用于复核报告中的具体数字，因此纳入版本控制。

- 根目录的 `summary*.json`：各训练配置或推理配置的评测汇总。
- `paired15/`、`repeated*/`：配对检验和重复评测的逐轮结果与统计汇总。
- `milp_solver_logs/`：CBC 求解过程日志；其中本机 Python 安装路径已脱敏。
- `openeuler_validation/`：openEuler 容器环境、依赖、测试与评测证据。
- `structural_generalization/`：结构化泛化场景配置和分组结果，保留原始 Residual 在宽并行 DAG 上的历史短板。
- `wide_parallel_adaptive.json`：决赛优化版宽并行 64/64 专项评测，记录自适应候选集、候选 makespan 和 MILP 打平最优性检查。
- `wide_parallel_adaptive_reproduce.json`：`scripts/reproduce.py --profile wide` 的较快 smoke 口径输出。
- `structural_generalization_adaptive_smoke/`：自适应候选集在 wide/deep/homogeneous/control 四组上的快速副作用检查。
- `reproducibility_manifest.json`：最近一次 `scripts/reproduce.py --write-manifest` 的环境、命令和耗时清单。
- `os_matrix.json`：openEuler/openKylin Docker 矩阵运行结果；若 Docker daemon 不可用，会记录 Docker context 和 WSL 诊断。
- `run_final_pipeline_*.log`：一键运行流程的端到端验证记录。
- `final_pipeline_lns_summary.json` 与 `final_pipeline_lns_repeats/`：最终一键脚本使用五个规范种子得到的 LNS 逐轮结果与统计汇总。

日志中的本机用户目录统一替换为 `<USER_HOME>`。脱敏不改变任何实验数字、随机种子、耗时或算法输出。
