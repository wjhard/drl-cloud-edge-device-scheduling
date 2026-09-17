# 评测证据

本目录保存技术报告引用的评测汇总、逐场景结果、统计检验和跨平台验证日志。与缓存或临时输出不同，这些文件用于复核报告中的具体数字，因此纳入版本控制。

- 根目录的 `summary*.json`：各训练配置或推理配置的评测汇总。
- `paired15/`、`repeated*/`：配对检验和重复评测的逐轮结果与统计汇总。
- `milp_solver_logs/`：CBC 求解过程日志；其中本机 Python 安装路径已脱敏。
- `openeuler_validation/`：openEuler 容器环境、依赖、测试与评测证据。
- `structural_generalization/`：结构化泛化场景配置和分组结果，保留原始 Residual 在宽并行 DAG 上的历史短板。
- `wide_parallel_adaptive.json`：决赛优化版宽并行 64/64 专项评测，记录自适应候选集、候选 makespan 和 MILP 打平最优性检查。
- `wide_parallel_adaptive_reproduce.json`：`scripts/reproduce.py --profile wide` 的较快 smoke 口径输出。
- `final_adaptive_large_scale.json`：30/40/50/60 任务、每场景 10 秒软预算的最终 Adaptive 质量与资源开销结果。
- `final_adaptive_large_scale_fixed.json`：Best-of-8、1 轮重定位、8 轮 LNS 的固定工作量规模测试，用于确定性复现和耗时趋势比较。
- `structural_generalization_adaptive_smoke/`：自适应候选集在 wide/deep/homogeneous/control 四组上的快速副作用检查。
- `reproducibility_manifest.json`：最近一次 `scripts/reproduce.py --write-manifest` 的环境、命令和耗时清单。
- `os_matrix.json`：openEuler/openKylin/Anolis OS 三套 Docker 矩阵运行结果；若 Docker daemon 不可用，会记录 Docker context 和 WSL 诊断。
- `video_live_schedule.json` 与 `video_live_schedule.png`：高清演示录制时在 openEuler 容器中真实运行得到的单场景调度输出，以及据此生成的 DAG 和资源甘特图。单场景 `ratio_to_heft=0.8583503930404773` 不替代五次重复评测的总体指标。
- `evaluation_summary_final.png` 与 `evaluation_summary_generalization.png`：由 `scripts/visualize_evaluation_summary.py` 读取正式 JSON 生成的结果展示图，原始数据仍以相应 JSON 为准。
- `run_final_pipeline_*.log`：一键运行流程的端到端验证记录。
- `final_pipeline_lns_summary.json` 与 `final_pipeline_lns_repeats/`：最终一键脚本使用五个规范种子得到的 LNS 逐轮结果与统计汇总。

大规模结果中的内存字段是调度进程的 RSS/峰值 RSS，CPU 时间是进程累计 CPU
时间；墙钟时间单独记录。时间预算协议可能因机器负载在最后若干邻域上产生细微
差异，固定工作量协议在相同随机种子和软件栈下输出相同调度。

日志中的本机用户目录统一替换为 `<USER_HOME>`。脱敏不改变任何实验数字、随机种子、耗时或算法输出。

## 演示结果可视化

录制使用的输入快照 `evaluation/scenarios/scenario_10_0.json` 已随仓库保存。
在项目根目录安装依赖后，可由原始 JSON 重新生成视频中展示的图：

```bash
python scripts/visualize_schedule.py --scenario evaluation/scenarios/scenario_10_0.json --result evaluation/results/video_live_schedule.json --output evaluation/results/video_live_schedule.png
python scripts/visualize_evaluation_summary.py --final evaluation/results/final_pipeline_lns_summary.json --wide evaluation/results/wide_parallel_adaptive.json --large evaluation/results/final_adaptive_large_scale.json --os-matrix evaluation/results/os_matrix.json --output evaluation/results/evaluation_summary.png --view all
```

上述命令仅可视化已有证据；重新进行调度、重复评测或容器验证，请使用根目录
README 中的 `scripts/schedule.py`、`scripts/reproduce.py` 和 `scripts/run_os_matrix.py`。
