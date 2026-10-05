# 最小运行演示

先按根目录 README 安装 Python 3.10–3.12、CPU PyTorch 与 requirements.txt。以下命令从项目根目录运行，使用仓库提供的一个 10 任务 DAG 和已训练 checkpoint，分别生成 HEFT 与 Adaptive 调度、检查合法性，并输出 DAG 和甘特图。

```bash
python demo/run_demo.py
```

Windows 也可运行 `demo.cmd` 或 `powershell -ExecutionPolicy Bypass -File demo/run_demo.ps1`；Linux 可运行 `bash demo/run_demo.sh`。包装脚本支持通过 `PYTHON` 环境变量指定解释器。

结果位于 `demo/output/heft.json`、`demo/output/adaptive.json` 和 `demo/output/schedule.png`。可用 `--output-dir <目录>` 隔离结果、`--skip-plot` 跳过作图、`--preset fast` 缩小搜索量。默认使用 quality（64 次采样、3 轮局部搜索、64 次 LNS）与 8 秒软时间预算。预算约束调度搜索，模型加载与 Python 启动另行耗时。

本入口不启动训练，也不覆盖 evaluation/results 中的正式批量实验。单场景演示不能作为 20 场景、5 次推理重复的统计结论；不同依赖版本或时间预算触发时，选中候选及耗时可能变化。原有演示视频与过程材料见根目录 README。
