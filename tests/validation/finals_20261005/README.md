# 2026-10-05 决赛目录重构验证

`validation.json` 与同目录日志记录本次 Windows 11 原生验证：原有 44 项 pytest、src/scripts/tests/demo 编译、训练和评测 CLI 参数解析、隔离的 smoke 场景生成、提供的 checkpoint 单场景 HEFT/Adaptive 推理。`validate.py` 可以从仓库根目录重新执行；本次未启动完整训练、五重复批量评测或国产系统容器矩阵。

本次 Python 3.12.4、PyTorch 2.13.0+cpu，以及实际 NumPy、NetworkX、Matplotlib、SciPy 版本详见 `validation.json`。它们部分高于 requirements.txt 的固定目标版本，因此本次记录只证明这一实际环境下的目录迁移与推理回归。历史 2026-08/09 的跨操作系统和性能数据保留在 evaluation/results；不可将这次 Windows 验证写成三个容器的最新测试结果。

`demo/` 是一个 10 任务 DAG 的真实输出，合法性校验通过；该示例不能替代正式 20 场景、同一模型 5 次推理采样的性能结论。程序比较全部 158 个既有 evaluation/results JSON 的 SHA-256，确认验证前后未变；模型哈希见 validation.json。

`editable_install.log` 与 `editable_install.json` 记录 setuptools 构建和隔离临时环境的 editable 安装。安装使用 `--no-deps --no-build-isolation`，依赖已由运行环境提供；此项只验证项目打包布局与模块导入位置，不声称重新解析或安装了 requirements.txt 的全部固定版本。配置、模型、场景保持在仓库根目录，运行与 editable 开发仍需保留完整仓库。

Dockerfile 的源码路径、PYTHONPATH 与镜像构建上下文 checkpoint 包含规则完成静态回归；本机 Docker Desktop Linux daemon 不可用，本轮未构建或启动三个系统镜像。

仓库交付的主验证集补齐为 20 个持久化 DAG，直接复制原项目工作树 evaluation/scenarios 的既有输入，没有重新生成 DAG 或重跑评测。10、15、20、25 任务各 5 个，文件名与五轮正式结果中的场景名及任务数、边数一致；原已跟踪的 scenario_10_0.json 保留，和复制来源语义一致。来源范围、逐文件 SHA-256 和语义哈希见 `shipped_scenarios.json`。配置中的生成种子为 1000000～1000019，逐文件映射按生成器的任务规模顺序和文件编号推导；场景 JSON 本身未嵌入种子，不将该映射声称为文件内独立记录。全部输入现可随 Git 提交，不再被整目录忽略。
