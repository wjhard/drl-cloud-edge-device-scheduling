# 国产开源操作系统验证

本目录提供三套可复现的容器构建文件：

- `Dockerfile.openeuler`：`hub.oepkgs.net/openeuler/openeuler:24.03-lts-sp4`
- `Dockerfile.openkylin`：`quay.io/openkylin/openkylin:2.0`
- `Dockerfile.anolis`：`registry.openanolis.cn/openanolis/anolisos:23`（容器 `/etc/os-release` 标识为 Anolis OS 23）

这些地址均由对应操作系统社区维护。采用社区官方仓库可避免部分网络环境下
Docker Hub 鉴权域名解析异常，同时不改变镜像发行版、版本标签或验证内容。
构建过程只安装项目必需的软件包，不执行与验证无关的全系统升级；openEuler
还会暂时禁用曾出现镜像同步校验差异的 `update`/`update-source` 仓库，
所需软件包仍从该发行版的官方 `OS`、`everything` 和 `EPOL` 仓库安装。
Python 依赖优先使用 PyPI 提供的 x86_64 预编译 wheel，因此镜像不额外安装
Git、GCC 和完整构建工具链，减少现场下载量和构建时间。
openKylin 构建只使用发行版主仓库，并移除基础镜像中与本项目依赖无关、
响应较慢的 `anything2.0` 附加 PPA。
Anolis 基础镜像已经包含 Python 3.10 和 `ensurepip`，因此直接启用内置 pip，
不访问当前可能返回 502 的 DNF 软件仓库，也不触发无关系统包升级。
PyTorch 固定安装 `torch==2.12.1+cpu`，避免 CPU 调度实验误下载 CUDA、
cuDNN、NCCL 等 GPU 运行库。默认使用上海交通大学同步的 PyTorch CPU wheel
索引；该索引覆盖 Python 3.10、3.11、3.12 的 x86_64 wheel，并保留 wheel 的
SHA-256 元数据。可通过 `--build-arg PYTORCH_INDEX_URL=<URL>` 切换镜像源。
该步骤只获取 torch 本体，其他通用依赖另行安装，最后使用 `pip check` 验证
依赖关系完整。
依赖安装层只复制 `requirements.txt`，完整源码在依赖层之后复制，因此修改
算法代码、测试或说明文档不会让 Docker 重复下载整套 Python 环境。
通用 Python 包默认使用南京大学 PyPI 镜像，并用 BuildKit cache mount 保存
下载缓存；可通过 `--build-arg PYPI_INDEX_URL=<URL>` 切换回 PyPI 或其他镜像。

三套镜像都会安装 Python 依赖；矩阵运行时会记录 `/etc/os-release`、
Python 版本，执行全项目字节码编译，并运行
`scripts/reproduce.py --profile smoke`（包含完整 pytest）。系统身份、编译、
测试与运行输出一并写入 `evaluation/results/os_matrix.json`，复现摘要写入
`evaluation/results/reproducibility_manifest.json`。

在项目根目录执行：

```bash
python scripts/run_os_matrix.py
python scripts/run_os_matrix.py --pull --strict
```

也可以手动执行单个镜像：

```bash
docker build -f docker/Dockerfile.openeuler -t drl-scheduler:openeuler .
docker run --rm drl-scheduler:openeuler

docker build -f docker/Dockerfile.openkylin -t drl-scheduler:openkylin .
docker run --rm drl-scheduler:openkylin

docker build -f docker/Dockerfile.anolis -t drl-scheduler:anolis .
docker run --rm drl-scheduler:anolis

```

说明：

1. openEuler 镜像用于正式的国产操作系统运行证明；
2. openKylin 镜像用于额外的国产操作系统兼容性证明；
3. Anolis 镜像再补一组国内主流开源系统的编译/运行证据；
4. 如果后续需要展示 OpenHarmony 关联性，应另外展示设备侧任务描述/
   调用协议，或在 OpenHarmony 官方开发环境中做交叉编译验证；不能把
   本项目在 Linux 用户态容器中的运行表述为 OpenHarmony 原生设备运行。
