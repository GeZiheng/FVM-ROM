# FVM-ROM

基于 WSL2 Ubuntu 中预编译的 **OpenFOAM v2312（ESI 版）** 构建降阶模型（ROM）。ROM 的具体技术路线尚未确定；
目前本仓库提供项目骨架和一个 **OpenFOAM MCP 服务**，让智能体可以搭建、运行和检查算例。

## 目录结构

```
src/fvmrom/
  config.py          配置（可通过环境变量覆盖）
  foam/              Windows -> WSL2 的 OpenFOAM 访问层（MCP 与 ROM 代码共用）
    wsl.py           在 WSL 中运行 bash 脚本，路径映射 D:\ <-> /mnt/d
    commands.py      为单个 OpenFOAM 程序生成安全的运行脚本
    case.py          算例目录：列出 / 读取 / 写入 / 摘要
    jobs.py          后台任务（基于文件的状态）
    logs.py          求解器日志的尾部读取与残差解析
    tutorials.py     列出 / 复制 $FOAM_TUTORIALS 中的教程算例
  mcp_server/        FastMCP stdio 服务（入口：fvmrom-mcp）
  rom/               ROM 代码（占位）
cases/               OpenFOAM 算例（Windows 与 WSL 均可访问）
data/                ROM 数据：快照、基、模型（已被 git 忽略）
docs/                架构与路线图
tests/               测试
scripts/             脚本（如 MCP 端到端冒烟测试）
opencode.json        向 OpenCode 注册 MCP 服务
```

## 安装

前置条件：Windows 已安装 [uv](https://docs.astral.sh/uv/)；WSL2 中 OpenFOAM v2312 安装在
`/usr/lib/openfoam/openfoam2312`。

```powershell
uv sync                 # MCP 服务 + 开发工具
uv sync --extra rom     # 额外安装 ROM 所需的 numpy/scipy
uv run pytest           # 单元测试 + 真实 WSL 集成测试；用 -m "not wsl" 可跳过后者
```

OpenCode 会通过 `opencode.json` 自动加载该服务（名称：`openfoam`）。手动运行：
`uv run fvmrom-mcp`。

## 配置（环境变量）

| 变量 | 默认值 |
| --- | --- |
| `FVMROM_WSL_DISTRO` | WSL 默认发行版 |
| `FVMROM_FOAM_BASHRC` | `/usr/lib/openfoam/openfoam2312/etc/bashrc` |
| `FVMROM_CASES_DIR` | `<项目>/cases` |
| `FVMROM_STATE_DIR` | `<项目>/.fvmrom`（任务状态、生成的脚本） |

## MCP 工具

- 环境：`foam_env_info`、`list_tutorials`
- 算例：`copy_tutorial`、`list_cases`、`case_summary`、`list_case_files`、`read_case_file`、
  `write_case_file`、`foam_dict_get`、`foam_dict_set`
- 运行：`run_foam_app`（前台运行，可用 `np` 指定 MPI 进程数）、`start_foam_job`、`job_status`、
  `list_jobs`、`stop_job`
- 日志：`read_log_tail`、`parse_solver_log`
