# FVM-ROM – 智能体须知

目标：基于 OpenFOAM v2312（ESI 版，apt 安装，WSL2 Ubuntu）构建降阶模型（ROM）。ROM 技术路线**尚未确定**——
在构建 `src/fvmrom/rom/` 之前，必须先与用户讨论。

## 环境事实
- 宿主机：Windows；Python 由 `uv` 管理（`uv sync --extra rom`、`uv run ...`）。不要依赖 WSL 里的 Python
  （没有 numpy，也没有 venv）。
- OpenFOAM：位于 WSL 内的 `/usr/lib/openfoam/openfoam2312`（`source <该路径>/etc/bashrc`），双精度（DP），
  label 32 位，OpenMPI 4.1.2，20 核 / 31 GB。
- 项目位于 `D:\Projects\FVM-ROM`，在 WSL 中对应 `/mnt/d/Projects/FVM-ROM`。算例放在 `cases/`。
- 宿主机的 shell 是 PowerShell；避免通过 `wsl -e bash -c` 传递复杂引号。请改用 MCP 服务或
  `fvmrom.foam.Wsl`（基于脚本文件）。

## 约定
- 通过 `openfoam` MCP 工具（或 `fvmrom.foam`）运行 OpenFOAM，不要临时拼 `wsl` 命令。
- 求解器/程序的输出写入 `<case>/log.<app>`。
- `np>1` 时算例必须已完成分解，且 `np` 必须等于 `numberOfSubdomains`。
- ROM 数据（快照、基、模型）放在 `data/`，不要放进 `cases/`。
- `uv run pytest` 会运行单元测试和真实 WSL 集成测试（`-m "not wsl"` 跳过后者）。
- `uv run python scripts/smoke_mcp.py` 对 MCP 服务做端到端测试（cavity 算例）。
- 项目文档（README、AGENTS.md、docs/）使用中文撰写。

## 文档维护
- 项目文档由 `doc-writer` 子代理负责更新和维护（定义在 `.opencode/agents/doc-writer.md`，模型
  `deepseek/deepseek-flash`，只能修改 `README.md`、`AGENTS.md`、`docs/**`）。
- 完成会影响文档的变更后（新增/修改 MCP 工具、目录结构、环境变量、约定、路线图进展），应调用它同步文档。
  子代理上下文是全新的，任务描述里要写清楚改了什么、涉及哪些文件。
