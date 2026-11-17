# FVM-ROM – 智能体须知

目标：基于 OpenFOAM v2312（ESI 版，apt 安装，WSL2 Ubuntu）构建降阶模型（ROM）。ROM 技术路线已确定为
三阶段（非侵入式稳态场重建 → 瞬态时间外推 → 几何泛化，详见 `docs/roadmap.md`），阶段 1 已实现。
后续阶段的技术方案若有变化，仍需先与用户讨论再动手。

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
- ROM 数据（快照、基、模型）放在 `data/`，不要放进 `cases/`。布局：
  `data/<problem>/<study>/{snapshots.npz,samples.json,report.json,models/*.pkl}`；FOM 算例在
  `cases/<problem>_<study>/run_xxxx/`，ROM 算例在 `cases/<problem>_<study>_rom_<regressor>/run_xxxx/`。
  `FVMROM_DATA_DIR` 可改数据根目录。
- 两个脚本入口：`scripts/generate_snapshots.py`（采样 + 生成快照）、`scripts/fit_and_eval.py`（拟合与评估）。
- MCP 服务占用 `.venv\Scripts\fvmrom-mcp.exe` 时 `uv run` 会因 sync 失败而报错，改用
  `uv run --no-sync ...`。
- ESI v2312 的湍流字典是 `constant/turbulenceProperties`（不是 `momentumTransport`）；算例用 ASCII 格式
  （`fvmrom.foam.fields` 只读 ASCII）。`fvmrom.foam` 依赖 numpy。
- `--study` 名需符合命名规则（见 README）。
- `uv run pytest` 运行单元测试和真实 WSL 集成测试，当前共 162 个测试（`-m "not wsl"` 跳过 WSL 集成测试）。
- `uv run python scripts/smoke_mcp.py` 对 MCP 服务做端到端测试（cavity 算例）。
- 项目文档（README、AGENTS.md、docs/）使用中文撰写。

## 子代理
`explore`、`implementer`、`doc-writer` 使用 `deepseek` provider 下的 `deepseek-flash`；`reviewer` 使用 `opencode` provider 的 `claude-haiku-5-5`（与实现者不同家族，审查更独立；免费额度用完后改用）。定义在 `.opencode/agents/`，`explore` 的模型在 `opencode.json` 中覆盖。
子代理上下文是全新的，任务描述里必须写清楚背景、计划文件路径、涉及的文件。

| 子代理 | 模型 | 职责与权限 |
| --- | --- | --- |
| `explore`（内置） | `deepseek/deepseek-flash` | 只读探索代码库 |
| `implementer` | `deepseek/deepseek-flash` | 按计划实现代码和测试；不改文档、`.opencode/**`，不做 git 提交 |
| `reviewer` | `opencode/claude-haiku-5-5` | 只读审查计划和实现；可运行 `uv run pytest`，不改任何文件 |
| `doc-writer` | `deepseek/deepseek-flash` | 只能改 `README.md`、`AGENTS.md`、`docs/**` |

## 开发流程
每个开发任务按四个阶段推进，**阶段切换都需要用户确认**，不得自行跳过。

1. **Phase 1 计划**：主 agent 分析需求，按 `.opencode/plans/_template.md` 写详细计划，保存为
   `.opencode/plans/YYYYMMDD-<英文短名>.md`。需求或技术路线有疑问时先问用户。计划写完后请用户确认；
   用户确认后将计划状态改为"已确认"，进入 Phase 2。
2. **Phase 2 实现**：调用 `implementer`，传入计划文件路径，按计划实现并测试。若它报告计划不清晰或不合理，
   回到 Phase 1 修改计划（在计划的"修订记录"中记一笔），重新请用户确认后再实现。实现完成后进入 Phase 3。
3. **Phase 3 审查**：调用 `reviewer`，传入计划路径和 implementer 的报告，审查计划与实现是否合理。
   审查发现的问题**如实反馈给用户**，由用户决定是否回到 Phase 1 调整计划。用户确认不再修改后进入 Phase 4。
   主 agent 不得替用户决定忽略或采纳审查意见。
4. **Phase 4 文档**：调用 `doc-writer`，传入变更说明和 implementer / reviewer 报告里的"文档影响"。
   把文档改动告知用户；用户确认不需要进一步修改后，任务结束（计划状态改为"已完成"）。

补充说明：
- 计划状态由主 agent 维护，随阶段推进更新：草稿（Phase 1）→ 已确认（用户确认计划）→ 实现中（调用
  `implementer` 前）→ 已实现（`implementer` 返回后）→ 评审中（Phase 3）→ 文档中（Phase 4）→ 已完成；
  用户否决或终止任务时改为已取消。回到 Phase 1 修改计划时状态退回草稿。
- 计划文件是流程的唯一依据，所有修订都要写回计划，不要只在对话里说。
- 琐碎改动（如改错别字、单行修复）可以与用户商定后跳过流程，但需要用户明确同意。
- 主 agent 自己不直接实现计划内的代码，实现交给 `implementer`。
