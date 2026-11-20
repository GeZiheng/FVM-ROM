# FVM-ROM – 智能体须知

目标：基于 OpenFOAM v2312（ESI 版，apt 安装，WSL2 Ubuntu）构建降阶模型（ROM）。ROM 技术路线已确定为
三阶段（非侵入式稳态场重建 → 瞬态时间外推 → 几何泛化，详见 `docs/roadmap.md`）：阶段 1 与阶段 2 已实现，
阶段 3（几何泛化）及后续验证算例 M6–M8 待讨论。后续阶段的技术方案若有变化，仍需先与用户讨论再动手。

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
- ROM 数据（快照、基、模型）放在 `data/`，不要放进 `cases/`。阶段 1 布局：
  `data/<problem>/<study>/{snapshots.npz,samples.json,report.json,models/*.pkl}`；FOM 算例在
  `cases/<problem>_<study>/run_xxxx/`，ROM 算例在 `cases/<problem>_<study>_rom_<regressor>/run_xxxx/`。
  阶段 2 布局：`data/cylinder_transient/<study>/` 下放 `snapshots.npz`、`forces.npz`、`meta.json`、
  `report.json`、`opinf_lambda_scan.json` 与 `models/<method>.pkl`（pilot study 只有 `forces.npz` 与
  `meta.json`）；FOM 算例在 `cases/cylinder_transient_<study>/run_0000/`，ROM 算例在
  `cases/cylinder_transient_<study>_rom_<method>/run_0000/`。正式 study 为 `re100`，pilot 为 `pilot_base`、
  `pilot_wide`、`pilot_fine`；`re100` 的快照约 639 MB、FOM 算例约 2.2 GB。`FVMROM_DATA_DIR` 可改数据根目录。
- 脚本入口：阶段 1 的 `scripts/generate_snapshots.py`（采样 + 生成快照）与 `scripts/fit_and_eval.py`（拟合与
  评估）；阶段 2 的 `scripts/generate_transient.py`（运行 FOM 并收集快照）、`scripts/shedding_report.py`
  （pilot 对比表）、`scripts/fit_and_eval_transient.py`（训练与评估）、`scripts/opinf_lambda_scan.py`
  （λ 敏感性只读诊断）。
- `src/fvmrom/` 分层：`core/`（`config` 配置、`params` 参数采样；`core/__init__.py` 刻意留空、不重导出，
  避免只装基础依赖的 `foam` 与 MCP 被迫导入 scipy）、`foam/`（OpenFOAM 访问层，含网格几何、阶段 1 扫描与
  阶段 2 瞬态运行）、`case_templates/`（算例模板：cavity、cylinder、cylinder_transient）、`rom/`（加权 POD、
  回归器、稳态 ROM；瞬态侧为潜空间动力学、读出映射与瞬态评估，见 `docs/transient.md`）。导入用完整路径（如
  `from fvmrom.core.config import ...`）；旧路径 `fvmrom.config` / `fvmrom.params` / `fvmrom.problems` 已不存在。
- MCP 服务是 harness 接入层的独立脚本：`.opencode/mcps/<功能>/server.py`（当前 `openfoam`），由
  `opencode.json` 以 `uv run --project . python .opencode/mcps/<功能>/server.py` 启动。它用绝对导入
  （`from fvmrom.foam import ...`）调用库，不是 `src/` 里的包，也没有 console script。
- 旧入口（`fvmrom.mcp_openfoam` 包、`openfoam-mcp.exe`）曾被 `uv run` 的 sync 阶段占用而报错；现在服务以
  `python` 运行脚本，不再占用 `.venv\Scripts` 下的 exe，一般可直接 `uv run`。若仍遇到 sync 失败，再用
  `uv run --no-sync ...`。`.venv\Scripts\` 下若残留旧的 `openfoam-mcp.exe`，下次 `uv sync` 会清理，不影响使用。
- ESI v2312 的湍流字典是 `constant/turbulenceProperties`（不是 `momentumTransport`）；算例用 ASCII 格式
  （`fvmrom.foam.fields` 只读 ASCII）。`fvmrom.foam` 依赖 numpy。
- `--study` 名需符合命名规则（见 README）。
- 阶段 1 的默认行为不变：`phi` 内积默认仍用单位权重；`scripts/fit_and_eval.py --phi-weights area` 是可选开关，
  报告里多一项 U–`phi` 一致性（`flux_consistency`）。阶段 2 的瞬态 ROM 默认用面积权重。
- 阶段 2 的瞬态算例是两段式（spin-up 写检查点、生产段按 `snapshot_dt` 写快照）；`--purge-fields` 会删掉
  spin-up 检查点与中间快照，清理后不能再 `--collect-only`，要重新跑生产段才能再收集。
- `uv run pytest` 运行单元测试和真实 WSL 集成测试，当前共 380 个测试（`-m "not wsl"` 为 370，跳过 10 个
  WSL 集成测试）。
- `uv run python scripts/smoke_mcp.py` 对 MCP 服务做端到端测试（cavity 算例）。
- 项目文档（README、AGENTS.md、docs/）使用中文撰写。

## 子代理
`explore`、`implementer`、`doc-writer` 使用 `deepseek` provider 下的 `deepseek-flash`；`reviewer` 使用 `opencode` provider 的 `claude-haiku-5-5`（与实现者不同家族，审查更独立；免费额度用完后改用）。定义在 `.opencode/agents/`，`explore` 的模型在 `opencode.json` 中覆盖。
子代理上下文是全新的，任务描述里必须写清楚背景、计划文件路径、涉及的文件。

| 子代理 | 模型 | 职责与权限 |
| --- | --- | --- |
| `explore`（内置） | `deepseek/deepseek-flash` | 只读探索代码库 |
| `implementer` | `deepseek/deepseek-flash` | 按计划实现代码和测试；不改文档、`.opencode/**`，不做 git 提交 |
| `reviewer` | `opencode/claude-haiku-5-5` | 只读审查计划和实现；可运行 `uv run --no-sync pytest`，不改任何文件 |
| `doc-writer` | `deepseek/deepseek-flash` | 只能改 `README.md`、`AGENTS.md`、`docs/**` |

补充经验：

- `glob` / `grep` 默认不显示隐藏目录（需 `hidden: true`），且遵守 `.gitignore`：搜不到 `data/`、`cases/`、
  `.opencode/`。子代理读计划文件、`data/**` 时要用 `read` 直接读完整路径；派活时把完整路径写进任务描述。
- 大任务容易让 `implementer`、`doc-writer` 步数耗尽（`doc-writer` 的 `steps` 已提到 60）：拆成小批，要求先
  把核心做完再收尾；临时文件放 `C:\Users\DELL\AppData\Local\Temp\opencode\`，结束前删除。
- reviewer 运行测试直接执行 `uv run --no-sync pytest ...`，不要加 `cd` 前缀，也不要用 `;` / `&&` 组合命令
  （组合命令会被拆开逐条检查权限，容易被误拒）。

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
