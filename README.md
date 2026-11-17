# FVM-ROM

基于 WSL2 Ubuntu 中预编译的 **OpenFOAM v2312（ESI 版）** 构建降阶模型（ROM）。ROM 技术路线定为三阶段：

1. **阶段 1**：稳态参数化场重建（已实现，见下文）
2. **阶段 2**：瞬态时间外推（未开始）
3. **阶段 3**：几何泛化（未开始）

详见 `docs/roadmap.md`。此外本仓库提供一个 **OpenFOAM MCP 服务**，让智能体可以搭建、运行和检查算例。

## 目录结构

```
src/fvmrom/
  config.py          配置（可通过环境变量覆盖）
  params.py          参数采样（Re 的对数 / 线性采样）
  foam/              Windows -> WSL2 的 OpenFOAM 访问层（MCP 与 ROM 代码共用）
    wsl.py           在 WSL 中运行 bash 脚本，路径映射 D:\ <-> /mnt/d
    commands.py      为单个 OpenFOAM 程序生成安全的运行脚本
    case.py          算例目录：列出 / 读取 / 写入 / 摘要
    jobs.py          后台任务（基于文件的状态）
    logs.py          求解器日志的尾部读取与残差解析
    tutorials.py     列出 / 复制 $FOAM_TUTORIALS 中的教程算例
  problems/          算例生成：cavity、cylinder 的网格模板、边界条件与 Allrun
  rom/               降阶模型：快照、POD 基、回归器（RBF / GPR / MLP）
  mcp_server/        FastMCP stdio 服务（入口：fvmrom-mcp）
cases/               OpenFOAM 算例（Windows 与 WSL 均可访问）
data/                ROM 数据：快照、模型、报告（已被 git 忽略）
docs/                架构、加权 POD、回归器与路线图
tests/               测试
scripts/             脚本：快照生成、拟合与评估、MCP 端到端冒烟测试
opencode.json        向 OpenCode 注册 MCP 服务
```

## 安装

前置条件：Windows 已安装 [uv](https://docs.astral.sh/uv/)；WSL2 中 OpenFOAM v2312 安装在
`/usr/lib/openfoam/openfoam2312`。

```powershell
uv sync                          # MCP 服务 + 开发工具（基础依赖含 numpy）
uv sync --extra rom              # 额外安装 scipy、scikit-learn
uv sync --extra rom --extra nn   # 再安装 torch，仅供 MLP 回归器使用
uv run pytest                    # 共 162 个测试；-m "not wsl" 可跳过真实 WSL 集成测试
```

若 MCP 服务正占用 `.venv\Scripts\fvmrom-mcp.exe`，`uv run` 会在 sync 阶段失败，此时改用
`uv run --no-sync ...`（例如 `uv run --no-sync python scripts/fit_and_eval.py ...`）。

OpenCode 会通过 `opencode.json` 自动加载该服务（名称：`openfoam`）。手动运行：
`uv run fvmrom-mcp`。

## 配置（环境变量）

| 变量 | 默认值 |
| --- | --- |
| `FVMROM_ROOT` | 项目根目录（由包位置推导） |
| `FVMROM_WSL_DISTRO` | WSL 默认发行版 |
| `FVMROM_FOAM_BASHRC` | `/usr/lib/openfoam/openfoam2312/etc/bashrc` |
| `FVMROM_CASES_DIR` | `<项目>/cases` |
| `FVMROM_DATA_DIR` | `<项目>/data`（快照、模型、报告的根目录） |
| `FVMROM_STATE_DIR` | `<项目>/.fvmrom`（任务状态、生成的脚本） |

## MCP 工具

- 环境：`foam_env_info`、`list_tutorials`
- 算例：`create_case`、`copy_tutorial`、`list_cases`、`case_summary`、`list_case_files`、`read_case_file`、
  `write_case_file`、`foam_dict_get`、`foam_dict_set`
- 运行：`run_foam_app`（前台运行，可用 `np` 指定 MPI 进程数）、`start_foam_job`、`job_status`、
  `list_jobs`、`stop_job`
- 日志：`read_log_tail`、`parse_solver_log`

## 阶段 1：稳态参数化场重建

非侵入式降阶：先用 `simpleFoam` 生成稳态流场快照，再做 POD 基（U、p、phi 分场；U、p 体积加权，
phi 单位权重），最后用 RBF / GPR / MLP 回归器由 Re 预测基系数并重建场。

- 算例：cavity（64×64，$\mathrm{Re} \in [100, 1000]$ 对数采样）、圆柱绕流（12 块 O 型网格、32768 单元，
  $\mathrm{Re} \in [10, 40]$ 线性采样）
- 物理：层流、稳态 `simpleFoam`，参数只有 Re

### 生成快照

```powershell
uv run python scripts/generate_snapshots.py --problem cavity   --study re1d --n-train 30 --n-test 10
uv run python scripts/generate_snapshots.py --problem cylinder --study re1d --n-train 30 --n-test 10
```

可选参数：`--max-parallel 16`（并行任务数）、`--poll-s 5`（轮询间隔）、`--seed 0`（采样种子）、
`--resume`（跳过已收敛样本）。

每个算例的 `Allrun` 依次执行 `blockMesh`、`checkMesh`、`simpleFoam`、
`postProcess -func writeCellVolumes -latestTime`；`log.checkMesh` 输出含 `Failed` 时该任务判为失败。

### 拟合与评估

```powershell
uv run python scripts/fit_and_eval.py --problem cavity   --study re1d --regressors rbf,gpr,mlp --learning-curve --write-fields 2
uv run python scripts/fit_and_eval.py --problem cylinder --study re1d --regressors rbf,gpr,mlp --learning-curve --write-fields 2
```

可选参数：`--energy-tol 1e-10`（POD 能量截断阈值）、`--n-modes U=10,p=10,phi=10`（各场模态数）、
`--sizes 5,8,12,16,20,30`（学习曲线的样本数）、`--mlp-seeds 5`（MLP 多种子稳定性统计）。

### 数据布局

```
cases/<problem>_<study>/run_xxxx/                   训练 / 测试算例（含 run_xxxx.foam、log.checkMesh 等）
cases/<problem>_<study>_rom_<regressor>/run_xxxx/   ROM 算例
data/<problem>/<study>/snapshots.npz                快照与 POD 基
data/<problem>/<study>/samples.json                 样本参数、收敛状态
data/<problem>/<study>/report.json                  误差与学习曲线报告
data/<problem>/<study>/models/*.pkl                 回归器模型
```

ROM 算例的时间目录 `1/` 含 `U p phi U_fom p_fom U_err p_err`，误差为 $\text{ROM} - \text{FOM}$；在 ParaView 中打开其中的
`.foam` 文件即可同时对比 FOM、ROM 与误差场。

关于 `samples.json`：cylinder 样本带 `cd`（取自 `forceCoeffs` 的最终 Cd，仅作合理性记录）；`n_faces` 是
`phi` 向量的长度（内部面 + 非 empty patch 的面），并非网格总面数；`--resume` 跳过的样本 `job_id` 为空、
`message` 为 `skipped: already converged`。

### 注意事项

- `--study` 命名：以字母或数字开头，仅含 `[A-Za-z0-9_.-]`；不得含 `..`，不得以 `.` 或空格结尾，不得是
  Windows 保留名（`CON`、`PRN`、`AUX`、`NUL`、`COM1`–`COM9`、`LPT1`–`LPT9`，不区分大小写，含 `CON.txt`
  形式）。与已有的 `cases/<problem>_<study>` 或 `data/<problem>/<study>` 仅大小写不同也视为冲突。
- `--resume` 只判断"日志已收敛且 `V` 存在"，不检查算例是否由当前模板生成；模板改动后请不带 `--resume`
  重新生成。不带 `--resume` 时会清空已有算例重跑，且清理 / 重写前会先终止该算例在 WSL 中仍在运行的任务。
  `run_sweep` 异常或中断时会停止已启动的后台任务，停止失败则抛出 `RuntimeError`（列出 job id）。
- 模型文件为 pickle，只加载本地可信文件；`save` 记录依赖版本，`load` 版本不一致会给出警告。

### 已验证结果（测试集 U 平均相对 L2 误差）

| 算例 | RBF | GPR | MLP |
| --- | --- | --- | --- |
| cavity | 1.2e-3 | 4.7e-5 | 1.4e-3 |
| cylinder | 2.2e-4 | 2.3e-5 | 9.2e-5 |

- 重建 `phi` 的最大单元散度约 1e-11，与 FOM 同量级。
- 30 样本时 MLP 比 GPR 差约 4–29 倍；学习曲线显示 $n \le 8$ 时 MLP / RBF 反而优于 GPR，$n \ge 12$ 后 GPR 领先。
- 圆柱的 Cd 随 Re 单调下降（Re=10.1 时 3.02，Re=39.4 时 1.63）。

### 已知限制

- 质量守恒指标只保证 `phi` 的离散无散度，不保证与重建 U 的通量一致（阶段 2 再评估）。
- `phi` 的内积用单位权重，仅适用于均匀网格（阶段 3 需要改）。
- 圆柱域 $x \in [-10, 25]$、$y \in [-10, 10]$ （阻塞比约 5%），未做域 / 网格无关性验证。
- Cd 仅作合理性记录；MLP 报告为单种子（种子 0），多种子仅用于均值与标准差的稳定性统计。
- GPR 拟合时 sklearn 的 `ConvergenceWarning` 属正常现象。

`uv run pytest` 运行全部 162 个测试；`uv run python scripts/smoke_mcp.py` 对 MCP 服务做端到端检查
（cavity 算例）。
