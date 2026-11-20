# FVM-ROM

基于 WSL2 Ubuntu 中预编译的 **OpenFOAM v2312（ESI 版）** 构建降阶模型（ROM）。ROM 技术路线定为三阶段：

1. **阶段 1**：稳态参数化场重建（已实现，见下文）
2. **阶段 2**：瞬态时间外推（已实现，见下文；细节见 `docs/transient.md`）
3. **阶段 3**：几何泛化（未开始）

详见 `docs/roadmap.md`。此外本仓库提供一个 **OpenFOAM MCP 服务**，让智能体可以搭建、运行和检查算例。

## 目录结构

```
src/fvmrom/
  core/              基础模块：配置与参数采样（不重导出，见 docs/architecture.md）
    config.py        配置（可通过环境变量覆盖）
    params.py        参数采样（Re 的对数 / 线性采样）
  foam/              Windows -> WSL2 的 OpenFOAM 访问层（MCP 与 ROM 代码共用）
    wsl.py           在 WSL 中运行 bash 脚本，路径映射 D:\ <-> /mnt/d
    commands.py      为单个 OpenFOAM 程序生成安全的运行脚本
    case.py          算例目录：列出 / 读取 / 写入 / 摘要
    jobs.py          后台任务（基于文件的状态）
    logs.py          求解器日志的尾部读取与残差解析
    tutorials.py     列出 / 复制 $FOAM_TUTORIALS 中的教程算例
    fields.py        ASCII 场读写、网格拓扑、phi 散度
    mesh.py          网格几何量与 phi 的面积权重
    sweep.py         阶段 1 的参数扫描与快照收集
    transient.py     阶段 2 的瞬态运行、续算与快照收集
  case_templates/    算例模板：cavity、cylinder 与瞬态圆柱 cylinder_transient
  rom/               降阶模型：加权 POD、回归器、稳态 / 瞬态 ROM 与潜空间动力学
.opencode/
  mcps/openfoam/server.py  OpenFOAM MCP 服务脚本（stdio；harness 接入层，不属于 src/）
cases/               OpenFOAM 算例（Windows 与 WSL 均可访问）
data/                ROM 数据：快照、模型、报告（已被 git 忽略）
docs/                架构、加权 POD、回归器、瞬态 ROM 与路线图
tests/               测试
scripts/             脚本：快照生成、拟合与评估、瞬态流程、MCP 端到端冒烟测试
opencode.json        向 OpenCode 注册 MCP 服务（名称 openfoam）
```

## 安装

前置条件：Windows 已安装 [uv](https://docs.astral.sh/uv/)；WSL2 中 OpenFOAM v2312 安装在
`/usr/lib/openfoam/openfoam2312`。

```powershell
uv sync                          # MCP 服务 + 开发工具（基础依赖含 numpy）
uv sync --extra rom              # 额外安装 scipy、scikit-learn
uv sync --extra rom --extra nn   # 再安装 torch，仅供 MLP 回归器使用
uv run pytest                    # 共 380 个测试；-m "not wsl" 为 370，跳过 10 个真实 WSL 集成测试
```

此前 MCP 服务是 console script，占用 `.venv\Scripts\openfoam-mcp.exe` 会让 `uv run` 在 sync 阶段失败；
现在服务以 `python` 运行脚本，一般可直接 `uv run`。若仍遇到 sync 失败，再用 `uv run --no-sync ...`
（例如 `uv run --no-sync python scripts/fit_and_eval.py ...`）。`.venv\Scripts\` 下若残留旧的
`openfoam-mcp.exe`，下次 `uv sync` 会清理，不影响使用。

OpenCode 会通过 `opencode.json` 自动加载该服务（名称：`openfoam`，启动命令
`uv run --project . python .opencode/mcps/openfoam/server.py`）。手动运行：
`uv run python .opencode/mcps/openfoam/server.py`。

命名约定：服务脚本放在 `.opencode/mcps/<功能>/server.py`，OpenCode 服务名为 `<功能>`，两者一一对应
（当前 `<功能>` 为 `openfoam`；将来新增 ROM 服务即 `.opencode/mcps/rom/server.py`、服务名 `rom`）。
服务脚本属于 harness 接入层，不放在 `src/`，也不注册 console script（没有 `<功能>-mcp` 命令，也没有
`fvmrom.mcp_<功能>` 包）。

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
phi 默认单位权重，可用 `--phi-weights area` 改面积权重），最后用 RBF / GPR / MLP 回归器由 Re 预测基系数
并重建场。

- 算例：cavity（64×64， $\mathrm{Re} \in [100, 1000]$ 对数采样）、圆柱绕流（12 块 O 型网格、32768 单元，
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
`--phi-weights {unit,area}`（`phi` 内积权重，默认 `unit`；`area` 用网格面积权重）、
`--sizes 5,8,12,16,20,30`（学习曲线的样本数）、`--mlp-seeds 5`（MLP 多种子稳定性统计）。
报告里额外给出 U–`phi` 一致性（`flux_consistency`：FOM 基线 / ROM / 比值）。

### 数据布局

```
cases/<problem>_<study>/run_xxxx/                   训练 / 测试算例（含 run_xxxx.foam、log.checkMesh 等）
cases/<problem>_<study>_rom_<regressor>/run_xxxx/   ROM 算例
data/<problem>/<study>/snapshots.npz                快照与 POD 基
data/<problem>/<study>/samples.json                 样本参数、收敛状态
data/<problem>/<study>/report.json                  误差与学习曲线报告
data/<problem>/<study>/models/*.pkl                 回归器模型
```

ROM 算例的时间目录 `1/` 含 `U p phi U_fom p_fom U_err p_err`，误差为 $\text{ROM} - \text{FOM}$ ；在 ParaView 中打开其中的
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
- `phi` 面积权重（`--phi-weights area`）对 cavity 无影响（均匀网格）；cylinder 上 `phi` 误差降低约 3–4%。
  FOM 自身的 U–`phi` 一致性基线：cavity 8.0e-3、cylinder 1.2e-4。
- 30 样本时 MLP 比 GPR 差约 4–29 倍；学习曲线显示 $n \le 8$ 时 MLP / RBF 反而优于 GPR， $n \ge 12$ 后 GPR 领先。
- 圆柱的 Cd 随 Re 单调下降（Re=10.1 时 3.02，Re=39.4 时 1.63）。

### 已知限制

- 质量守恒指标只保证 `phi` 的离散无散度，不保证与重建 U 的通量一致；`flux_consistency` 目前只报告、不设阈值。
- `phi` 的内积默认仍用单位权重（只适用于均匀网格）；`--phi-weights area` 改用面积权重，默认行为不变。
  阶段 2 的瞬态 ROM 默认用面积权重（见 `docs/transient.md`）。
- 圆柱域 $x \in [-10, 25]$ 、 $y \in [-10, 10]$ （阻塞比约 5%）；阶段 1 的稳态算例未做域 / 网格无关性验证
  （阶段 2 在 Re=100 瞬态上做了 pilot，见 `docs/transient.md`）。
- Cd 仅作合理性记录；MLP 报告为单种子（种子 0），多种子仅用于均值与标准差的稳定性统计。
- GPR 拟合时 sklearn 的 `ConvergenceWarning` 属正常现象。

## 阶段 2：瞬态时间外推

非侵入式时间外推：网格和 Re 固定，用前 3 个周期（90 个快照）训练，之后不再调用 OpenFOAM。ROM 只在 `U`
的潜空间（默认 8 个模态）上做动力学——标准 DMD、Optimized DMD、BOP-DMD 作基线，Operator Inference
（二次形式）为主方法；`p`、`phi` 与 Cd/Cl 由潜变量经多项式读出映射预测，`phi` 用面积权重并保持离散质量
守恒。算法、指标与完整结果见 `docs/transient.md`。

- 算例：二维圆柱绕流，Re=100，层流 `pimpleFoam`，base 网格 32768 单元
- 运行：两段式（spin-up 到 t=60 后每 0.2 个时间单位写快照），生产段 t=60–160，共 501 个快照（约 17 个周期）

### 生成 FOM 与快照

```powershell
uv run python scripts/generate_transient.py --study re100 --t-spin 60 --t-end 160
# 收集完成后再清理中间场；清理后不能再 --collect-only
uv run python scripts/generate_transient.py --study re100 --t-spin 60 --t-end 160 --collect-only --purge-fields
```

可选参数：`--pilot`（只跑到 `t_spin`，只看力系数）、`--resume`（续算未完成的算例）、`--collect-only`、
`--purge-fields`、`--checkpoint-dt`（spin-up 检查点间隔）、`--workers`（快照读取进程数）、
`--geometry {base,wide,fine}`。续算前会校验算例字典与当前配置一致。

### 训练与评估

```powershell
uv run python scripts/fit_and_eval_transient.py --study re100 `
    --methods dmd,optdmd,bopdmd,opinf,opinf_norecheck --train-periods 3
```

可选参数：`--n-modes U=8,p=8,phi=8`（`auto` 交给 `--energy-tol`）、`--sweep-modes 4,6,8,10,12`、
`--sweep-train-periods 2,3,4,5`（只报告、不设阈值）、`--write-fields N`、`--long-periods 20`（超出数据的
潜空间稳定性检查）、`--no-diagnostics`（跳过标量读出阶 1 诊断）。`scripts/shedding_report.py --studies
pilot_base,pilot_wide,pilot_fine` 输出 pilot 对比表；`scripts/opinf_lambda_scan.py --study re100` 是 OpInf 的
λ 敏感性只读诊断。

### 数据布局

```text
cases/cylinder_transient_<study>/run_0000/                FOM 算例（re100 约 2.2 GB）
cases/cylinder_transient_<study>_rom_<method>/run_0000/   ROM 算例（--write-fields 生成）
data/cylinder_transient/<study>/snapshots.npz             快照（re100 约 639 MB）
data/cylinder_transient/<study>/forces.npz                全程 Cd/Cl（每步一行）
data/cylinder_transient/<study>/meta.json                 配置、网格、Courant、脱落统计、数据质量
data/cylinder_transient/<study>/report.json               各方法误差、验收与扫描
data/cylinder_transient/<study>/opinf_lambda_scan.json    λ 敏感性诊断
data/cylinder_transient/<study>/models/<method>.pkl       模型
```

pilot study（`pilot_base`、`pilot_wide`、`pilot_fine`）只有 `forces.npz` 与 `meta.json`。

### 已验证结果（`re100`，训练 3 个周期）

| 方法 | U τ≤5 最大误差 | Cl 幅值误差 | verdict |
| --- | --- | --- | --- |
| POD 投影下界 | 1.96e-3 | — | — |
| `dmd` | 2.53e-3 | 6.45% | FAIL（仅 Cl 幅值） |
| `optdmd` | 2.37e-3 | 0.41% | PASS |
| `bopdmd` | 2.36e-3 | 0.16% | PASS |
| `opinf` | 1.44e-2 | 0.27% | FAIL（仅 U 误差；不发散） |
| `opinf_norecheck` | NaN | NaN | FAIL（τ≈4.3 发散） |

后续验证算例按 M6（圆柱 Re=150–180，下一步，另开计划）→ M7（方腔非定常启动）→ M8（后向台阶，视 M6、M7
结果再定）的顺序推进，已写入计划，待确认后各自细化；M5（latent 神经 ODE）是有条件的备选，排在 M6 结果之后
再决定是否做。

### 已知限制

- 验收结论只适用于默认配置（U=8、3 个训练周期）；U=10、12 时 OptDMD/BOP 的 Cl 幅值误差约 9–19%；
  训练 4 个周期时为 1.97%–1.99%，贴着 2% 的限值。
- OpInf 的 U 误差未达标，对模态数与训练窗口敏感；λ 的留出段只有约 0.7 个周期。
- 标准 DMD 的 Cl 幅值 FAIL 主要来自二次读出放大潜空间误差，不是动力学本身。
- FOM 的 Cl 幅值有约 5% 的网格敏感性；能量与相位指标的口径细节见 `docs/transient.md`。

`uv run pytest` 运行全部 380 个测试（`-m "not wsl"` 为 370，跳过 10 个 WSL 集成测试）；
`uv run python scripts/smoke_mcp.py` 对 MCP 服务做端到端检查（cavity 算例）。
