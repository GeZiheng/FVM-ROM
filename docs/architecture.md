# 架构

```mermaid
flowchart LR
    subgraph WIN["Windows 宿主"]
        direction TB
        AGENT["OpenCode 智能体"]
        MCP["openfoam MCP 服务<br/>(.opencode/mcps/openfoam/server.py, stdio)"]
        SCRIPTS["scripts/*.py<br/>generate_snapshots / fit_and_eval<br/>generate_transient / fit_and_eval_transient"]
        ROM["fvmrom.core.params / case_templates / rom<br/>(numpy, scipy, sklearn, torch)"]
        FOAM["fvmrom.foam<br/>Wsl · Cases · Jobs · sweep / transient · fields · mesh"]
        DATA[("cases/ 与 data/<br/>D:\Projects\FVM-ROM")]

        AGENT -- stdio --> MCP
        MCP --> FOAM
        SCRIPTS --> ROM
        ROM --> FOAM
        FOAM -- "读写（ASCII 场、快照、模型）" --> DATA
    end

    subgraph WSL["WSL2 (Ubuntu)"]
        direction TB
        BASH["bash 脚本<br/>source OpenFOAM bashrc"]
        APPS["blockMesh / checkMesh / simpleFoam / pimpleFoam / mpirun"]
        MNT[("/mnt/d/Projects/FVM-ROM/<br/>cases/ 与 data/")]

        BASH --> APPS
        APPS -- 读写 --> MNT
    end

    FOAM -- "脚本文件<br/>wsl.exe --exec bash" --> BASH
    DATA <-. "同一目录（跨系统共享）" .-> MNT
```

两个入口并列，都依赖共享的 `fvmrom.foam`：`.opencode/mcps/openfoam/server.py` 提供的 MCP 服务面向智能体的
交互调试；ROM 脚本和库（采样、POD、回归、瞬态动力学、评估）直接导入 `fvmrom.foam`，**不经过 MCP 进程**。
`fvmrom.foam` 通过脚本文件驱动 WSL 里的 OpenFOAM，并在 Windows 侧直接读写 `cases/` 与 `data/`（WSL 中对应
`/mnt/d/Projects/FVM-ROM/...`）。`fvmrom.foam` 各模块的作用与调用关系见 `docs/foam.md`。

## 设计决策

- **MCP 服务运行在 Windows 侧。** 数值 ROM 技术栈统一在一个 Python 环境（uv）中；WSL 仅作为
  OpenFOAM 的执行后端。
- **用脚本文件，不用命令行。** 每次调用都会在 `.fvmrom/scripts/` 下写一个 LF 换行的脚本，再执行
  `wsl.exe --exec bash <script>`，避免 PowerShell/cmd/bash 的引号问题。
- **只运行 OpenFOAM 可执行文件。** `type -P` 的解析结果必须位于 `$WM_PROJECT_DIR`、
  `$FOAM_USER_APPBIN` 或 `$FOAM_SITE_APPBIN` 之下（或是算例内的脚本，如 `./Allrun`）。
  文件访问被限制在 `cases/<case>/` 内（`safe_join`）。
- **后台任务基于文件**（`.fvmrom/jobs/<id>/{meta.json,pid,exit_code}`），并运行在独立会话
  （`setsid`）中，因此 MCP 重启后，状态查询和 `stop_job` 依然有效。任务运行期间会保持 `wsl.exe`
  句柄打开，避免 WSL 认为发行版处于空闲状态。
- **算例放在 `/mnt/d`。** 简单，且两侧都可见。通过 9P 桥的文件 I/O 比 ext4 慢；如果写快照成为瓶颈，
  可将 `FVMROM_CASES_DIR` 指向 Windows 可访问的 `\\wsl.localhost\...` 路径，或把重计算挪到 WSL 文件系统。
- **MCP 服务是 harness 接入层的独立脚本。** 服务脚本放在 `.opencode/mcps/<功能>/server.py`（当前
  `openfoam`），用绝对导入（`from fvmrom.foam import ...`）调用库；`src/fvmrom/` 只保留 ROM 代码与
  OpenFOAM 访问层，不承担 OpenCode 服务注册。服务名与目录名一致（`<功能>`），也没有 `<功能>-mcp` 命令。
- **基础模块集中在 `fvmrom/core/`，且刻意不重导出。** `core.config`（`Settings`、`PROJECT_ROOT`）只依赖
  标准库，供 `fvmrom.foam` 使用；`core.params`（`ParamSpace`）依赖 scipy，由 `case_templates` 与 `rom`
  共用。`core/__init__.py` 留空：若在其中重导出 `params`，只装基础依赖的 `foam` 与 MCP 服务也会被迫导入
  scipy。依赖方向：`core.config` ← `foam` ← `case_templates`、`rom`；`core.params` ← `case_templates`、`rom`。
- **ROM 功能先做在 Python 库里，不动 MCP。** 批量采样、场读取、POD、回归、误差评估都由 `fvmrom`
  直接提供（`uv run ...` 即可使用）；MCP 服务保持为交互调试接口，确实需要时再把常用流程封装成工具。
- **瞬态算例单列模板，不套用 `Problem` 协议。** 阶段 2 是单一轨迹、没有参数空间，`CylinderTransientProblem`
  只负责写算例（`TransientConfig` 描述 Re、域、时间步与两段式时间点），因此不进 `get_problem`；网格与
  边界复用 `cylinder` 的模板函数。
- **训练 / 预测严格按时间切分。** 周期 `T` 只用于决定切分点；POD 基、读出映射、动力学与 λ 选择都只用
  训练窗口拟合，预测窗口的数据不参与，评估里另报 POD 投影误差作下界。
- **动力学只做在 `U` 的潜空间，其余量用读出映射。** `U` 的体积加权 POD 系数是状态；`p`、`phi` 与标量
  输出由它经多项式（默认 `p` 阶 2、`phi` 阶 1、标量阶 2）读出，`phi` 仍是模态的线性组合，保持离散
  无散度。细节见 [docs/transient.md](transient.md)。
- **两类动力学的初始条件处理不同。** `Dynamics` 协议统一为 `fit(t, A)` / `predict(t, a0, t0)` /
  `spectrum()`；DMD 家族对整条轨迹做全局拟合、`predict` 解析求值（忽略 `a0`、`t0`），OpInf 从给定状态
  积分（默认取训练窗口末的真实状态）。报告必须注明这一点，避免把初始条件差异当成动力学优劣。
- **瞬态算例两段式运行、可续算。** spin-up 只写检查点并用 `purgeWrite` 限盘，生产段按 `snapshot_dt` 写
  快照；`Allrun` 按最新时间跳过已完成的阶段，续算前逐文件比对算例字典，防止用不同配置接着算。
- **瞬态求解用 `pimpleFoam`（层流）。** `forceCoeffs` 需要求解器注册输运 / 湍流模型，`icoFoam` 不提供；
  `nOuterCorrectors 1` 即退化为 PISO。
- **验收按"报告式"判定。** 每项指标缺失或非有限一律 FAIL；唯一允许 SKIP 的是缺少网格时的 `phi` 散度，
  此时整个方法判 `INCOMPLETE`；每个方法给出 `verdict`（`PASS` / `FAIL` / `INCOMPLETE`），阈值不因结果放宽。
- **场 I/O 只支持 ASCII。** `fvmrom.foam.fields` 只处理 ASCII 文本场，二进制/压缩文件直接给出明确错误；
  因此阶段 1 的算例统一用 `writeFormat ascii`、`writePrecision 12`（同时 `purgeWrite 1` 只保留最新时间），
  阶段 2 的瞬态算例用 `writePrecision 10` 写快照。
- **数据目录与算例目录解耦。** `Settings.data_dir`（`FVMROM_DATA_DIR`，默认 `<项目>/data`）独立于
  `cases_dir`：算例随时清理重跑，快照与模型仍可保留。
- **`numpy` 放在基础依赖里**，因为 `fvmrom.foam.fields` 需要它，不应等到 `--extra rom` 才可用。
- **大文件不入库。** `.gitignore` 整体忽略 `cases/*` 与 `data/*`，仅保留目录下的 `.gitkeep`。

## ROM 实现现状

技术路线已确定：阶段 1 用加权 POD（`WeightedPOD`）+ 参数到模态系数的回归（RBF / GPR / MLP）构成稳态
ROM；阶段 2 用分场 POD + `U` 潜空间的动力学 + 读出映射构成瞬态 ROM。两者的实现都在 `src/fvmrom/` 内，
不依赖 MCP。

### 模块分工

- `fvmrom.core.params`：`ParamSpace` 描述参数（名称、上下界、对数标志），提供 `to_unit`/`from_unit`
  归一化与 Sobol / LHS 采样；Sobol 前缀嵌套（增量扩样时旧样本仍是新样本的子集），对数参数要求 > 0。
- `fvmrom.core.config`：`Settings.data_dir`（`FVMROM_DATA_DIR`，默认 `<项目>/data`）与 `cases_dir` 解耦。
- `fvmrom.foam.fields`：ASCII 场读写（`read_field`/`write_field`，`FoamField` 含 `dimensions`）、
  `latest_time`、网格拓扑 `read_topology`（owner/neighbour/boundary），以及 `phi_vector`/`phi_divergence`。
  计算 phi 散度时按 patch 的 `type == "empty"` 跳过（2D 算例的空方向），非 empty 却无数值的 patch 直接报错。
- `fvmrom.foam.mesh`：面 / 单元几何量（面心与面积矢量、体心与体积、内部面的 `linear` 插值权重），按
  `primitiveMeshTools` 的公式计算，与 OpenFOAM 写出的 `V` 在舍入误差内一致。`phi_weights` 给出 `phi`
  的近似单元体积 L2 权重：内部面 $(V_P + V_N)/(2\lVert\mathbf{S}_f\rVert^2)$ 、非 empty patch 面
  $V_P/\lVert\mathbf{S}_f\rVert^2$ 。只读几何，不生成网格。
- `fvmrom.foam.sweep`：参数扫描与快照收集。
  - `run_sweep`：并发执行各算例的 `./Allrun`，通过 `Jobs.status` 轮询进度，用 `parse_log` 判定收敛
    并要求 `V` 场存在；支持 `resume`；`cases`/`jobs` 可注入便于测试；重写已有算例前先逐段校验路径一致，
    并调用 `Jobs.stop_case` 停掉残留任务；`finally` 中停止本次启动的任务，停止失败抛 `RuntimeError`
    并列出 job id（`stop` 因尚无 pid 失败时先 `wait_pid` 不超过 $20$ s 再重试）。
  - `collect_snapshots`：写出 `snapshots.npz` 与 `samples.json`，逐一比较各样本的 owner/neighbour/patches
    以确保拓扑一致；`n_faces` 取 phi 向量长度；为每个算例写 `<算例名>.foam` 便于 ParaView 打开。
    cylinder 的阻力系数 `cd` 取 `postProcessing/forceCoeffs` 下数值最大时间目录的 `coefficient.dat`
    最后一行第 2 列。
  - `validate_study`：校验算例命名规则、保留名与大小写冲突。
- `fvmrom.foam.transient`：阶段 2 的单轨迹运行与快照收集。`run_transient` 跑两段式 `./Allrun`（可重入，
  `resume` 时校验算例字典），以日志中 `End` 的次数判断完成；`read_forces` 合并各重启目录的
  `coefficient.dat`，可把 6 位有效数字的时间列吸附回 $k\,dt$ ；`collect_transient` 选时间目录、检查间隔
  均匀与缺失、并行读 `U/p/phi` 并与力系数对齐，写 `snapshots.npz`；`inspect_run` 只检查已有算例。
- `fvmrom.foam.jobs`：新增 `Jobs.wait_pid` 与 `Jobs.stop_case`——终止属于该算例的运行中/启动中任务的进程组
  （含 WSL 里的 simpleFoam/MPI 子进程），轮询到全部退出，超时抛错。
- `fvmrom.foam.logs`：FPE 判定改为行首匹配 `^Floating point exception`，避免把启动 banner
  `trapFpe: Floating point exception trapping enabled` 误判为崩溃。
- `fvmrom.case_templates`（算例模板）：`Problem` 协议（`name`、`space`、`fields`、`write_case`）与 `get_problem`，内置两个：
  - `cavity`：64×64×1， $\nu = 1/\mathrm{Re}$ ，`pRefCell 0`，SIMPLEC（`consistent yes`）。
  - `cylinder`： $D = 1$ ，12 块 O 型网格、32768 单元，域 $x \in [-10, 25]$ 、 $y \in [-10, 10]$ ；inlet 固定速度、outlet 固定压力、
    上下 `slip`、圆柱 `noSlip`，并输出 `forceCoeffs`。
  两者都按 ESI v2312 的写法：湍流在 `constant/turbulenceProperties`（`simulationType laminar`）、
  物性在 `constant/transportProperties`；`Allrun` 依次执行 `blockMesh`、`checkMesh`（日志含 `Failed` 则失败）、
  `simpleFoam`、`postProcess -func writeCellVolumes -latestTime`。
  `CylinderGeometry`（`x_in`、`x_out`、`y_far`、`half_size`、`res`）参数化域与分辨率，默认值逐字节复现原
  `blockMeshDict`；瞬态模板 `cylinder_transient`（`TransientConfig`、`CylinderTransientProblem`）沿用同一
  网格与边界，按 `t_spin` / `t_end` 生成一段或两段 `Allrun`，不在 `get_problem` 里。
- `fvmrom.rom`：
  - `WeightedPOD`：加权内积下做瘦 SVD，按 `energy_tol` 或 `n_modes` 截断，`project`/`reconstruct` 完成
    系数与场的互转。内积权重、公式与截断规则见 `docs/pod.md`。
  - `regressors`：`RBFRegressor` / `GPRRegressor` / `MLPRegressor` 三项，统一接口
    `fit(Z, A)` / `predict(Z, return_std=False)`，`make_regressor` 按名构造（`REGRESSOR_KINDS`）。
    算法、超参与 `.pkl` 存储格式见 `docs/regressors.md`。
  - `SteadyROM`：U、p 用体积权重，phi 默认单位权重（传入 `phi_weights` 可改用面积权重）；`fit`/`predict`/
    `save`/`load`；`save` 记录依赖版本，`load` 发现版本不一致发 `RuntimeWarning`（pickle 只应加载本地
    可信文件）。
  - `evaluate`：`rel_l2_error`、`projection_error`、`retained_energy`、`loo_cv`（每折重做 POD 与回归）、
    `evaluate_on_test`（含 phi 散度与 U–`phi` 一致性 `flux_consistency`）、`learning_curve`（嵌套 Sobol
    前缀、固定测试集，MLP 单种子 0）、`std_error_correlation`（GPR 标准差与误差的 Spearman 相关）。
  - `dynamics`：`Dynamics` 协议与 `DMD` / `OptDMD` / `BOPDMD` / `OpInf`、`make_dynamics`。DMD 家族解析
    求值，OpInf 是 ODE 模型（四阶差分、归一化岭回归、留出选 λ、重拟合后复检）；算法与结果见
    [docs/transient.md](transient.md)。
  - `transient`：`Readout`（潜变量到 `p`/`phi`/标量的多项式映射）与 `TransientROM`（分场 POD、潜空间
    动力学、读出、`save`/`load`），只用给定的训练窗口拟合。
  - `evaluate_transient`：误差曲线、预测时长、主频、相位、能量漂移、`shedding_summary` 与
    `evaluate_transient` 报告；指标定义见 [docs/transient.md](transient.md)。

### 数据流

```mermaid
flowchart TD
    subgraph GEN["scripts/generate_snapshots.py：生成快照"]
        direction TB
        S1["ParamSpace.sample<br/>Sobol 采样，前 n_train 个训练、后 n_test 个测试"]
        S2["case_templates.write_case<br/>每个样本写一个算例 cases/problem_study/run_xxxx"]
        S3["sweep.run_sweep<br/>并发运行 ./Allrun<br/>blockMesh → checkMesh → simpleFoam → writeCellVolumes"]
        S4["collect_snapshots<br/>检查收敛与拓扑一致，只保留收敛样本"]
        S1 --> S2 --> S3 --> S4
    end

    SNAP[("data/problem/study/<br/>snapshots.npz + samples.json")]

    subgraph FIT["scripts/fit_and_eval.py：拟合与评估"]
        direction TB
        F1["SteadyROM.fit<br/>U、p、phi 分场 WeightedPOD<br/>回归器：RBF / GPR / MLP"]
        F2["evaluate<br/>测试集误差 · 投影下界 · LOO-CV<br/>学习曲线 · phi 散度 · GPR 不确定度"]
        F1 --> F2
    end

    OUT[("data/problem/study/<br/>report.json + models/*.pkl")]
    ROMCASE[("cases/problem_study_rom_regressor/run_xxxx/<br/>时间目录 1/：U p phi U_fom p_fom U_err p_err")]

    S4 --> SNAP --> F1
    F2 --> OUT
    F2 -. "--write-fields N" .-> ROMCASE
```

`--write-fields` 把 ROM 预测写成 `cases/<problem>_<study>_rom_<regressor>/run_xxxx/`（时间目录 `1/` 含
`U p phi U_fom p_fom U_err p_err`）。

### 数据流（阶段 2）

```mermaid
flowchart TD
    subgraph GEN2["scripts/generate_transient.py：运行 FOM 并收集快照"]
        direction TB
        T1["CylinderTransientProblem.write_case<br/>cases/cylinder_transient_study/run_0000"]
        T2["foam.transient.run_transient<br/>Allrun：spin-up 检查点 → 生产段快照<br/>resume 时校验算例字典"]
        T3["collect_transient<br/>选时间目录、并行读 U/p/phi、对齐 Cd/Cl"]
        T1 --> T2 --> T3
    end

    SNAP2[("data/cylinder_transient/study/<br/>snapshots.npz + forces.npz + meta.json")]

    subgraph FIT2["scripts/fit_and_eval_transient.py：训练与评估"]
        direction TB
        G1["TransientROM.fit<br/>分场 POD（只用训练窗口）<br/>U 潜空间动力学 + 读出映射"]
        G2["evaluate_transient<br/>误差 · 预测时长 · 频率/相位<br/>能量 · phi 散度 · 一致性"]
        G1 --> G2
    end

    OUT2[("data/cylinder_transient/study/<br/>report.json + models/method.pkl")]
    ROMCASE2[("cases/cylinder_transient_study_rom_method/run_0000/<br/>U p phi U_fom p_fom U_err p_err")]

    T3 --> SNAP2 --> G1
    G2 --> OUT2
    G2 -. "--write-fields N" .-> ROMCASE2
```

`generate_transient.py` 负责调度瞬态算例并写 `forces.npz`/`meta.json`；`fit_and_eval_transient.py` 读
`snapshots.npz`，训练与评估都在 Windows 侧完成。`--write-fields` 把预测快照写成
`cases/cylinder_transient_<study>_rom_<method>/run_0000/<time>/`，FOM 真值取自 `snapshots.npz`，字段模板
取自 FOM 同时刻的时间目录。

### 质量守恒

phi 的均值与模态都是（离散无散度）快照的线性组合，因此重建 phi 的散度残差也是快照残差的线性组合。
这只保证 phi 在离散意义下无散度，不保证与重建的 U 通量一致。

阶段 2 的 `phi` 用面积权重做 POD，重建仍是均值 + 模态的线性组合，结论不变（实测散度约 $10^{-11}$ ）；
它与预测 `U` 的通量差异由 `flux_consistency` 单独度量，只报告、不设阈值。
