# 路线图

## 已完成
- [x] 项目骨架（uv、src 布局、测试）
- [x] WSL2 <-> OpenFOAM 访问层
- [x] OpenFOAM MCP 服务（18 个工具）：环境、教程、算例、运行（前台/后台、MPI）、日志
- [x] MCP 服务脚本从 Python 包（`fvmrom.mcp_openfoam` + `openfoam-mcp` console script）迁到
  `.opencode/mcps/openfoam/server.py`：`src/` 只保留 ROM 与 OpenFOAM 访问层，服务脚本归 harness 接入层
- [x] ROM 技术路线讨论与确认（见下）
- [x] 阶段 1（M1–M6）：参数空间与采样；ASCII 场读写与网格拓扑；批量采样调度器（并发、resume、停止/清理
  残留任务）；cavity 与 cylinder 算例生成；分场加权 POD；RBF/GPR/MLP 回归器；评估（测试集、投影下界、
  LOO-CV、学习曲线、`phi` 散度、GPR 不确定度相关性）；ROM 写回 OpenFOAM 时间目录（含
  `U_fom/p_fom/U_err/p_err`，ParaView 可对比）。
- [x] 阶段 2（M0–M4）：网格几何与 `phi` 面积权重；U–`phi` 通量一致性指标；圆柱几何参数化与 Re=100 的
  域/网格 pilot；两段式瞬态快照生成（含续算与收集）；瞬态 ROM 框架与评估（训练/预测切分、读出映射）；
  DMD 家族；Operator Inference（含重拟合后复检与 λ 敏感性扫描）。当前共 380 个测试通过
  （`-m "not wsl"` 为 370，跳过 10 个 WSL 集成测试）。
- [x] 模块重组（纯重构，行为不变）：`fvmrom.config` / `fvmrom.params` → `fvmrom.core.config` /
  `fvmrom.core.params`，`fvmrom.problems` → `fvmrom.case_templates`；旧 `.pkl` 通过兼容 unpickler 仍可加载。

## ROM 技术路线（2026-11-16 已确认）

总体分三个阶段，依次推进。每个阶段的详细计划放在 `.opencode/plans/`。

### 总体原则
- **先非侵入式，后侵入式。** 非侵入式 ROM 只读写场文件，与 OpenFOAM ESI v2312 兼容性最好。侵入式
  （POD-Galerkin、DEIM 等）需要取得 FV 算子，留到后续再做。
- **批量采样、场读取、ROM 全部写成 Python 库**（`fvmrom.foam`、`fvmrom.rom` 等），不走 MCP。
  OpenFOAM MCP 保持为交互调试用的接口；ROM 专用的 MCP 工具等代码稳定后再考虑。
- **物理一致性**：把面通量 `phi` 作为独立场做 POD，重建的 `phi` 保持离散连续性（质量守恒）。
- **网络回归器可替换**：回归器统一接口，GPR / RBF 为默认，MLP 作为对比项，其代码可在阶段 3 复用。
- 层流起步；湍流量（nut、k、epsilon）暂不处理。

### 阶段 1：稳态参数化场重建（已完成，M1–M6）
- 网格固定，在物理参数空间采样（Sobol / LHS），用 OpenFOAM 生成稳态快照，对任意参数预测全场。
- 算例：顶盖驱动方腔（cavity）和圆柱绕流（cylinder），层流，**只用 Re 一个参数**。
  圆柱稳态只在 $\mathrm{Re} \lesssim 47$ 成立，参数范围取 10–40。
- 方法：体积加权 POD（分场）+ 回归（RBF、GPR；MLP 作对比项），算法见 `docs/pod.md` 与
  `docs/regressors.md`。
- 评估：测试集误差、POD 投影误差、留一交叉验证、`phi` 散度残差。
- 已验证结论：
  - 测试集 `U` 平均相对 L2 误差：cavity RBF 1.2e-3 / GPR 4.7e-5 / MLP 1.4e-3；cylinder RBF 2.2e-4 /
    GPR 2.3e-5 / MLP 9.2e-5。
  - 重建 `phi` 的最大单元散度约 1e-11，与 FOM 同量级（离散连续性得以保持）。
  - 30 样本下 MLP 比 GPR 差约 4–29 倍；学习曲线显示 $n \le 8$ 时 MLP/RBF 优于 GPR， $n \ge 12$ 后 GPR 领先。
- 之后扩展：边界条件参数化、多参数（留待后续）。
- 计划：`.opencode/plans/20261116-stage1-steady-rom.md`

### 阶段 2：瞬态时间外推（已完成，M0–M4）
- 网格与 $\mathrm{Re}$ 固定（二维圆柱绕流， $\mathrm{Re} = 100$ 的卡门涡街），用前 3 个周期（90 个快照）
  训练，之后不再调用 OpenFOAM，由潜空间动力学把解推进下去。
- 方法（非侵入式）：分场 POD + 只在 `U` 潜空间（默认 8 个模态）上的动力学，`p`、`phi` 与 Cd/Cl 由潜变量
  经多项式读出映射预测。基线为**标准 DMD、Optimized DMD、BOP-DMD**，主方法为 **Operator Inference**
  （二次形式，契合不可压 NS）。HODMD、LSTM 不做；latent 神经 ODE 作为有条件的 M5。算法与指标见
  [docs/transient.md](transient.md)。
- 已完成：
  - **M0 前置**：网格几何读取与 `phi` 面积权重（`foam.mesh`）；U–`phi` 通量一致性指标
    （`flux_consistency`，阶段 1 的评估也接入）；圆柱几何参数化（域大小、分辨率）； $\mathrm{Re} = 100$ 的
    域 / 网格 pilot（base/wide/fine）。
  - **M1 快照**：`pimpleFoam` 两段式运行（spin-up 检查点 + 生产段每 0.2 个时间单位写快照）、`--resume`
    续算与算例字典一致性校验；`re100` 共 501 个快照（ $t = 60\text{–}160$ ）。
  - **M2 框架与评估**：训练 / 预测按时间切分（基、读出、动力学、λ 只用训练窗口拟合）；`TransientROM` 与
    读出映射；误差、预测时长、频率、相位、能量、`phi` 散度、U–`phi` 一致性等指标；`--write-fields` 写回
    ROM 算例。
  - **M3 DMD 家族**：Optimized DMD（变量投影）、BOP-DMD（bagging，特征值一对一匹配与相位对齐后取均值）。
  - **M4 OpInf**：四阶差分 + 列归一化岭回归、前向留出选 λ、**重拟合后复检**、λ 敏感性只读扫描。
- 验收结果（默认配置：`U` 8 个模态、训练 3 个周期；数值见 `docs/transient.md`）：Optimized DMD 与
  BOP-DMD 全部 PASS；标准 DMD 仅 Cl 幅值 FAIL（6.45%，主要来自二次读出放大潜变量偏差）；OpInf 不发散，
  仅 U 误差 FAIL（ $\tau \le 5$ 内最大 1.44e-2）；未复检的 `opinf_norecheck` 在 $\tau \approx 4.3$ 发散。
- 已知限制与待决问题：
  - 验收结论只限定于默认配置：`U` = 10、12 时 Optimized/BOP-DMD 的 Cl 误差约 9–19%；训练 4 个周期时为
    1.97%–1.99%，贴着 2% 的限值。
  - OpInf 的改进（离散形式、约束、更长留出段、多步打靶）留给后续计划。
  - 标量读出阶次与岭参数的选择准则在 M6 计划里定（目前默认阶 2，并附阶 1 诊断列）。
  - FOM 的 Cl 幅值有约 5% 的网格敏感性，且相对文献偏高；ROM 对比同一 FOM 的快照，不受影响。
- 后续验证算例（已写入计划，各自细化；不属于本轮 M0–M4 的验收）。里程碑编号只表示计划里的名称，不表示
  先后顺序；**当前拟定的顺序是 M6 →（视 M6 结果决定 OpInf 改进与 M5）→ M7 → M8**：
  - **M6（下一步，另开计划）**：圆柱 $\mathrm{Re} = 150\text{–}180$ ，复用现有模板；2D 假设上限取 180
    （3D 失稳约在 190 前后），需重做 pilot 与验收阈值。排第一是因为成本最低，且能回答"结论对
    $\mathrm{Re}$ 是否稳健"，同时作为标量读出阶次 / 岭参数选择准则的独立检验算例。**开跑前必须在其计划里
    写明：读出阶次 / 岭参数的选择准则、验收阈值（不能直接沿用 $\mathrm{Re}=100$ 的）、OpInf 是原样上还是
    先做改进。**
  - **M5（latent 神经 ODE，MLP + 多步打靶）**：**有条件**的备选，本轮未做，排在 M6 之后、是否做看 M6 的
    结果。触发条件是 DMD 家族与 OpInf 在验收上都不够好；当前 $\mathrm{Re}=100$ 的默认配置下 Optimized DMD
    与 BOP-DMD 已全部 PASS，条件尚未成立。$\mathrm{Re}=100$ 近乎纯周期，DMD 已贴近 POD 投影下界
    （2.4e-3 对 1.96e-3），再比较神经网络的信息量很小；训练只有 90 个快照，神经方法易过拟合。若 DMD 家族
    在 $\mathrm{Re}=150\text{–}180$ 仍通过，M5 的必要性更低；若谐波增强后 DMD 开始失败，才有明确证据，且
    M6 的结果可作为对照基准。若目标改为展示神经网络方法本身，M5 可与 M6 同时做、在两个 $\mathrm{Re}$ 上
    比较，而不是只在 $\mathrm{Re}=100$ 上做。
  - **M7**：方腔非定常启动，衰减型瞬态；指标改为到达稳态的时间尺度（衰减率）与全场 L2 曲线随时间，
    不用频率 / 相位。排在 M6 之后。
  - **M8**：后向台阶（ $\mathrm{Re}$ 约 500–800），必须先 pilot 确认有可预测的周期 / 准周期行为；备选
    双圆柱串列。排在 M6、M7 之后，是否做视二者结果再定。
- 计划：`.opencode/plans/20261117-stage2-transient-rom.md`

### 阶段 3：几何泛化（待讨论）
- 对同一类问题（圆柱，之后可能是汽车外流场），从几何参数空间采样，预测稳态解或推进瞬态解。
- 核心难点：不同几何的网格不同，快照没有逐点对应关系。
- 路线：
  - **稳态，先做网格变形到同拓扑**（参数化 blockMesh 或 RBF/FFD 变形基准网格，在 Python 端改写
    `polyMesh/points`），再套用阶段 1 的方法；从 2D 圆柱起步。`phi` 的面积权重已在阶段 2 提供
    （阶段 1 用 `--phi-weights area` 可选），非均匀网格不再需要单独改权重。
  - **变形过大或拓扑变化时用神经方法**：GNN（MeshGraphNets）、神经算子（Geo-FNO、GINO）、SDF 条件
    隐式表示等；已确认有 GPU 可用。
  - 瞬态几何泛化难度最高（涡脱落频率和相位随参数变化，线性基插值失效），放到最后；候选方向是参数化
    Operator Inference 或条件潜空间 ODE。
  - 汽车外流场的单次 RANS 代价很高，样本量不足时只能考虑稳态；暂不作为起步目标。

## 待确定的问题
- 阶段 2 的后续：顺序为 M6（下一步，另开计划）→ 视 M6 结果决定 OpInf 改进与 M5 → M7 → M8；各自的验收
  阈值与算例细节在各自计划里定。做新算例的理由：现有结论只在单个算例的默认配置上成立（扫描显示模态数、
  训练周期稍变，排序就变）； $\mathrm{Re}=100$ 对 DMD 过于友好，OpInf 的价值未被真正检验；读出阶次准则与
  OpInf 改进方向需要独立算例验证，以免在 $\mathrm{Re}=100$ 上过拟合。
- 阶段 3：几何参数化的具体方式；3D 汽车外流场是否纳入。
- 侵入式 ROM 何时引入，以及采用什么形式（超降阶、稳定化、supremizer 等）。
- 自适应采样（贪心加点，基于 GPR 方差）是否纳入阶段 1 之后的迭代。
- ROM 专用 MCP 工具：阶段 1 不做（脚本 `generate_snapshots.py`、`fit_and_eval.py` 已覆盖流程，采样是长任务
  不适合工具调用，且接口在阶段 2、3 会变）。阶段 2/3 视使用情况再评估；若要做，建议另起独立的 MCP 服务：
  `.opencode/mcps/rom/server.py`，服务名 `rom`——命名规则是目录 `.opencode/mcps/<功能>/` 与服务名 `<功能>`
  一一对应，不再有 `<功能>-mcp` 命令和 `fvmrom.mcp_<功能>` 包；独立服务还能避免拖慢 `openfoam` 服务、
  工具列表也不至于过长。只做薄封装，第一批工具可以是 `list_studies`、`study_report`、`rom_predict`；
  采样仍走脚本和后台作业。

## 阶段 2 的前置事项（已完成）
- [x] U–`phi` 通量一致性指标：`fvmrom.rom.evaluate.flux_consistency`，阶段 1 与阶段 2 的评估都已接入；
  只报告、不设阈值。
- [x] `phi` 的面积权重：`fvmrom.foam.mesh.phi_weights`；阶段 1 用 `--phi-weights area` 可选（默认仍为单位
  权重），阶段 2 的瞬态 ROM 默认使用。
- [x] 圆柱域与网格： $\mathrm{Re} = 100$ 的 pilot（base/wide/fine）对比 St、平均 Cd、Cl 幅值与极限环时间；
  fine 的 Cl 幅值相对 base 为 −5.37%，超过 5% 阈值，经确认沿用 base，并把约 5% 的网格敏感性记为已知限制。

## 技术路线确定后的下一步（阶段 1、阶段 2 已全部完成）
1. [x] 阶段 1：场读取器（OpenFOAM 场 -> numpy，带单元体积权重）—— `fvmrom.foam.fields`。
2. [x] 阶段 1：通过 `fvmrom.foam` 驱动的稳态参数扫描与快照生成 —— `scripts/generate_snapshots.py`。
3. [x] 阶段 1：在 `fvmrom/rom/` 中实现基计算与 ROM 构建 —— `scripts/fit_and_eval.py`。
4. [x] 阶段 1：在线预测并写回 OpenFOAM 时间目录，输出 FOM 与 ROM 的误差报告 ——
   `cases/<problem>_<study>_rom_<regressor>/`。
5. [x] 阶段 2：网格几何与 `phi` 面积权重、U–`phi` 通量一致性 —— `fvmrom.foam.mesh`、
   `fvmrom.rom.evaluate.flux_consistency`。
6. [x] 阶段 2：两段式瞬态运行、续算与快照收集 —— `scripts/generate_transient.py`。
7. [x] 阶段 2：瞬态 ROM（分场 POD + 潜空间动力学 + 读出映射）与评估 ——
   `scripts/fit_and_eval_transient.py`、`scripts/opinf_lambda_scan.py`。
8. [ ] 剩余（按当前拟定顺序）：M6（圆柱 Re=150–180，下一步，另开计划）→ 视 M6 结果决定 OpInf 改进与
   M5（latent 神经 ODE，有条件）→ M7（方腔非定常启动）→ M8（后向台阶，视 M6、M7 结果再定），以及阶段 3
   （几何泛化）—— 各自待确认后另行细化计划，见上文相应小节。
