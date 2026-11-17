# 加权 POD

`fvmrom.rom.pod.WeightedPOD` 负责稳态 ROM 的基提取：把快照矩阵 `X`（每行一个快照，形状 `(n, m)`）
降到少数几个模态上。加权内积让"物理上的场能量"（例如单元体积）而不是网格点数决定模态的重要性。

- 代码：`src/fvmrom/rom/pod.py`
- 按场选择权重的规则：`src/fvmrom/rom/steady.py` 的 `_weights_for`
- 在 ROM 中的位置见 `docs/architecture.md`；系数到场的回归见 `docs/regressors.md`

## 加权内积

对长度为 $m$ 的场向量 $\mathbf{a}$, $\mathbf{b}$ 与逐分量权重 $w_i > 0$ ：

$$
\langle \mathbf{a}, \mathbf{b} \rangle_W = \sum_{i=1}^{m} w_i a_i b_i,
\qquad
\lVert \mathbf{a} \rVert_W = \sqrt{\langle \mathbf{a}, \mathbf{a} \rangle_W}
$$

$a_i$, $b_i$ 是展平后场向量的第 $i$ 个分量, $w_i$ 是对应的权重（代码里是 `weights`）。

- `weights=None` 表示单位权重。
- 非 `None` 时，权重必须与展平后的场长度一致、且全部为正的有限值，否则 `ValueError`。
- 单元中心场 `U`、`p` 用单元体积作权重（`steady.py` 的 `VOLUME_WEIGHTED = frozenset({"U", "p"})`）：
  - `p` 的展平长度等于 `n_cells`，权重直接用 `volumes`；
  - `U` 以 `(n_cells, 3)` 存储，展平后长度为 $3 \times$ `n_cells`，权重按 `np.repeat(volumes, 3)` 逐分量重复；
  - 其它长度组合直接抛 `ValueError`。
- 面通量 `phi` 用单位权重：目前只适用于均匀网格，阶段 3 需要改为面积权重。
- 不在 `VOLUME_WEIGHTED` 中的场一律用单位权重。

## 拟合

`WeightedPOD(weights=None, n_modes=None, energy_tol=1e-10)`，调用 `fit(X)`：

```mermaid
flowchart LR
    X["快照矩阵 X<br/>(n, m)"] --> MEAN["mean_ = X.mean(axis=0)<br/>算术平均"]
    X --> CENTRE["去均值<br/>X - mean_"]
    MEAN --> CENTRE
    CENTRE --> SCALE["乘 sqrt(w)<br/>加权坐标"]
    SCALE --> SVD["np.linalg.svd(..., full_matrices=False)"]
    SVD --> RANK["按 energy_tol / n_modes 截断得到 r"]
    RANK --> MODES["modes_ = Vt[:r] / sqrt(w)"]
```

公式（快照矩阵 $\mathbf{X} \in \mathbb{R}^{n \times m}$ ，权重向量 $\mathbf{w}$ ）：

$$
\bar{\mathbf{x}} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}_i, \qquad
\mathbf{X}_c = \mathbf{X} - \mathbf{1}\bar{\mathbf{x}}^{\mathsf{T}}
$$

$$
\left(\mathbf{X}_c \sqrt{\mathbf{W}}\right)_{ij} = (\mathbf{X}_c)_{ij} \sqrt{w_j},
\qquad
\mathbf{X}_c \sqrt{\mathbf{W}} = \mathbf{U} \mathbf{S} \mathbf{V}^{\mathsf{T}} \ (\text{瘦 SVD})
$$

$$
\Phi_{kj} = \frac{\left(\mathbf{V}^{\mathsf{T}}\right)_{kj}}{\sqrt{w_j}} \quad (k = 1, \dots, r),
\qquad
\langle \mathbf{\phi}_k, \mathbf{\phi}_l \rangle_W = \delta_{kl}
$$

- 符号对应： $\bar{\mathbf{x}}$ = `mean_`（算术平均，不是体积加权平均）， $\mathbf{1}$ 是全 1 列向量，
  $\mathbf{X}_c$ = 去均值后的快照， $\mathbf{S}$ = `S`， $\mathbf{V}^{\mathsf{T}}$ = `Vt`， $\Phi$ = `modes_`
  （第 $k$ 行是模态 $\mathbf{\phi}_k$ ，即 $\mathbf{V}^{\mathsf{T}}$ 的前 $r$ 行），
  `singular_values_` = 前 $r$ 个奇异值 $s_1, \dots, s_r$ 。
- `Xc * sqrt(w)` 是逐列乘 $\sqrt{w_j}$ （一维数组按列广播）；除以 $\sqrt{w_j}$ 换回原坐标后，模态在加权内积下
  仍是标准正交的。

### 截断

先在加权坐标里按能量截断，再用数值秩兜底，最后应用 `n_modes` 上限。累计能量与保留的模态数为

$$
E_j = \frac{\sum_{i=1}^{j} s_i^2}{\sum_{i} s_i^2}, \qquad
r_{\text{energy}} = \min\left\{ j : E_j \ge 1 - \varepsilon \right\}
$$

$$
r_{\text{rank}} = \left| \left\{ i : s_i > 10^{-14} s_1 \right\} \right|, \qquad
r = \min\left( r_{\text{energy}},\ r_{\text{rank}},\ n_{\text{modes}} \right)
$$

其中 $\varepsilon$ = `energy_tol`， $n_{\text{modes}}$ = `n_modes`（`None` 时这一项不参与取小）。对应的代码在
`pod.py` 的 `_rank`：`np.cumsum(s**2) / np.sum(s**2)`、`np.searchsorted(energy, 1 - energy_tol, side="left") + 1`、
`np.count_nonzero(s > s[0] * 1e-14)`。`side="left"` 取第一个满足 $E_j \ge 1-\varepsilon$ 的下标（0 起），
加 1 后即保留的模态数。

- `energy_tol` 默认 `1e-10`，允许范围 `[0, 1)`；`n_modes` 必须非负（可以为 0）。
- `S` 为空、或 `S[0] <= 0` 时 `r = 0`，即没有可用模态。

## 投影与重建

`project(X)` 给出系数 $\mathbf{A}$ （形状 $(n, r)$ ），`reconstruct(A)` 换回场：

$$
A_{ik} = \sum_{j=1}^{m} w_j \left(X_{ij} - \bar{x}_j\right) \Phi_{kj},
\qquad
\mathbf{A} = (\mathbf{X} - \mathbf{1}\bar{\mathbf{x}}^{\mathsf{T}})\, \mathbf{W}\, \mathbf{\Phi}^{\mathsf{T}}
$$

$$
\hat{X}_{ij} = \sum_{k=1}^{r} A_{ik} \Phi_{kj} + \bar{x}_j,
\qquad
\hat{\mathbf{X}} = \mathbf{A} \mathbf{\Phi} + \mathbf{1}\bar{\mathbf{x}}^{\mathsf{T}}
$$

其中 $\mathbf{W} = \operatorname{diag}(w_1, \dots, w_m)$ ； $\mathbf{A}$ 、 $\hat{\mathbf{X}}$ 就是 `project` /
`reconstruct` 的返回值。代码写作 `(X - mean_) @ (w[:, None] * modes_.T)` 与 `A @ modes_ + mean_`。

- `project` 要求 `X` 的列数等于 `mean_.size`；`reconstruct` 要求 `A` 的形状为 `(n, r)`。
- 未拟合（`modes_ is None`）时调用 `project`/`reconstruct` 会抛 `RuntimeError`。

## 属性与 pickle 兼容

| 名称 | 说明 |
| --- | --- |
| `weights` | 只读属性，返回内积权重（`None` 表示单位权重）；内部字段是 `_weights` |
| `n_modes` | 模态数上限，`None` 表示只按能量截断 |
| `energy_tol` | 能量截断阈值 |
| `mean_` | 快照的算术平均 |
| `modes_` | 形状 `(r, m)` 的模态 |
| `singular_values_` | 截断后的奇异值 $s_1, \dots, s_r$ （加权坐标下） |

`weights` 曾经是普通可写字段。旧的 pickle 把权重存成状态里的 `weights` 键，`__setstate__` 会把它改名为
`_weights` 再写入 `__dict__`，因此旧模型仍能加载；新代码不应再直接给 `weights` 赋值。
