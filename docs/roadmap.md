# Roadmap

## Done
- [x] Project skeleton (uv, src layout, tests)
- [x] WSL2 <-> OpenFOAM access layer
- [x] OpenFOAM MCP server (17 tools): env, tutorials, cases, run (fg/bg, MPI), logs

## To be decided together (open questions)
- Target problem: which physics / geometry / parameters? (e.g. lid-driven cavity, cylinder wake,
  parametrised Re / inlet velocity / geometry)
- Intrusive vs non-intrusive ROM:
  - Non-intrusive: POD (+ RBF / GP / NN) regression of coefficients - no OpenFOAM source needed.
  - Intrusive: POD-Galerkin / PG on FV operators - needs matrix/operator access (OpenFOAM C++
    utility or re-assembly in Python); the apt install ships `openfoam2312-dev` so custom
    applications can be compiled.
- Stability/closure for convection-dominated flow (supremizer, stabilisation, eddy-viscosity).
- Parametric sampling strategy (LHS / greedy) and the error metric against the full-order model.

## Next steps once the technique is chosen
1. Field reader (OpenFOAM fields -> numpy) with cell-volume weights.
2. Snapshot generation workflow driven through the MCP tools (parameter sweep as jobs).
3. Basis computation + ROM construction in `fvmrom/rom/`.
4. Online solve, reconstruction to an OpenFOAM time directory (for ParaView), FOM-vs-ROM error report.
