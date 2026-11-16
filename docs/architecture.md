# 架构

```
Windows                                         WSL2 (Ubuntu)
-------                                         -------------
OpenCode 智能体 --stdio--> fvmrom-mcp
                              |  fvmrom.foam.Wsl
ROM 代码 (numpy/scipy/...) ---+--> 脚本文件 --wsl.exe --exec bash--> source OpenFOAM bashrc
        |                                                             blockMesh / 求解器 / mpirun
        +--- 读写 cases/ 和 data/  <======= /mnt/d/Projects/FVM-ROM/... =======+
```

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

## ROM 开发的扩展点

- `fvmrom.foam` 可直接在 ROM 脚本中导入：运行算例、启动任务、读取摘要。
- 场数据 I/O（OpenFOAM ASCII/二进制 -> numpy 快照矩阵）应放在未来的 `fvmrom/foam/fields.py`；
  用于内积的网格/体积权重同理。
- ROM 专用的 MCP 工具（快照生成、基计算、ROM 求解与 FOM 误差对比）待技术路线确定后，可加入
  `mcp_server/server.py`（或另建模块）。
