# Architecture

```
Windows                                         WSL2 (Ubuntu)
-------                                         -------------
OpenCode agent --stdio--> fvmrom-mcp
                              |  fvmrom.foam.Wsl
ROM code (numpy/scipy/...) ---+--> script file --wsl.exe --exec bash--> source OpenFOAM bashrc
        |                                                                 blockMesh / solver / mpirun
        +--- reads/writes cases/ and data/  <======= /mnt/d/Projects/FVM-ROM/... =======+
```

## Decisions

- **MCP server on the Windows side.** The numerical ROM stack stays in one Python environment
  (uv); WSL is used purely as an OpenFOAM execution backend.
- **Script files, not command lines.** Each call writes an LF script under `.fvmrom/scripts/` and
  runs `wsl.exe --exec bash <script>`. Avoids PowerShell/cmd/bash quoting problems.
- **Only OpenFOAM executables.** `type -P` must resolve into `$WM_PROJECT_DIR`,
  `$FOAM_USER_APPBIN` or `$FOAM_SITE_APPBIN` (or a script inside the case, `./Allrun`).
  File access is confined to `cases/<case>/` (`safe_join`).
- **Background jobs are file-based** (`.fvmrom/jobs/<id>/{meta.json,pid,exit_code}`) and run in
  their own session (`setsid`), so status and `stop_job` work after an MCP restart. The `wsl.exe`
  handle is kept open while the job runs so WSL does not treat the distro as idle.
- **Cases on `/mnt/d`.** Simple and visible from both sides. File I/O through the 9P bridge is
  slower than ext4; if snapshot writing becomes a bottleneck, point `FVMROM_CASES_DIR` at a
  Windows-visible `\\wsl.localhost\...` path or move heavy runs to the WSL filesystem.

## Extension points for the ROM work

- `fvmrom.foam` is importable from ROM scripts: run cases, start jobs, read summaries.
- Field I/O (OpenFOAM ASCII/binary -> numpy snapshot matrices) belongs in a future
  `fvmrom/foam/fields.py`; mesh/volume weights for the inner product likewise.
- ROM-specific MCP tools (snapshot generation, basis computation, ROM solve vs. FOM error) can be
  added to `mcp_server/server.py` (or a second module) once the technique is chosen.
