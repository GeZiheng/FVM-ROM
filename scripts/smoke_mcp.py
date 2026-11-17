"""Talk to the MCP server over stdio like a real client: cavity case, mesh -> solve -> inspect.

Usage: uv run python scripts/smoke_mcp.py
"""

import asyncio
import json
import shutil
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from fvmrom.core.config import get_settings

CASE = "_smoke_cavity"
SERVER = Path(__file__).resolve().parents[1] / ".opencode" / "mcps" / "openfoam" / "server.py"


async def call(session, name, **args):
    r = await session.call_tool(name, args)
    text = "".join(c.text for c in r.content if c.type == "text")
    if r.isError:
        raise RuntimeError(f"{name}: {text}")
    return json.loads(text) if text[:1] in "{[" else text


async def main():
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    shutil.rmtree(get_settings().cases_dir / CASE, ignore_errors=True)
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        tools = [t.name for t in (await s.list_tools()).tools]
        print("tools:", len(tools))
        print("env:", await call(s, "foam_env_info"))
        await call(s, "copy_tutorial", tutorial="incompressible/icoFoam/cavity/cavity", case_name=CASE)
        print("endTime:", await call(s, "foam_dict_get", case=CASE, file="system/controlDict", entry="endTime"))
        print("blockMesh rc:", (await call(s, "run_foam_app", case=CASE, app="blockMesh"))["returncode"])
        print("checkMesh rc:", (await call(s, "run_foam_app", case=CASE, app="checkMesh"))["returncode"])
        job = await call(s, "start_foam_job", case=CASE, app="icoFoam")
        while (st := await call(s, "job_status", job_id=job["job_id"]))["state"] in ("starting", "running"):
            await asyncio.sleep(1)
        print("job:", st["state"], st["exit_code"], st["elapsed_s"], "s")
        print("residuals:", json.dumps((await call(s, "parse_solver_log", case=CASE, log="log.icoFoam"))["fields"]["p"]))
        print("times:", (await call(s, "case_summary", case=CASE))["time_dirs"])
        try:
            await call(s, "read_case_file", case=CASE, path="../../pyproject.toml")
        except RuntimeError as e:
            print("path escape rejected:", str(e)[:80])


asyncio.run(main())
