"""Print val/loss per eval step for every arm log in a directory."""

import pathlib
import re
import sys

logs = sorted(pathlib.Path(sys.argv[1]).glob("logs/*.log"))
rows = {}
for log in logs:
    evals = {int(s): float(v) for s, v in re.findall(r"step\s+(\d+) val/loss ([\d.]+)", log.read_text())}
    rows[log.stem] = evals
steps = sorted({s for e in rows.values() for s in e})
print(f"{'arm':22s}" + "".join(f"{s:>8d}" for s in steps))
for name, e in rows.items():
    print(f"{name:22s}" + "".join(f"{e[s]:8.4f}" if s in e else f"{'-':>8s}" for s in steps))
