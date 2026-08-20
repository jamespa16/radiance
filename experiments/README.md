# experiments/

Working records for A/Bs whose *method* is worth keeping, not just their verdict — the config
generator, the driver, the raw logs, and the dead ends.

The headline numbers live in [docs/results.md](../docs/results.md), which stays the single place to
look up "what did we measure". A directory appears here when the result came with reusable
machinery or with methodological findings that outlived the question being asked.

| directory | question | verdict |
|---|---|---|
| [loop-vs-depth/](loop-vs-depth/) | does `blocks[1:]`'s weight-shared loop beat plain depth, at equal parameters and at equal FLOPs? | **No** — dominated by real depth and by MoE. Also produced three methodology findings that apply to every A/B on this page. |

These are records, not maintained code: the scripts pin the repo as it stood when they ran and are
not covered by the test suite. Read them for method, re-derive before re-running.
