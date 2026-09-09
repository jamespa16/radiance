# experiments/

Working records for A/Bs whose *method* is worth keeping, not just their verdict — the config
generator, the driver, the raw logs, and the dead ends.

The headline numbers live in [docs/results.md](../docs/results.md), which stays the single place to
look up "what did we measure". A directory appears here when the result came with reusable
machinery or with methodological findings that outlived the question being asked.

| directory | question | verdict |
|---|---|---|
| [loop-vs-depth/](loop-vs-depth/) | does `blocks[1:]`'s weight-shared loop beat plain depth, at equal parameters and at equal FLOPs? | **No** — dominated by real depth and by MoE. Also produced three methodology findings that apply to every A/B on this page. |
| [under-50m/](under-50m/) | how should a hard **50M total parameter** budget be spent on TinyStories? | The tokenizer first (gpt2's 50k vocab wastes a third of the budget), then the loop and differential attention — both of which docs/results.md rejects on cost grounds a parameter cap does not impose. **MoE inverts**: the page's largest win becomes its largest loss. |

These are records, not maintained code: the scripts pin the repo as it stood when they ran and are
not covered by the test suite. Read them for method, re-derive before re-running.
