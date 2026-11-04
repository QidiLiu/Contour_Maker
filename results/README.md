# results/

Curated, reproducible benchmark summaries. These files **are** committed (unlike
`reports/`, which holds per-image CSVs and visualisations).

| File | Description |
|---|---|
| `summary.csv` | one row per (scheme, runtime) with detection/segmentation/speed |
| `summary.md` | the same table, rendered for reading |

Generate them after running the Python validators and the C++ binaries:

```bash
python scripts/collect_results.py --reports reports --out results
```

`reports/<dataset>_<scheme>_{py,cxx}/summary.json` is the input. Each summary
follows the same schema (see `common/eval.py` and
`cxx-common/src/eval_harness.cpp`).
