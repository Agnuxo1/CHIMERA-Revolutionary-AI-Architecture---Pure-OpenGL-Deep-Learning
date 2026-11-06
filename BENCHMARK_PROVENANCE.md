# Benchmark and publication provenance

The current README records that several displayed benchmark and memory values are constants. `demos/chimera_benchmark_comparison.py` uses fixed `base_time` values, and `demos/chimera_benchmark_results.json` stores the resulting figures. `demos/DEMO_OFFICIAL_RESULTS.md` explicitly describes the results as simulations and says no real OpenGL compute shaders or physical-GPU performance were tested. Those files remain unchanged.

No runtime suite or GPU benchmark was run during this attribution update. The baseline contains 20 Python files, nine of which already fail Python 3.13.7 parsing; this matches the limitation documented in the README. The README also reports the limited CPU/software-OpenGL checks performed on 1 November 2026. No new performance result is claimed here.

The demo README refers to an ARC Prize 2025 entry, but this repository contains no Kaggle submission ID. The official [ARC Prize 2025 archive](https://arcprize.org/competitions/2025/archive) states that an award paper must correspond to an eligible Kaggle submission. Entry identity and eligibility remain unverified. The paper artifacts are preserved; their HTML says the manuscript was submitted to Nature Machine Intelligence / IEEE TNNLS, but acceptance, publication, and DOI were not verified.
