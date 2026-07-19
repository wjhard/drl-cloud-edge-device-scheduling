# Autonomous Exploration Evidence

This directory preserves every initial screen, independent paired repeat, terminal log, and statistical summary from the bounded three-direction exploration started on 2026-07-19.

## Layout

- `direction1_local_search/`: Residual Best-of-64 versus precedence-feasible single-task relocation.
- `direction2_lns/`: direction 1 versus randomized destroy-and-repair large-neighborhood search.
- `direction3_gumbel_beam/`: Best-of-64 + LNS versus Gumbel-diversified policy beam search + LNS.
- `compute_matched_sampling/`: timing calibration and five paired Best-of-128 versus Best-of-64 + local-search + LNS runs.

Each directory contains `initial_screen.json`, five `repeat_N.json` files when the initial result looked better, matching `.log` files, and `paired_repeats_summary.json`. Seeds were sampled from system entropy and were not selected after observing results.

## Result

Direction 2 is the best statistically supported refinement in this scan (`mean_ratio=0.920889698460 +/- 0.001729491028`, five repeats, 20/20 scenarios better than HEFT). Its direct paired difference from the original Residual Best-of-64 is `-0.029924552672 +/- 0.003080309578` (`p=0.000026568512`). Against the slightly more expensive pure Residual Best-of-128, its paired difference is `-0.021157810659 +/- 0.001148009015` (`p=0.000002072081`), so the gain is not explained by extra wall-clock budget alone. Direction 3 did not show a significant improvement (`p=0.904282284548`). See `docs/自主探索日志.md` for the chronological decision record and literature provenance.
