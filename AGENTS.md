# Notes for agents working on HQP (Weighted-Hierarchical-QP)

`hqp::HierarchicalQP` solves hierarchical quadratic programs: levels of linear constraints `l ≤ A x ≤ u`, each
minimized in priority order over the optima of the levels above, then the point closest to a reference in a metric.
It is a header-only C++20 library on Eigen, with a pybind11 module (`pyhqp`) and Doxygen docs published to
`gh-pages`. This file is the handoff from the sessions that last worked on it (November 2026).

**Its successor is [hlq](https://github.com/gianluca-garofalo/hlq).** It solves the same problem with one
priority-ordered LQ factorization of all levels, updated by rotations, with no dependencies. It is about ten times
as fast on 20–40 variables (69 µs against about 780 µs for HierarchicalQP and 670 µs for LexLS) and at least as
accurate. New algorithmic work belongs there. This repository stays as a reference: hlq's comparisons build
HierarchicalQP at a pinned commit (`cmake/hlqCompetitors.cmake` in hlq, currently `384670c`, this `main`). After a
change here, bump that pin in hlq.

## Map

| Path | What |
|---|---|
| `include/hqp/hqp.hpp` | The class: members, sizes (fixed or dynamic through template parameters), declarations |
| `include/hqp/solvers.tpp` | `solve`, `equality_hqp` (per-level complete orthogonal decompositions on nested null spaces), `inequality_hqp` (the active-set search), `ratio_test`, `activate_violated`, `dual_update` (multipliers), `increment_primal`, `increment_from`/`decrement_from` |
| `include/hqp/constraints.tpp` | `activate_constraint`, `deactivate_constraint`, `lock_constraint`, `swap_constraints` |
| `include/hqp/accessors.tpp` | `set_problem`, `set_metric`, `get_primal`, `get_slack`, `get_parent` |
| `include/hqp/constructors.tpp`, `utilities.tpp` | Constructors; `print_active_set` |
| `tests/` | Numbered tests: `03_set_stack`, `07_template_demo`, `08_warm_start`, and one per fix: `09_equality_lock`, `10_ratio_test`, `11_level_multipliers`, `12_rank_threshold` |
| `python/` | `pyhqp` (pybind11), `setup.py`, `test_pyhqp.py` |
| `doc/` | Doxygen configuration; `ci-docs.yml` publishes to `gh-pages` |
| `cmake/HQP_config.cmake.in`, `CMakePresets.json` | The installed package (`find_package(HQP)`, target `HQP::HQP`); presets `release`, `debug`, `python-release`, `python-debug` |

## Build and test

```bash
cmake --preset=release && cmake --build build --preset release && ctest --preset release   # 7 tests
cmake --preset=python-release && cmake --build build                                      # pyhqp
```

- Needs Eigen 3.4 and CMake 3.22 or later.
- CI (`.github/workflows`):
  - `ci-cpp`: gcc and clang, Debug and Release; coverage on gcc Debug; a `downstream` job that builds hqp-tasks and
    hqp-compare against this checkout.
  - `ci-python`; `ci-docs`.
  - Every job has a timeout. Eigen comes from a cached `.deb` (`.github/actions/install-eigen`), so CI no longer
    depends on Ubuntu's package mirror.

**Checking a change against other solvers:** hlq's `benchmarks/compare.cpp` (`hlq_compare`) compares hlq, LexLS,
HierarchicalQP and DAQP level by level on 2,000 problems per family. To include this checkout, build hlq's
comparisons with `-DHLQ_BUILD_COMPARISONS=ON -DFETCHCONTENT_SOURCE_DIR_HLQ_HQP=/path/to/this/repo` and run
`build/benchmarks/hlq_compare 2000`. hqp-compare's tests check HierarchicalQP against LexLS and DAQP on single
hierarchies.

## History

- **#9: the split.** The stack of tasks (`task/task.hpp`, `pyhqp.Task`/`StackOfTasks`) moved to
  [hqp-tasks](https://github.com/gianluca-garofalo/hqp-tasks), and the comparisons with DAQP and LexLS to
  [hqp-compare](https://github.com/gianluca-garofalo/hqp-compare). `HQPConfig.cmake` is installed, so
  `find_package(HQP)` works. The `downstream` CI job was added.
- **#10: four solver bugs**, each with a test that fails without its fix:
  - `set_problem` read `equalitySet_` after a swap, so some equalities were never locked.
  - `inequality_hqp` had no ratio test: the iterate jumped to each active set's solution, broke solved levels and
    cycled, and a stale change budget cut the cycles short.
  - `dual_update` read another level's columns of `inverse_`, so the multipliers were wrong.
  - Ranks used Eigen's relative COD threshold, so rounding noise got rank; the threshold is now absolute.

  Checked against LexLS on 100,000 problems per family: before the fixes, main was worse than LexLS on 472, 40, 664
  and 3,154 of 20,000 per family; after them, never. Warm re-solves need no changes, and cold solves are about a
  third faster.
- **#11: CI hardening:** job timeouts, and Eigen without the package mirror.
- **#1–#5** were automated PRs, closed. Two pieces of them were taken into #9: the mask cleanup of #1 and the clang
  CI fix of #4.

## Known limits

- **Speed:** each change recomputes the per-level decompositions from the changed level down, which is what makes it
  about ten times slower than hlq on 20–40 variables.
- **Ill-conditioned problems:** in a 20,000-problem scan after #10, the search hit its 500-change safety cap on 57
  problems with an ill-conditioned level and 33 degenerate ones. In hlq's comparison, hlq is lexicographically
  better on 8 of 2,000 random problems with an ill-conditioned level, and HierarchicalQP better on 1 degenerate one.

## Open

- **Issue #8:** the split removed public API, but the versions disagree: `0.1.1` in `CMakeLists.txt` and
  `setup.py`, against a `v1.0.0` tag. Set 2.0.0, tag `v2.0.0`, and pin the companions' `HQP_GIT_TAG` to it.
- **README:** a few stale lines: a clone URL `your-repo/HQP`, and a duplicated `breaks = np.array(...)` line between
  sections.
- **Stale branches**, safe to delete:
  - `bolt-*`, `jules-*`, `fix-ci-cpp-workflow-*` and `refactor-split-repos-*` (the closed automated PRs);
  - `claude/ci-hardening` (merged as #11);
  - `claude/friendly-cannon-fhetcc`: #10's commits, plus a prototype of the LQ solver (`afb9fc1`, `67f0c20`) that
    became hlq.

  Keep `gh-pages`.

## Conventions

- **Commits and PR titles:** `type> description.`, lowercase, imperative, with a final period, for example
  `fix> decide ranks with an absolute threshold.` Types: `feat`, `fix`, `chore`, `docs`, `test`, `ci`.
- **Every fix comes with a numbered test** that fails without it. Compare against LexLS, and against hlq where it
  applies, level by level, before claiming correctness.
- **No AI attribution:** no `Co-Authored-By`/session trailers, and no model names, in commits, code or PRs.
- **Branches and merging:** work on a branch and open a PR; never merge without the owner's explicit go-ahead. Single
  PRs have been merged by rebase.
- **PR descriptions:** the bug and its cause, the fix, the numbers, the tests. Keep the prose short.
- **Companions:** a change to the public API or the CMake package breaks hqp-tasks and hqp-compare, which build
  against `main` by default. The `downstream` CI job shows it.
