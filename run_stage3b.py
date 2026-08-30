"""Stage 3B: does SMOTE beat the Stage 3 class-weighting strategy?

Run from the project root:

    python run_stage3b.py

A supplementary experiment. It changes nothing about the project's official
results: the Stage 4 winner stays the final model, no artifact from Stages 1-5
is rewritten, and the test split is never opened.

Sections:
  1. training data
  2. the two strategies, and the single parameter that separates them
  3. audit -- where SMOTE actually runs
  4. cross-validation, both arms
  CV RESULTS / COMPARISON TABLE
  5. reproduction check against Stage 3
  6. figure
  SMOTE EXPERIMENT FINDINGS
  7. integrity checks
  CURRENT OFFICIAL FINAL MODEL
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# File-access audit hook
# --------------------------------------------------------------------------
# Installed before any other import so it observes every file this process
# opens, the training-data read included. CPython raises the "open" audit event
# for reads and writes alike, so a path absent from this set was neither read
# nor written. `check_file_access()` uses that to test the test-split claim
# against something the interpreter observed rather than against a promise.
#
# Scope, stated plainly: this watches the parent process. `cross_validate`
# fans folds out to worker processes, which do not inherit the hook -- but they
# receive `X` and `y` already in memory and execute only `src/`, which the AST
# scan covers. The training-data read is the positive control: if the hook were
# somehow inert, `train.csv` would be missing from the set and the check fails
# loudly rather than passing by seeing nothing.
PROJECT_ROOT = Path(__file__).resolve().parent
_PROJECT_PREFIX = os.path.normcase(str(PROJECT_ROOT))
_OPENED: set[str] = set()


def _record_open(event: str, args) -> None:
    if event != "open":
        return
    try:
        target = args[0]
        if not isinstance(target, (str, bytes, os.PathLike)):
            return  # already-open file descriptor, not a path
        resolved = os.path.normcase(os.path.abspath(os.fsdecode(target)))
    except Exception:  # noqa: BLE001 - an audit hook must never raise
        return
    if resolved.startswith(_PROJECT_PREFIX) and ".venv" not in resolved:
        _OPENED.add(resolved)


sys.addaudithook(_record_open)

import textwrap  # noqa: E402  (must follow the audit hook)
import time  # noqa: E402

import pandas as pd  # noqa: E402

from src import config, evaluate, train  # noqa: E402
from src import smote_experiment as sx  # noqa: E402


def rule(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def ok(passed: bool) -> str:
    return "PASS" if passed else "*** FAIL ***"


def opened(path: Path) -> bool:
    return os.path.normcase(str(Path(path).resolve())) in _OPENED


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
_FULL_WIDTHS = {"Accuracy": 8, "Precision": 9, "Recall": 6, "F1": 6,
                "ROC-AUC": 7, "PR-AUC": 6}


def print_full_results(results: pd.DataFrame) -> None:
    """Every model and strategy against all six metrics."""
    header = ("Model".ljust(20) + "Strategy".ljust(14)
              + "".join(f" | {m:>{w}}" for m, w in _FULL_WIDTHS.items()))
    print("  " + header)
    print("  " + "-" * len(header))
    previous = None
    for _, row in results.iterrows():
        if previous is not None and row["Model"] != previous:
            print()
        cells = "".join(f" | {row[m]:>{w}.4f}" for m, w in _FULL_WIDTHS.items())
        print("  " + row["Model"].ljust(20) + row["Strategy"].ljust(14) + cells)
        previous = row["Model"]


def print_comparison_table(results: pd.DataFrame, best_index) -> None:
    """The required table: Model | Strategy | PR-AUC | Recall | F1 | ROC-AUC."""
    widths = {"PR-AUC": 7, "Recall": 7, "F1": 7, "ROC-AUC": 7}
    header = ("Model".ljust(20) + "Strategy".ljust(14)
              + "".join(f" | {m:>{w}}" for m, w in widths.items()))
    print("  " + header)
    print("  " + "-" * len(header))
    for index, row in results.iterrows():
        cells = "".join(f" | {row[m]:>{w}.4f}" for m, w in widths.items())
        marker = "   <-- highest PR-AUC" if index == best_index else ""
        print("  " + row["Model"].ljust(20) + row["Strategy"].ljust(14)
              + cells + marker)


def print_per_model_findings(table: pd.DataFrame) -> None:
    """What SMOTE did to each model's PR-AUC, Recall, and F1."""
    metrics = [sx.PRIMARY_METRIC] + sx.SUPPORTING_METRICS
    short = {"unchanged within fold noise": "within fold noise"}
    header = ("Model".ljust(20)
              + "".join(f"{m + ' (SMOTE - CW)':<29}" for m in metrics))
    print("  " + header)
    print("  " + "-" * len(header))
    for entry in sx.per_model_findings(table):
        cells = ""
        for metric in metrics:
            item = entry[metric]
            verdict = short.get(item["verdict"], item["verdict"])
            cells += f"{item['delta']:+.4f}  {verdict:<20}"
        print("  " + entry["Model"].ljust(20) + cells)
    print()
    print("  \"within fold noise\" means the change is smaller than the pooled")
    print("  fold-to-fold standard deviation of the two arms it came from")
    print(f"  ({sx.MEANINGFUL_SD_MULTIPLE:.0f} sd). That is a descriptive yardstick, "
          "not a significance test.")


# --------------------------------------------------------------------------
# Integrity
# --------------------------------------------------------------------------
def check_file_access(failures: list[str]) -> None:
    """The test split was never opened; the training split was."""
    train_seen = opened(config.TRAIN_CSV)
    test_seen = opened(config.TEST_CSV)
    # Bytecode caches are still recorded and still checked -- they are left out
    # of the count only because Python writes them on a cold cache and not on a
    # warm one, which would make this line differ between two identical runs.
    real = [path for path in _OPENED if "__pycache__" not in path]

    print("  a) runtime file-access audit  (sys.addaudithook on the \"open\" event)")
    print(f"       project files opened, bytecode caches aside : {len(real)}")
    print(f"       data/processed/train.csv opened      : {train_seen}   "
          f"{ok(train_seen)}   <-- positive control")
    print(f"       data/processed/test.csv opened       : {test_seen}   "
          f"{ok(not test_seen)}")
    print("       The control matters: a hook that observed nothing would report")
    print("       the test split as untouched for the wrong reason. Seeing the")
    print("       training read proves the hook was live when it mattered.")
    print("       CPython raises this event for reads and writes alike, so a path")
    print("       absent from the set was neither read nor written.")
    if not train_seen:
        failures.append(
            "the audit hook never saw train.csv, so its test.csv result "
            "cannot be trusted"
        )
    if test_seen:
        failures.append("data/processed/test.csv was opened during Stage 3B")


def check_source_scans(failures: list[str]) -> None:
    """AST scans: the test split is never read, no tuning API is called."""
    print("\n  b) the test split is never read  (every use of the path, read from")
    print("     the AST, so a mention in a comment or docstring cannot count)")
    print("       The integrity checks below have to name the file in order to")
    print("       prove it was left alone, so the claim under test is not that the")
    print("       name never appears -- it is that the path is never used to read")
    print("       the data. Any operation not on the non-reading list counts as a")
    print("       read, so an unrecognised use fails rather than slipping through.")
    print(f"       non-reading operations: "
          f"{', '.join(sorted(sx.NON_READING_USES))}")
    print()
    for rel, records in sx.test_split_uses().items():
        reads = [r for r in records if r["reading"]]
        print(f"       {rel}   {len(records)} use(s), {len(reads)} reading   "
              f"{ok(not reads)}")
        for record in records:
            label = "READS DATA" if record["reading"] else "no read"
            print(f"         line {record['line']:<5} "
                  f"{record['use']:<16} {label}")
        if reads:
            failures.append(
                f"{rel} reads the test split: "
                + ", ".join(f"line {r['line']} {r['use']}" for r in reads)
            )

    print("\n  c) no search or tuning API is called")
    for rel, hits in sx.check_no_tuning().items():
        print(f"       {rel:<26} {hits or 'none'}   {ok(not hits)}")
        if hits:
            failures.append(f"{rel} calls a tuning API: {hits}")
    print(f"       watched names: {', '.join(sorted(sx.TUNING_APIS))}")


def check_one_knob(failures: list[str], y) -> None:
    """Exactly one hyperparameter separates each model's two arms."""
    print("\n  d) only the imbalance knob differs between the two arms")
    models = train.build_models(train.compute_scale_pos_weight(y))
    for key, spec in models.items():
        changed = sx.imbalance_change(key, spec)
        single = len(changed) == 1
        print(f"       {spec['name']:<20} {', '.join(changed) or 'nothing':<28} "
              f"{ok(single)}")
        if not single:
            failures.append(
                f"{key}: {len(changed)} parameters differ between arms "
                f"({changed}), so the comparison is not controlled"
            )
    print("       Every other hyperparameter is carried across from Stage 3 by")
    print("       rebuilding from its own get_params(), not by being retyped.")


def check_protected_files(
    paths, before: dict[str, str], test_stat_before: dict, failures: list[str]
) -> None:
    """Stage 1-5 outputs are byte-identical; the test split is untouched."""
    print("\n  e) every Stage 1-5 output is byte-identical after this stage")
    after = sx.file_checksums(paths)
    changed = [name for name in before if before.get(name) != after.get(name)]
    groups = {
        "model pickles": [f"{k}.pkl" for k in evaluate.MODELS],
        "stage 3-4 artifacts": ["model_cv_results.csv", "model_metadata.json",
                                "test_results.csv"],
        "stage 2/4/5 figures": [n for n in before if n.endswith(".png")],
        "training split": ["train.csv"],
    }
    for label, names in groups.items():
        present = [n for n in names if n in before]
        clean = not [n for n in present if n in changed]
        print(f"       {label:<22} {len(present):>2} files   {ok(clean)}")
    print(f"       total watched          {len(before):>2} files   "
          f"changed: {changed or 'none'}   {ok(not changed)}")
    print("       Stage 3B's own two outputs are excluded on purpose -- this")
    print("       stage rewrites them, and watching them would make the check")
    print("       fail for the wrong reason.")
    if changed:
        failures.append(f"Stage 1-5 outputs changed during Stage 3B: {changed}")

    test_stat_after = sx.test_csv_stat()
    unchanged = test_stat_before == test_stat_after
    print("\n  f) the test split, checked without opening it  (stat only)")
    print(f"       size    {test_stat_before['size']:,} bytes -> "
          f"{test_stat_after['size']:,} bytes")
    print(f"       mtime   unchanged: {unchanged}   {ok(unchanged)}")
    if not unchanged:
        failures.append("data/processed/test.csv was modified during Stage 3B")


def check_reproduction(results: pd.DataFrame, failures: list[str]) -> None:
    """The class-weight arm must land on Stage 3's published numbers."""
    deltas = sx.reproduction_deltas(results)
    tolerance = 1e-9
    print("  Stage 3B recomputes the class-weight arm rather than reading Stage")
    print("  3's numbers, so both arms share one set of folds and one scorer. That")
    print("  is only trustworthy if the recomputation reproduces Stage 3 exactly.")
    print()
    print(f"  {'Model':<20}{'max |delta| vs Stage 3':>24}")
    print("  " + "-" * 44)
    for _, row in deltas.iterrows():
        print(f"  {row['Model']:<20}{row['max_abs_delta']:>24.2e}")
    worst = float(deltas["max_abs_delta"].max())
    passed = bool(deltas["found_in_stage_3"].all()) and worst < tolerance
    print()
    print(f"  worst deviation across 5 models x 6 metrics : {worst:.2e}")
    print(f"  within {tolerance:.0e}                                : {ok(passed)}")
    print("  A non-zero delta would mean a hyperparameter, a fold, or a metric")
    print("  definition moved -- and the SMOTE comparison would be measuring that")
    print("  instead of measuring the strategy.")
    if not passed:
        failures.append(
            f"the class-weight arm deviates from Stage 3 by {worst:.2e}, "
            "so Stage 3's hyperparameters were not reproduced exactly"
        )


# --------------------------------------------------------------------------
def main() -> int:
    started = time.perf_counter()
    failures: list[str] = []
    config.ensure_dirs()

    watched = sx.protected_files()
    checksums_before = sx.file_checksums(watched)
    test_stat_before = sx.test_csv_stat()

    rule("SMOTE EXPERIMENT  (Stage 3B -- supplementary, training data only)")
    print("  Question: does SMOTE beat the Stage 3 class-weighting strategy?")
    print()
    print("  The test split was already spent on the Stage 4 final evaluation.")
    print("  Using it again to choose between two imbalance strategies would turn")
    print("  a held-out score into a selection score, so this experiment runs on")
    print("  the training split only and the test split is never opened.")

    # ---------------------------------------------------------------- 1
    rule("1. TRAINING DATA")
    train_df = train.load_train()
    X = train_df[config.FEATURE_COLS]
    y = train_df[config.TARGET]
    spw = train.compute_scale_pos_weight(y)
    print(f"  file                     : "
          f"{config.TRAIN_CSV.relative_to(config.PROJECT_ROOT)}")
    print(f"  rows                     : {len(train_df):,}")
    print(f"  churn positives          : {int(y.sum()):,} / {len(y):,} "
          f"= {y.mean():.4f}")
    print(f"  scale_pos_weight (Stage 3): {spw:.6f}  "
          "(negatives/positives, training labels only)")

    # ---------------------------------------------------------------- 2
    rule("2. THE TWO STRATEGIES")
    print(f"  {'Model':<20}{'Class Weight arm':<34}SMOTE arm")
    print("  " + "-" * 92)
    models = train.build_models(spw)
    for key, spec in models.items():
        changed = sx.imbalance_change(key, spec)
        if key == "xgboost":
            smote_side = (f"scale_pos_weight={sx.NEUTRAL_SCALE_POS_WEIGHT:.1f} "
                          "(neutral) + SMOTE")
        else:
            smote_side = "class_weight=None + SMOTE"
        print(f"  {spec['name']:<20}{spec['imbalance']:<34}{smote_side}")
    print()
    print("  Both arms use the Stage 3 estimators unchanged apart from that one")
    print("  knob. The SMOTE arm is built from each Stage 3 object's own")
    print("  get_params(), with a single key replaced, so nothing else can drift.")
    print("  Applying scale_pos_weight and SMOTE together would correct the")
    print("  imbalance twice, which is why XGBoost's knob goes neutral.")
    print()
    print("  Pipeline order, class weight : prep -> classifier")
    print("  Pipeline order, SMOTE        : prep -> SMOTE -> classifier")
    print("  imbalanced-learn's Pipeline runs samplers during fit() only and")
    print("  skips them for predict(), so the validation fold keeps its natural")
    print("  class balance. SMOTE sits after prep because it interpolates")
    print("  numerically and cannot read raw category strings.")

    # ---------------------------------------------------------------- 3
    rule("3. WHERE SMOTE ACTUALLY RUNS  (observed, not asserted)")
    audit = sx.audit_smote_inside_cv(X, y)
    print(f"  A real 5-fold cross_validate on {audit['model']}, single-process, "
          "with the")
    print("  sampler instrumented to record every resample it performs.")
    print()
    print(f"  {'fold':<6}{'rows in':>10}{'rows out':>10}"
          f"{'negatives':>12}{'positives':>11}")
    print("  " + "-" * 49)
    for index, record in enumerate(audit["log"], start=1):
        print(f"  {index:<6}{record['rows_in']:>10,}{record['rows_out']:>10,}"
              f"{record['negatives_out']:>12,}{record['positives_out']:>11,}")
    print()
    checks = [
        ("sampler fired at all", audit["recorded_anything"],
         f"{audit['resamples']} resamples recorded"),
        ("one resample per training fold", audit["one_per_training_fold"],
         f"{audit['resamples']} == n_splits {audit['expected_resamples']}"),
        ("input rows total 4 x the training set", audit["rows_add_up"],
         f"{audit['rows_in_total']:,} == "
         f"{train.N_SPLITS - 1} x {audit['training_rows']:,} = "
         f"{audit['rows_in_expected']:,}"),
        ("never saw the whole training set", audit["never_saw_full_training_set"],
         f"largest fold {audit['largest_fold_seen']:,} < "
         f"{audit['training_rows']:,}"),
        ("every fold grew", audit["grew_every_fold"], "oversampling took effect"),
        ("every output exactly balanced", audit["balanced_output"],
         "negatives == positives in all folds"),
    ]
    for label, passed, detail in checks:
        print(f"  {label:<40} {detail:<38} {ok(passed)}")
        if not passed:
            failures.append(f"SMOTE placement audit failed: {label}")
    print()
    print("  The row arithmetic is the proof. Each row sits in the training part")
    print("  of 4 folds out of 5, so the sampler must see 4 x 5,634 rows in total.")
    print("  SMOTE applied before the split, or to the validation folds as well,")
    print("  or to anything involving the test set, could not produce that total.")

    # ---------------------------------------------------------------- 4
    rule(f"4. CROSS-VALIDATION  (StratifiedKFold n_splits={train.N_SPLITS}, "
         f"shuffle=True, random_state={config.SEED})")
    results = sx.run_experiment(X, y)

    rule("CV RESULTS  (5-fold, training split only)")
    print_full_results(results)

    table = sx.comparison_table(results)
    best_index = results[sx.PRIMARY_METRIC].idxmax()

    rule("COMPARISON TABLE")
    print_comparison_table(results, best_index)

    # ---------------------------------------------------------------- 5
    rule("5. REPRODUCTION CHECK  (class-weight arm vs Stage 3's stored results)")
    check_reproduction(results, failures)

    csv_path = sx.save_results(results)
    print(f"\n  written: {csv_path.relative_to(config.PROJECT_ROOT)}  "
          f"({len(results)} rows)")

    # ---------------------------------------------------------------- 6
    rule("6. FIGURE")
    figure = sx.plot_comparison(table, len(X))
    print(f"  {figure.relative_to(config.PROJECT_ROOT)}  "
          f"({figure.stat().st_size:,} bytes)")
    print("  form                     : one dumbbell per model, six panels, one")
    print("                             figure -- the question is movement")
    print("                             between two states of the same model")
    print("  axis                     : all six panels share one x-range, so a")
    print("                             small change reads as small instead of")
    print("                             being magnified by a per-panel zoom")
    print("  colour                   : two categorical hues, validated in light")
    print("                             mode on the Stage 4 surface; the tritan")
    print("                             margin sits in the 6-8 floor band, so")
    print("                             marker shape carries identity too")
    if not figure.exists():
        failures.append("smote_comparison.png was not written")

    # ---------------------------------------------------------------- findings
    rule("SMOTE EXPERIMENT FINDINGS")
    print_per_model_findings(table)
    print()
    for number, finding in enumerate(sx.findings(results, table), start=1):
        print(textwrap.indent(
            textwrap.fill(f"{number}. {finding}", width=88,
                          subsequent_indent=" " * (len(str(number)) + 2)),
            "  ",
        ))
        print()

    # ---------------------------------------------------------------- 7
    rule("7. INTEGRITY CHECKS")
    check_file_access(failures)
    check_source_scans(failures)
    check_one_knob(failures, y)
    check_protected_files(watched, checksums_before, test_stat_before, failures)

    # ---------------------------------------------------------------- verdict
    verdict = sx.official_model_statement(results, table)
    rule("CURRENT OFFICIAL FINAL MODEL")
    print("  Random Forest with Stage 3 class weighting.")
    print("  artifacts/random_forest.pkl, selected in Stage 4 on test PR-AUC")
    print("  0.6517, explained in Stage 5. Unchanged by this experiment.")
    print()
    print(f"  Its cross-validated PR-AUC in this experiment : "
          f"{verdict['official_pr_auc']:.4f}")
    print(f"  Best PR-AUC anywhere in this experiment       : "
          f"{verdict['best_pr_auc']:.4f}  "
          f"({verdict['best_model']}, {verdict['best_strategy']})")
    print(f"  Margin                                        : "
          f"{verdict['margin']:+.4f}")
    print(f"  SMOTE's effect on the Random Forest           : "
          f"{verdict['forest_delta']:+.4f} PR-AUC against a pooled fold sd of "
          f"{verdict['forest_pooled_sd']:.4f}")
    print()
    if verdict["consider_replacement"]:
        print("  ENOUGH EVIDENCE TO CONSIDER A CHANGE: yes. A SMOTE arm clears the")
        print("  official model's cross-validated PR-AUC by more than the folds'")
        print("  own spread. This is a recommendation to review, not a change:")
        print("  confirming it on the test set would spend a split already used")
        print("  in Stage 4, so the decision is a human one.")
    else:
        print("  ENOUGH EVIDENCE TO CONSIDER A CHANGE: no. No SMOTE arm clears the")
        print("  official model's cross-validated PR-AUC by more than the folds'")
        print("  own spread, so the experiment gives no reason to revisit the")
        print("  Stage 4 selection. The Random Forest with class weighting stays")
        print("  the final model and no Stage 4 or Stage 5 output was regenerated.")
    print()
    print("  Nothing here is a causal claim. These are cross-validated scores on")
    print("  training folds: they describe how each strategy changed the models'")
    print("  measured performance, not what causes a customer to churn.")

    # ---------------------------------------------------------------- done
    rule("STAGE 3B RESULT")
    print(f"  runtime : {time.perf_counter() - started:.1f}s")
    if failures:
        print(f"  FAILED -- {len(failures)} problem(s):")
        for problem in failures:
            print(f"    - {problem}")
        return 1
    print("  All checks passed. Stage 3B complete.")
    print("  Supplementary only: Stages 1-5 stand as they were.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
