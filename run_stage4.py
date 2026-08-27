"""Stage 4: final evaluation of the Stage 3 models on the held-out test set.

Run from the project root:

    python run_stage4.py

This is the first stage that reads `data/processed/test.csv`. It loads the five
saved pipelines exactly as Stage 3 wrote them, scores each one once, and reports
metrics, curves, confusion matrices, a CV-versus-test comparison, and findings
drawn only from the numbers actually produced.

Nothing here fits, tunes, or re-selects a model. The final section proves that:
checksums of the five `.pkl` files are compared before and after, the Stage 4
source is scanned for fit-family calls, and each loaded estimator's parameters
are checked against what Stage 3 recorded.
"""

from __future__ import annotations

import textwrap
import time

from src import config, evaluate


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def ok(passed: bool) -> str:
    return "PASS" if passed else "*** FAIL ***"


def print_results_table(results, best_key: str) -> None:
    """Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC."""
    header = (f"  {'Model':<21}" + "".join(f"{m:>11}" for m in evaluate.METRICS))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in results.iterrows():
        marker = "  <-- best PR-AUC" if row["key"] == best_key else ""
        print(f"  {row['Model']:<21}"
              + "".join(f"{row[m]:>11.4f}" for m in evaluate.METRICS)
              + marker)


def print_comparison_table(comparison) -> None:
    """Model | CV PR-AUC | Test PR-AUC, ordered by the CV ranking."""
    ordered = comparison.sort_values("CV PR-AUC", ascending=False)
    print(f"  {'Model':<21}{'CV PR-AUC':>12}{'Test PR-AUC':>14}"
          f"{'Difference':>13}{'CV rank':>10}{'Test rank':>11}")
    print("  " + "-" * 79)
    for _, row in ordered.iterrows():
        print(f"  {row['Model']:<21}{row['CV PR-AUC']:>12.4f}"
              f"{row['Test PR-AUC']:>14.4f}{row['Difference']:>+13.4f}"
              f"{row['CV rank']:>10}{row['Test rank']:>11}")


def main() -> int:
    started = time.perf_counter()
    failures: list[str] = []

    # ---------------------------------------------------------------- 1
    rule("1. LOAD SAVED STAGE 3 PIPELINES  (loaded as-is, never refitted)")
    checksums_before = evaluate.model_checksums()
    models = evaluate.load_models()
    for key, name in evaluate.MODELS.items():
        path = evaluate.model_path(key)
        n_features = models[key].named_steps["prep"].n_features_in_
        print(f"  {name:<21} {path.name:<25} {path.stat().st_size:>9,} bytes  "
              f"features in: {n_features}")
    print(f"  models loaded            : {len(models)} / 5  "
          f"{ok(len(models) == 5)}")
    print("  sha256 recorded before evaluation for all 5 files")
    if len(models) != 5:
        failures.append("not all five models loaded")

    # ---------------------------------------------------------------- 2
    rule("2. LOAD HELD-OUT TEST SET  (first use of test.csv in this project)")
    X_test, y_test = evaluate.load_test()
    print(f"  file                     : "
          f"{config.TEST_CSV.relative_to(config.PROJECT_ROOT)}")
    print(f"  rows x features          : {len(X_test)} x {X_test.shape[1]}")
    print(f"  churners / retained      : {int(y_test.sum())} / "
          f"{int((1 - y_test).sum())}")
    print(f"  churn rate               : {y_test.mean():.4f}")
    print(f"  majority-class accuracy  : {1 - y_test.mean():.4f}"
          "   <-- the bar accuracy must clear")
    print("  manual preprocessing     : none -- each model is a full Pipeline")
    features_match = list(X_test.columns) == config.FEATURE_COLS
    print(f"  columns match config     : {ok(features_match)}")
    if not features_match:
        failures.append("test-set columns do not match config.FEATURE_COLS")

    # ---------------------------------------------------------------- 3
    rule("3. SCORE THE TEST SET  (one pass per model)")
    predictions, call_log = evaluate.evaluate_models(models, X_test)
    for key in call_log:
        print(f"  {evaluate.MODELS[key]:<21} predict + predict_proba on "
              f"{len(X_test)} rows")
    results = evaluate.results_frame(predictions, y_test)
    csv_path = evaluate.save_results(results)

    rule("MODEL TEST-SET RESULTS  (held-out data, never seen in training)")
    best_key, best_name, best_value = evaluate.best_by(
        results, evaluate.SELECTION_METRIC
    )
    print_results_table(results, best_key)
    print(f"\n  saved: {csv_path.relative_to(config.PROJECT_ROOT)}")

    # ---------------------------------------------------------------- selection
    rule("FINAL MODEL SELECTION")
    print(f"  Best model by PR-AUC: {best_name}")
    print(f"  PR-AUC: {best_value:.4f}")
    print("\n  PR-AUC was fixed as the selection metric in Stage 3, before the")
    print("  test set was opened, because the positive class is only "
          f"{y_test.mean():.1%}.")

    print("\n  Ranking under the other metrics:")
    agreement = {}
    for metric in ("PR-AUC", "F1", "ROC-AUC", "Recall"):
        order = evaluate.ranking(results, metric)
        agreement[metric] = order[0][0]
        tag = "  (selection metric)" if metric == evaluate.SELECTION_METRIC else ""
        print(f"    {metric:<8}: "
              + " > ".join(f"{name} {value:.4f}" for name, value in order[:3])
              + " > ..." + tag)

    distinct = sorted(set(agreement.values()))
    if len(distinct) == 1:
        print(f"\n  All four metrics rank {distinct[0]} first, so the choice is")
        print("  not sensitive to which of them is used.")
    else:
        print(f"\n  The metrics do not agree on a single winner: "
              f"{len(distinct)} different")
        print(f"  models lead ({', '.join(distinct)}). "
              f"{best_name} is reported as the")
        print("  selection under PR-AUC, not as the best model on every measure.")

    # ---------------------------------------------------------------- 4
    rule("4. CROSS-VALIDATION (Stage 3) vs TEST SET (Stage 4)")
    comparison = evaluate.cv_vs_test(results)
    print_comparison_table(comparison)
    ranks_identical = bool((comparison["CV rank"] == comparison["Test rank"]).all())
    print(f"\n  CV artifact modified     : no  (read-only: "
          f"{evaluate.CV_RESULTS_CSV.name})")
    print(f"  PR-AUC ranking identical : {ranks_identical}")
    print(f"  largest |difference|     : "
          f"{comparison['Difference'].abs().max():.4f}")

    # ---------------------------------------------------------------- 5
    rule("5. FIGURES")
    figure_paths = evaluate.make_all_figures(predictions, y_test, results)
    for path in figure_paths:
        print(f"  {path.relative_to(config.PROJECT_ROOT)}  "
              f"({path.stat().st_size:,} bytes)")
    print(f"  figures written          : {len(figure_paths)}  "
          f"{ok(len(figure_paths) == 8)}")
    print("  colour                   : CVD-validated 5-hue set; curves also")
    print("                             carry per-model dash patterns, so the")
    print("                             series are separable without colour")
    if len(figure_paths) != 8:
        failures.append(f"expected 8 figures, wrote {len(figure_paths)}")

    # ---------------------------------------------------------------- findings
    rule("EVALUATION FINDINGS  (drawn from the test results above)")
    for i, finding in enumerate(
        evaluate.findings(results, comparison, y_test, predictions), start=1
    ):
        print(textwrap.indent(
            textwrap.fill(f"{i}. {finding}", width=68, subsequent_indent="   "),
            "  ",
        ))
        print()

    # ---------------------------------------------------------------- 6
    rule("6. INTEGRITY CHECKS  (no retraining, no retuning)")
    integrity_checks(models, checksums_before, failures)

    # ---------------------------------------------------------------- done
    rule("STAGE 4 RESULT")
    print(f"  runtime : {time.perf_counter() - started:.1f}s")
    if failures:
        print(f"  FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  All checks passed. Stage 4 complete.")
    return 0


def integrity_checks(models, checksums_before: dict[str, str],
                     failures: list[str]) -> None:
    """Print the four checks that make "nothing was retrained" measurable."""
    print("  a) saved model files are byte-identical after evaluation")
    checksums_after = evaluate.model_checksums()
    for key, name in evaluate.MODELS.items():
        unchanged = checksums_before.get(key) == checksums_after.get(key)
        print(f"       {name:<21} sha256 {checksums_before[key][:16]}...  "
              f"{ok(unchanged)}")
        if not unchanged:
            failures.append(f"{key}.pkl changed during evaluation")

    print("\n  b) no fit-family call exists in Stage 4 source")
    print("       (read from each file's AST, so a mention in a comment or")
    print("        docstring cannot satisfy the check -- only real calls count)")
    for rel, hits in evaluate.check_no_fit_calls().items():
        print(f"       {rel:<21} {sorted(evaluate.FIT_METHODS)}")
        print(f"       {'':<21} call sites found: {hits or 'none'}   "
              f"{ok(not hits)}")
        if hits:
            failures.append(f"{rel} contains fit-family calls: {hits}")

    print("\n  c) the test set is scored in exactly one place")
    call_sites = evaluate.count_prediction_call_sites()
    single = all(count == 1 for count in call_sites.values())
    for method, count in call_sites.items():
        print(f"       src/evaluate.py  .{method}() call sites: {count}   "
              f"{ok(count == 1)}")
    print("       every metric, curve and matrix is derived from that one pass")
    if not single:
        failures.append(f"unexpected prediction call sites: {call_sites}")

    print("\n  d) loaded hyperparameters match what Stage 3 recorded")
    for key, differences in evaluate.check_hyperparameters_unchanged(models).items():
        print(f"       {evaluate.MODELS[key]:<21} "
              f"differences vs model_metadata.json: "
              f"{differences or 'none'}   {ok(not differences)}")
        if differences:
            failures.append(f"{key} hyperparameters differ from Stage 3")


if __name__ == "__main__":
    raise SystemExit(main())
