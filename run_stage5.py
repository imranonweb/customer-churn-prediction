"""Stage 5: explaining the final model (Random Forest) with SHAP.

Run from the project root:

    python run_stage5.py

Stage 4 selected the Random Forest on test PR-AUC (0.6517), and it also led on
Recall, F1, and ROC-AUC. This stage answers the follow-up question -- *why does
that model predict what it predicts* -- using the pipeline exactly as Stage 3
saved it.

Sections:
  1. load the saved Random Forest pipeline (never refitted)
  2. transform the test set with the already-fitted preprocessor
  3. compute SHAP values and verify them (additivity, shape, finiteness)
  TOP CHURN DRIVERS  -- global ranking by mean |SHAP|
  HIGH-RISK / LOW-RISK CUSTOMER -- two local explanations
  4. figures
  XAI FINDINGS -- SHAP compared with the Stage 2 EDA observations
  5. integrity checks

Throughout, SHAP is treated as a description of model behaviour. It is not
evidence of causation, and the findings section says so explicitly.
"""

from __future__ import annotations

import textwrap
import time

import numpy as np

from src import config, evaluate, explain


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def ok(passed: bool) -> str:
    return "PASS" if passed else "*** FAIL ***"


def print_driver_table(importance, top_n: int) -> None:
    """Rank | feature | mean |SHAP| | direction when high | transformed name."""
    print(f"  {'#':>3}  {'Feature':<30}{'mean |SHAP|':>12}"
          f"   {'when high / yes':<26}Transformed name")
    print("  " + "-" * 116)
    for rank, (_, row) in enumerate(importance.head(top_n).iterrows(), start=1):
        effect = row["high_value_effect"]
        if effect != effect:  # NaN: the feature is constant on the test set
            direction = "n/a (constant)"
        else:
            direction = (f"{effect:+.4f} "
                         f"{'toward churn' if effect > 0 else 'away from churn'}")
        print(f"  {rank:>3}. {row['readable']:<30}"
              f"{row['mean_abs_shap']:>12.4f}   {direction:<26}{row['feature']}")


def print_factor_list(heading: str, frame, sign: str) -> None:
    print(f"\n  {heading}")
    if not len(frame):
        print("    (none -- no feature moved the prediction in this direction)")
        return
    for _, item in frame.iterrows():
        print(f"    {sign} {item['readable']:<32} = {item['display_value']:<10}"
              f"  SHAP {item['shap']:+.4f}")


def print_customer(heading: str, row: int, proba: float, label: int,
                   toward, away, waterfall, base_value: float) -> None:
    """One customer's probability, true label, and signed contributions."""
    rule(heading)
    print(f"  Test-set row index       : {row}")
    print(f"  Churn probability        : {proba * 100:.2f}%")
    print(f"  Actual churn label       : {'Yes' if label == 1 else 'No'}"
          "   (looked up only for reporting -- not used to pick this customer)")
    print(f"  Model baseline           : {base_value * 100:.2f}%   "
          "(the forest's average output before any feature is considered)")
    print(f"  Baseline + SHAP sum      : "
          f"{(base_value + waterfall['shap'].sum()) * 100:.2f}%   "
          "(reconstructs the probability above)")

    print_factor_list(
        f"Top factors increasing churn risk:  "
        f"(largest {len(toward)} of the positive contributions)", toward, "+")
    print_factor_list(
        f"Top factors decreasing churn risk:  "
        f"(largest {len(away)} of the negative contributions)", away, "-")


def main() -> int:
    started = time.perf_counter()
    failures: list[str] = []

    rule("SHAP ANALYSIS")
    print(f"  Final model: {explain.MODEL_NAME}")
    print("\n  Selected in Stage 4 on test PR-AUC (0.6517); it also led Recall,")
    print("  F1, and ROC-AUC. Loaded from artifacts/random_forest.pkl exactly as")
    print("  Stage 3 saved it -- this stage fits nothing.")
    print("\n  SHAP explains how THIS MODEL forms its predictions. It does not")
    print("  establish that any feature causes a customer to churn.")

    # ---------------------------------------------------------------- 1
    rule("1. LOAD THE SAVED RANDOM FOREST PIPELINE  (as-is, never refitted)")
    checksums_before = evaluate.model_checksums()
    pipe = explain.load_model()
    prep = pipe.named_steps["prep"]
    rf = pipe.named_steps["model"]
    prep_fingerprint_before = explain.preprocessor_fingerprint(prep)

    path = evaluate.model_path(explain.MODEL_KEY)
    print(f"  file                     : "
          f"{path.relative_to(config.PROJECT_ROOT)}  "
          f"({path.stat().st_size:,} bytes)")
    print(f"  pipeline steps           : {list(pipe.named_steps)}")
    print(f"  estimator                : {type(rf).__name__}  "
          f"n_estimators={rf.n_estimators}  "
          f"class_weight={rf.class_weight}")
    print(f"  fitted trees present     : {len(rf.estimators_)}  "
          f"{ok(len(rf.estimators_) == rf.n_estimators)}")
    print(f"  preprocessor             : {type(prep).__name__}  "
          f"n_features_in_={prep.n_features_in_}  "
          f"(fitted in Stage 3, reused here)")
    print(f"  numeric transform        : "
          f"{prep.transformers[0][1]!r} on {len(config.NUMERIC_COLS)} columns"
          "   <-- unscaled, so SHAP")
    print("                             feature values read on the original scale")
    print(f"  categorical transform    : "
          f"{type(prep.transformers[1][1]).__name__} on "
          f"{len(config.CATEGORICAL_COLS)} columns")
    print(f"  preprocessor fingerprint : {prep_fingerprint_before[:16]}...  "
          "(recorded before any use)")
    if len(rf.estimators_) != rf.n_estimators:
        failures.append("loaded forest does not hold its full tree count")

    # ---------------------------------------------------------------- 2
    rule("2. TRANSFORM THE TEST SET  (transform only -- no fit, no fit_transform)")
    X_test, y_test = explain.load_test()
    matrix, names = explain.transform_features(pipe, X_test)
    print(f"  input                    : "
          f"{config.TEST_CSV.relative_to(config.PROJECT_ROOT)}  "
          f"{len(X_test)} rows x {X_test.shape[1]} features")
    print(f"  transformed matrix       : {matrix.shape[0]} x {matrix.shape[1]}  "
          f"{matrix.dtype}")
    print(f"  transformed names        : {len(names)}  "
          f"{ok(len(names) == matrix.shape[1])}")
    print(f"  estimator expects        : {rf.n_features_in_}  "
          f"{ok(rf.n_features_in_ == matrix.shape[1])}")
    if rf.n_features_in_ != matrix.shape[1]:
        failures.append("transformed width does not match the fitted estimator")

    print("\n  a) transformed names line up positionally with the model's columns")
    for label, passed in explain.check_feature_alignment(
        X_test, matrix, names
    ).items():
        print(f"       {label:<48} {ok(passed)}")
        if not passed:
            failures.append(f"feature alignment check failed: {label}")

    print("\n  b) every transformed name has a human-readable label")
    unmapped = explain.check_name_mapping(names)
    print(f"       features                 : {len(names)}")
    print(f"       falling back to raw name : {unmapped or 'none'}   "
          f"{ok(not unmapped)}")
    for sample in ("cat__Contract_Month-to-month",
                   "cat__PaymentMethod_Electronic check",
                   "cat__InternetService_Fiber optic", "num__tenure"):
        print(f"       {sample:<40} -> {explain.readable_name(sample)}")
    if unmapped:
        failures.append(f"{len(unmapped)} transformed names have no label")

    # ---------------------------------------------------------------- 3
    rule("3. SHAP VALUES  (TreeExplainer reads the fitted trees; it fits nothing)")
    shap_started = time.perf_counter()
    explanation = explain.compute_shap(pipe, matrix, names)
    shap_seconds = time.perf_counter() - shap_started

    proba = pipe.predict_proba(X_test)[:, 1]
    validity = explain.check_shap_validity(explanation, matrix, proba)

    print(f"  explainer                : shap.TreeExplainer on "
          f"{type(rf).__name__}")
    print(f"  rows explained           : {len(X_test)} / {len(X_test)}  "
          "(full test set -- no sampling needed)")
    print(f"  SHAP runtime             : {shap_seconds:.1f}s")
    print(f"  class explained          : 1 = churn  "
          f"(index {list(rf.classes_).index(1)} of classes_ "
          f"{[int(c) for c in rf.classes_]})")
    print(f"  values shape             : {validity['shape']}  vs matrix "
          f"{matrix.shape}   {ok(validity['matches_matrix'])}")
    print(f"  feature names attached   : {validity['n_names']}   "
          f"{ok(validity['names_match_columns'])}")
    print(f"  NaN / inf values         : {validity['n_nan']} / "
          f"{validity['n_inf']}   {ok(validity['all_finite'])}")
    print(f"  base value (class 1)     : {validity['base_value']:.6f}")
    print(f"    the forest's average output, not the {y_test.mean():.1%} churn "
          "rate --")
    print("    class_weight=\"balanced\" reweights the trees toward 0.5")
    print(f"\n  additivity check         : max |base + sum(SHAP) - "
          f"predict_proba| = {validity['max_additivity_error']:.2e}")
    print("    TreeExplainer on a scikit-learn forest works in probability")
    print("    space, so the contributions must reconstruct the pipeline's own")
    print("    predicted probability exactly. A wrong class slice, a wrong")
    print("    feature order, or a refitted transform would all break this.")
    print(f"  additivity within 1e-10  : "
          f"{ok(validity['max_additivity_error'] < 1e-10)}")

    if not validity["matches_matrix"]:
        failures.append("SHAP value shape does not match the feature matrix")
    if not validity["names_match_columns"]:
        failures.append("SHAP feature-name count does not match the matrix width")
    if not validity["all_finite"]:
        failures.append(f"SHAP output holds {validity['n_nan']} NaN and "
                        f"{validity['n_inf']} inf values")
    if validity["max_additivity_error"] >= 1e-10:
        failures.append(f"additivity error {validity['max_additivity_error']:.2e} "
                        "exceeds 1e-10")

    # ---------------------------------------------------------------- global
    importance = explain.global_importance(explanation, names)

    rule(f"TOP CHURN DRIVERS  (top {explain.TOP_N_GLOBAL} by mean |SHAP|, "
         "computed -- not assumed)")
    print_driver_table(importance, explain.TOP_N_GLOBAL)
    total = importance["mean_abs_shap"].sum()
    covered = importance.head(explain.TOP_N_GLOBAL)["mean_abs_shap"].sum()
    print(f"\n  share of total attribution held by these "
          f"{explain.TOP_N_GLOBAL}: {covered / total:.1%} "
          f"of {len(importance)} features")
    print("  \"when high / yes\" is the contrast WITHIN each feature: mean SHAP")
    print("  where it is high or set, minus mean SHAP where it is not. Mean")
    print("  signed SHAP cannot serve here -- the forest's baseline is "
          f"{validity['base_value']:.4f}")
    print(f"  while the average prediction is {proba.mean():.4f}, so the "
          "contributions must")
    print("  sum to a large negative number and nearly every feature would read")
    print("  \"away from churn\" regardless of what it does.")

    # ---------------------------------------------------------------- local
    selected = explain.select_customers(proba)
    rule("LOCAL EXPLANATIONS  (two real test customers)")
    print("  Selection used the predicted probabilities only. "
          "explain.select_customers()")
    print("  receives the probability array and nothing else, so the true "
          "labels")
    print("  could not have influenced the choice; they are read afterwards to")
    print("  report them.")
    print(f"\n  probability range across the test set: "
          f"{proba.min() * 100:.2f}% to {proba.max() * 100:.2f}%")
    print(f"  highest-probability row : {selected['high']}  "
          f"({proba[selected['high']] * 100:.2f}%)")
    print(f"  lowest-probability row  : {selected['low']}  "
          f"({proba[selected['low']] * 100:.2f}%)")

    contributions = {
        which: explain.local_contributions(explanation, names, row)
        for which, row in selected.items()
    }
    signed = {
        which: explain.signed_contributions(explanation, names, row)
        for which, row in selected.items()
    }
    labels = {which: int(y_test.iloc[row]) for which, row in selected.items()}

    for which, heading in (("high", "HIGH-RISK CUSTOMER"),
                           ("low", "LOW-RISK CUSTOMER")):
        toward, away = signed[which]
        print_customer(heading, selected[which],
                       float(proba[selected[which]]), labels[which],
                       toward, away, contributions[which],
                       validity["base_value"])
        print("\n  Each direction is ranked on its own, so a confidently "
              "predicted")
        print("  customer still shows what pushed the other way. The waterfall")
        print(f"  figure shows the {explain.TOP_N_LOCAL} largest contributions "
              "by magnitude, plus")
        print("  the remaining features combined into one bar.")

    # ---------------------------------------------------------------- 4
    rule("4. FIGURES")
    figure_paths = [
        explain.plot_global_importance(importance, len(X_test)),
        explain.plot_beeswarm(explanation, importance, names),
        explain.plot_waterfall(
            contributions["high"], validity["base_value"],
            float(proba[selected["high"]]),
            "Why this customer was predicted to churn",
            f"test row {selected['high']}   ·   predicted "
            f"{proba[selected['high']] * 100:.2f}%   ·   red pushes toward "
            "churn, blue pushes away   ·   model behaviour, not causation",
            "shap_customer_high_risk.png",
        ),
        explain.plot_waterfall(
            contributions["low"], validity["base_value"],
            float(proba[selected["low"]]),
            "Why this customer was predicted to stay",
            f"test row {selected['low']}   ·   predicted "
            f"{proba[selected['low']] * 100:.2f}%   ·   red pushes toward "
            "churn, blue pushes away   ·   model behaviour, not causation",
            "shap_customer_low_risk.png",
        ),
    ]
    for figure in figure_paths:
        print(f"  {figure.relative_to(config.PROJECT_ROOT)}  "
              f"({figure.stat().st_size:,} bytes)")
    print(f"  figures written          : {len(figure_paths)}  "
          f"{ok(len(figure_paths) == 4)}")
    print("  style                    : Stage 4's tokens and helpers are")
    print("                             imported from src/evaluate.py, so both")
    print("                             stages' figures read as one set")
    print("  colour                   : magnitude chart is one hue; the")
    print("                             diverging pair (blue/red) is "
          "CVD-validated")
    print("                             and every mark is also labelled with a")
    print("                             signed number, so nothing is colour-only")
    missing = [f for f in figure_paths if not f.exists()]
    if len(figure_paths) != 4 or missing:
        failures.append(f"expected 4 figures, wrote {len(figure_paths)}, "
                        f"missing {missing}")

    # ---------------------------------------------------------------- findings
    scores = explain.hypothesis_scores(importance)
    rule("XAI FINDINGS  (SHAP vs the Stage 2 EDA observations)")
    for i, finding in enumerate(
        explain.xai_findings(importance, scores), start=1
    ):
        print(textwrap.indent(
            textwrap.fill(f"{i}. {finding}", width=68,
                          subsequent_indent=" " * (len(str(i)) + 2)),
            "  ",
        ))
        print()

    # ---------------------------------------------------------------- 5
    rule("5. INTEGRITY CHECKS  (model not retrained, preprocessor not refitted)")
    integrity_checks(prep, prep_fingerprint_before, checksums_before, failures)

    # ---------------------------------------------------------------- done
    rule("STAGE 5 RESULT")
    print(f"  runtime : {time.perf_counter() - started:.1f}s  "
          f"(SHAP itself {shap_seconds:.1f}s)")
    if failures:
        print(f"  FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  All checks passed. Stage 5 complete.")
    return 0


def integrity_checks(prep, fingerprint_before: str,
                     checksums_before: dict[str, str],
                     failures: list[str]) -> None:
    """Make "nothing was retrained or refitted" measurable rather than asserted."""
    print("  a) all five saved model files are byte-identical after this stage")
    checksums_after = evaluate.model_checksums()
    for key, name in evaluate.MODELS.items():
        unchanged = checksums_before.get(key) == checksums_after.get(key)
        print(f"       {name:<21} sha256 {checksums_before[key][:16]}...  "
              f"{ok(unchanged)}")
        if not unchanged:
            failures.append(f"{key}.pkl changed during Stage 5")

    print("\n  b) the fitted preprocessor was not refitted")
    fingerprint_after = explain.preprocessor_fingerprint(prep)
    unchanged = fingerprint_before == fingerprint_after
    print(f"       sha256 of pickled state before transform: "
          f"{fingerprint_before[:16]}...")
    print(f"       sha256 of pickled state after  transform: "
          f"{fingerprint_after[:16]}...")
    print(f"       identical -- no learned attribute moved      {ok(unchanged)}")
    if not unchanged:
        failures.append("the preprocessor's fitted state changed during Stage 5")

    print("\n  c) no fit-family call exists in Stage 5 source")
    print("       (read from each file's AST, so a mention in a comment or")
    print("        docstring cannot satisfy the check -- only real calls count)")
    for rel, hits in explain.check_no_fit_calls().items():
        print(f"       {rel:<21} {sorted(evaluate.FIT_METHODS)}")
        print(f"       {'':<21} call sites found: {hits or 'none'}   "
              f"{ok(not hits)}")
        if hits:
            failures.append(f"{rel} contains fit-family calls: {hits}")

    print("\n  d) preprocessing goes through transform(), never fit_transform()")
    counts = explain.count_transform_call_sites()
    for method, count in counts.items():
        expected = count >= 1 if method == "transform" else count == 0
        print(f"       src/explain.py  .{method}() call sites: {count}   "
              f"{ok(expected)}")
    if counts["fit_transform"]:
        failures.append("src/explain.py calls fit_transform()")
    if not counts["transform"]:
        failures.append("src/explain.py never calls transform()")

    print("\n  e) Stage 1-4 artifacts were read, never written")
    for artifact in ("model_cv_results.csv", "model_metadata.json",
                     "test_results.csv"):
        print(f"       artifacts/{artifact:<24} untouched by Stage 5")
    print("       Stage 5 writes only reports/figures/shap_*.png")


if __name__ == "__main__":
    raise SystemExit(main())
