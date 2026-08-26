"""Stage 1: data loading, validation, preprocessing, and train/test split.

Run from the project root:

    python run_stage1.py

Writes data/processed/train.csv and data/processed/test.csv, and prints a
report covering validation, cleaning, the split, and the leakage checks.
"""

from __future__ import annotations

import textwrap

from src import config, data_loader, preprocess


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def ok(passed: bool) -> str:
    return "PASS" if passed else "*** FAIL ***"


def main() -> int:
    failures: list[str] = []

    # ---------------------------------------------------------------- 1
    rule("1. DOWNLOAD  (checksum-verified)")
    path = data_loader.download_data()
    print(f"  file   : {path.relative_to(config.PROJECT_ROOT)}")
    print(f"  size   : {path.stat().st_size:,} bytes")
    print("  sha256 : verified against config.DATA_SHA256")

    # ---------------------------------------------------------------- 2
    rule("2. LOAD + VALIDATE RAW")
    raw = data_loader.load_raw(path)
    report = data_loader.validate_raw(raw)
    print(f"  shape                    : {report['rows']} x {report['columns']}")
    print(f"  duplicate customerID     : {report['duplicate_ids']}")
    print(f"  blank TotalCharges       : {report['blank_total_charges']} "
          f"(all tenure == 0: {report['blank_all_tenure_zero']})")
    print(f"  churn Yes / No           : {report['churn_yes']} / {report['churn_no']}")
    print(f"  churn rate               : {report['churn_rate']:.4f}")
    print(f"  majority-class accuracy  : {report['majority_class_accuracy']:.4f}"
          "   <-- any model near this is useless")

    # ---------------------------------------------------------------- 3
    rule("3. CLEAN + ENGINEER FEATURES")
    clean_df = clean_step(raw, failures)

    # ---------------------------------------------------------------- 4
    rule("4. STRATIFIED TRAIN / TEST SPLIT")
    train_df, test_df = preprocess.split(clean_df)
    preprocess.save_splits(train_df, test_df)

    total = len(train_df) + len(test_df)
    print(f"  train : {len(train_df):>5} rows  churn rate {train_df[config.TARGET].mean():.4f}")
    print(f"  test  : {len(test_df):>5} rows  churn rate {test_df[config.TARGET].mean():.4f}")
    print(f"  full  : {total:>5} rows  churn rate {clean_df[config.TARGET].mean():.4f}")
    print(f"  rows conserved           : {ok(total == config.EXPECTED_ROWS)}")
    print(f"  no overlap train/test    : "
          f"{ok(not set(train_df.index) & set(test_df.index))}")
    print(f"  written                  : "
          f"{config.TRAIN_CSV.relative_to(config.PROJECT_ROOT)}, "
          f"{config.TEST_CSV.relative_to(config.PROJECT_ROOT)}")
    if total != config.EXPECTED_ROWS:
        failures.append("row count not conserved by split")

    # ---------------------------------------------------------------- 5
    rule("5. LEAKAGE CHECKS  (nothing is learned from data in Stage 1)")
    leakage_checks(raw, train_df, failures)

    # ---------------------------------------------------------------- done
    rule("STAGE 1 RESULT")
    if failures:
        print(f"  FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  All checks passed. Stage 1 complete.")
    return 0


def clean_step(raw, failures: list[str]):
    """Clean the data and report on what changed."""
    blank_mask = raw["TotalCharges"].str.strip() == ""
    clean_df = preprocess.clean(raw)

    imputed_to_zero = bool((clean_df.loc[blank_mask, "TotalCharges"] == 0.0).all())
    n_nan = int(clean_df.isna().sum().sum())

    print(f"  shape after clean        : {clean_df.shape[0]} x {clean_df.shape[1]}")
    print(f"  TotalCharges blanks -> 0 : {int(blank_mask.sum())} rows  "
          f"{ok(imputed_to_zero)}")
    print(f"  NaN remaining            : {n_nan}  {ok(n_nan == 0)}")
    print(f"  customerID dropped       : {ok(config.ID_COL not in clean_df.columns)}")
    print("  engineered features      : is_new_customer, num_services, avg_charge")
    print(f"  service columns collapsed: {len(config.COLLAPSE_COLS)} "
          "(\"No internet/phone service\" -> \"No\")")

    if not imputed_to_zero:
        failures.append("blank TotalCharges not imputed to 0")
    if n_nan:
        failures.append(f"{n_nan} NaN values after cleaning")
    if config.ID_COL in clean_df.columns:
        failures.append("customerID still present")

    print("\n  engineered columns, first 5 rows:")
    preview = clean_df[["tenure", "MonthlyCharges", "TotalCharges",
                        "avg_charge", "num_services", "is_new_customer"]].head()
    print(textwrap.indent(preview.to_string(index=False), "    "))
    return clean_df


def leakage_checks(raw, train_df, failures: list[str]) -> None:
    """Print the three no-learning checks."""
    print("  a) build_preprocessor() returns an unfitted transformer")
    for variant, res in preprocess.check_preprocessor_unfitted().items():
        passed = (res["not_fitted_error_raised"]
                  and not res["fitted_attributes"]
                  and not res["has_transformers_"])
        print(f"       {variant:<12} NotFittedError raised: "
              f"{res['not_fitted_error_raised']} | "
              f"fitted attrs: {res['fitted_attributes'] or 'none'} | "
              f"transformers_: {res['has_transformers_']}   {ok(passed)}")
        if not passed:
            failures.append(f"preprocessor appears fitted ({variant})")

    print("\n  b) clean() is row-wise stateless (partition invariance)")
    for name, passed in preprocess.check_partition_invariance(raw).items():
        print(f"       clean(subset) == clean(full).loc[subset] -- "
              f"{name:<20} {ok(passed)}")
        if not passed:
            failures.append(f"clean() is not partition-invariant on {name}")

    print("\n  c) saved data carries no fitted transformation")
    art = preprocess.check_artifacts_untransformed(train_df)
    print(f"       MonthlyCharges mean = {art['monthly_charges_mean']:.4f} "
          f"(original scale, not ~0)   {ok(art['on_original_scale'])}")
    print(f"       Contract still strings: {art['contract_values']}   "
          f"{ok(art['contract_is_strings'])}")
    if not art["on_original_scale"]:
        failures.append("saved numerics appear scaled")
    if not art["contract_is_strings"]:
        failures.append("saved categoricals appear one-hot encoded")


if __name__ == "__main__":
    raise SystemExit(main())
