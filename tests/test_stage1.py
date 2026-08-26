"""Stage 1 tests: correctness, reproducibility, and no-leakage guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src import config, data_loader, preprocess


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    data_loader.download_data()
    return data_loader.load_raw()


@pytest.fixture(scope="module")
def clean_df(raw) -> pd.DataFrame:
    return preprocess.clean(raw)


# ---------------------------------------------------------------- validation
def test_raw_matches_expectations(raw):
    report = data_loader.validate_raw(raw)
    assert report["rows"] == config.EXPECTED_ROWS
    assert report["columns"] == config.EXPECTED_COLS
    assert report["duplicate_ids"] == 0
    assert report["blank_total_charges"] == config.EXPECTED_BLANK_TOTAL_CHARGES
    assert report["churn_yes"] == 1869
    assert report["churn_no"] == 5174


def test_validation_rejects_corrupt_data(raw):
    """Validation must actually fail on bad input, not just pass on good."""
    broken = raw.drop(columns=["Contract"])
    with pytest.raises(ValueError, match="validation failed"):
        data_loader.validate_raw(broken)


# ---------------------------------------------------------------- cleaning
def test_blank_total_charges_imputed_with_zero_not_median(raw, clean_df):
    """The 11 blanks are tenure-0 customers; the correct value is 0.

    A median would inject ~1397 of invented billing history into exactly the
    newest-customer segment the model most needs to get right.
    """
    blank_mask = raw["TotalCharges"].str.strip() == ""
    assert blank_mask.sum() == 11
    assert (raw.loc[blank_mask, "tenure"] == 0).all()

    imputed = clean_df.loc[blank_mask, "TotalCharges"]
    assert (imputed == 0.0).all()

    median = pd.to_numeric(raw["TotalCharges"].str.strip(),
                           errors="coerce").median()
    assert median > 1000, "sanity: the rejected median really is large"
    assert not np.isclose(imputed, median).any()


def test_is_new_customer_flag(raw, clean_df):
    assert clean_df["is_new_customer"].sum() == 11
    assert (clean_df.loc[clean_df["is_new_customer"] == 1, "tenure"] == 0).all()


def test_no_nan_after_cleaning(clean_df):
    assert clean_df.isna().sum().sum() == 0


def test_customer_id_dropped(clean_df):
    assert config.ID_COL not in clean_df.columns


def test_target_is_binary(clean_df):
    assert set(clean_df[config.TARGET].unique()) == {0, 1}
    assert clean_df[config.TARGET].sum() == 1869


def test_service_columns_collapsed(clean_df):
    """No structural placeholders should survive cleaning."""
    for col in config.COLLAPSE_COLS:
        values = set(clean_df[col].unique())
        assert "No internet service" not in values
        assert "No phone service" not in values
        assert values <= {"Yes", "No"}, f"{col} has unexpected values: {values}"


def test_internet_service_still_three_way(clean_df):
    """Collapsing must not flatten InternetService itself."""
    assert set(clean_df["InternetService"].unique()) == {"DSL", "Fiber optic", "No"}


def test_avg_charge_is_zero_for_new_customers(clean_df):
    new = clean_df[clean_df["is_new_customer"] == 1]
    assert (new["avg_charge"] == 0.0).all()


def test_num_services_within_bounds(clean_df):
    # 8 Yes/No service columns + 1 internet flag
    assert clean_df["num_services"].between(0, 9).all()


def test_columns_are_exactly_the_configured_features(clean_df):
    assert list(clean_df.columns) == config.FEATURE_COLS + [config.TARGET]


# ---------------------------------------------------------------- split
def test_split_sizes_and_stratification(clean_df):
    train_df, test_df = preprocess.split(clean_df)
    assert len(train_df) == 5634
    assert len(test_df) == 1409
    assert len(train_df) + len(test_df) == config.EXPECTED_ROWS

    full_rate = clean_df[config.TARGET].mean()
    assert train_df[config.TARGET].mean() == pytest.approx(full_rate, abs=1e-3)
    assert test_df[config.TARGET].mean() == pytest.approx(full_rate, abs=1e-3)


def test_split_has_no_overlap(clean_df):
    train_df, test_df = preprocess.split(clean_df)
    assert not set(train_df.index) & set(test_df.index)


def test_split_is_deterministic(clean_df):
    """Same seed must give byte-identical splits across runs."""
    a_train, a_test = preprocess.split(clean_df)
    b_train, b_test = preprocess.split(clean_df)
    assert_frame_equal(a_train, b_train)
    assert_frame_equal(a_test, b_test)


def test_clean_is_deterministic(raw):
    assert_frame_equal(preprocess.clean(raw), preprocess.clean(raw))


# ------------------------------------------------- NO-LEARNING GUARANTEES
def test_build_preprocessor_returns_unfitted():
    """The preprocessor must carry no learned state."""
    for scale in (True, False):
        pre = preprocess.build_preprocessor(scale=scale)
        with pytest.raises(NotFittedError):
            check_is_fitted(pre)
        fitted_attrs = [
            a for a in vars(pre) if a.endswith("_") and not a.startswith("__")
        ]
        assert fitted_attrs == [], f"scale={scale} exposes {fitted_attrs}"
        assert not hasattr(pre, "transformers_")
        assert not hasattr(pre, "n_features_in_")


def test_clean_is_partition_invariant(raw):
    """The core statelessness proof.

    If any cleaning step used an aggregate (a mean, a median, a category set),
    cleaning a subset would differ from cleaning everything and selecting those
    rows. The single-row case makes that impossible to fake.
    """
    results = preprocess.check_partition_invariance(raw)
    assert results, "no subsets were checked"
    failed = [name for name, passed in results.items() if not passed]
    assert not failed, f"clean() is not stateless on: {failed}"


def test_partition_invariance_check_is_sensitive(raw, monkeypatch):
    """Guard against a vacuous test.

    Deliberately make cleaning data-dependent (median imputation) and confirm
    the invariance check catches it. A check that cannot fail proves nothing.
    """
    original_clean = preprocess.clean

    def leaky_clean(df: pd.DataFrame) -> pd.DataFrame:
        out = original_clean(df)
        # An aggregate over whatever rows were passed in -- the classic leak.
        out["avg_charge"] = out["avg_charge"] + out["MonthlyCharges"].mean()
        return out

    monkeypatch.setattr(preprocess, "clean", leaky_clean)
    results = preprocess.check_partition_invariance(raw)
    assert not all(results.values()), (
        "the invariance check passed a knowingly leaky clean() -- "
        "it is not actually testing anything"
    )


def test_saved_splits_are_untransformed(clean_df, tmp_path, monkeypatch):
    """Nothing fitted may be baked into the files on disk."""
    monkeypatch.setattr(config, "TRAIN_CSV", tmp_path / "train.csv")
    monkeypatch.setattr(config, "TEST_CSV", tmp_path / "test.csv")

    train_df, test_df = preprocess.split(clean_df)
    preprocess.save_splits(train_df, test_df)
    reloaded_train, reloaded_test = preprocess.load_splits()

    art = preprocess.check_artifacts_untransformed(reloaded_train)
    assert art["on_original_scale"], "numerics look scaled"
    assert art["contract_is_strings"], "categoricals look one-hot encoded"
    assert art["monthly_charges_mean"] == pytest.approx(64.9, abs=1.0)
    assert set(art["contract_values"]) == {"Month-to-month", "One year", "Two year"}
    assert len(reloaded_train) == 5634
    assert len(reloaded_test) == 1409


def test_fitting_on_train_differs_from_fitting_on_all(clean_df):
    """Show that a leaked fit would be detectable rather than silent.

    If train-only and full-data statistics were identical, the whole
    train/test discipline would be unfalsifiable.
    """
    X = clean_df.drop(columns=[config.TARGET])
    train_df, _ = preprocess.split(clean_df)
    X_train = train_df.drop(columns=[config.TARGET])

    fitted_on_train = preprocess.build_preprocessor().fit(X_train)
    fitted_on_all = preprocess.build_preprocessor().fit(X)

    mean_train = fitted_on_train.named_transformers_["num"].mean_
    mean_all = fitted_on_all.named_transformers_["num"].mean_
    assert not np.allclose(mean_train, mean_all)


def test_preprocessor_transforms_test_without_refitting(clean_df):
    """Standard usage: fit on train, transform test. Test must not alter state."""
    train_df, test_df = preprocess.split(clean_df)
    X_train = train_df.drop(columns=[config.TARGET])
    X_test = test_df.drop(columns=[config.TARGET])

    pre = preprocess.build_preprocessor().fit(X_train)
    before = pre.named_transformers_["num"].mean_.copy()
    out = pre.transform(X_test)
    after = pre.named_transformers_["num"].mean_

    np.testing.assert_array_equal(before, after)
    assert out.shape[0] == len(X_test)
    # Scaled train data centres near 0; test data is scaled by *train* stats.
    assert pre.transform(X_train)[:, 0].mean() == pytest.approx(0, abs=1e-9)
