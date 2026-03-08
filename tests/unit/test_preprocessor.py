"""preprocessor 모듈 단위 테스트."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import (
    handle_missing_values,
    handle_outliers,
    run_pipeline,
    scale_features,
    split_timeseries,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """100행 테스트용 DataFrame을 생성한다."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "feature_a": np.random.randn(n) * 10 + 50,
            "feature_b": np.random.randn(n) * 5 + 20,
            "feature_c": np.random.randn(n) * 100,
            "target": np.random.choice([-1, 0, 1], size=n),
        },
        index=dates,
    )


class TestHandleMissingValues:
    """handle_missing_values 테스트."""

    def test_drop(self, sample_df: pd.DataFrame) -> None:
        """NaN 행이 제거되어야 한다."""
        df = sample_df.copy()
        df.iloc[0, 0] = np.nan
        df.iloc[5, 1] = np.nan
        result = handle_missing_values(df, method="drop")
        assert result.isna().sum().sum() == 0
        assert len(result) == 98

    def test_ffill(self, sample_df: pd.DataFrame) -> None:
        """NaN이 forward fill되어야 한다."""
        df = sample_df.copy()
        df.iloc[5, 0] = np.nan
        result = handle_missing_values(df, method="ffill")
        assert result.isna().sum().sum() == 0
        assert len(result) == 100

    def test_interpolate(self, sample_df: pd.DataFrame) -> None:
        """NaN이 보간되어야 한다."""
        df = sample_df.copy()
        df.iloc[5, 0] = np.nan
        result = handle_missing_values(df, method="interpolate")
        assert result.isna().sum().sum() == 0

    def test_no_missing(self, sample_df: pd.DataFrame) -> None:
        """결측치가 없으면 그대로 반환한다."""
        result = handle_missing_values(sample_df)
        assert len(result) == len(sample_df)


class TestHandleOutliers:
    """handle_outliers 테스트."""

    def test_clips_extreme_values(self, sample_df: pd.DataFrame) -> None:
        """극단값이 클리핑되어야 한다."""
        df = sample_df.copy()
        df.iloc[0, 0] = 99999.0  # 극단 이상치
        result = handle_outliers(df, z_threshold=3.0)
        assert result.iloc[0, 0] < 99999.0

    def test_preserves_normal_values(self, sample_df: pd.DataFrame) -> None:
        """정상값은 변경되지 않아야 한다."""
        result = handle_outliers(sample_df, z_threshold=5.0)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_skips_target(self, sample_df: pd.DataFrame) -> None:
        """target 컬럼은 건드리지 않아야 한다."""
        df = sample_df.copy()
        original_target = df["target"].copy()
        result = handle_outliers(df)
        pd.testing.assert_series_equal(result["target"], original_target)


class TestScaleFeatures:
    """scale_features 테스트."""

    def test_standard_scaling(self, sample_df: pd.DataFrame) -> None:
        """StandardScaler 적용 후 평균 ~0, 표준편차 ~1이어야 한다."""
        df, scaler = scale_features(sample_df.copy(), exclude_columns=["target"], method="standard")
        assert abs(df["feature_a"].mean()) < 0.1
        assert abs(df["feature_a"].std() - 1.0) < 0.2

    def test_minmax_scaling(self, sample_df: pd.DataFrame) -> None:
        """MinMaxScaler 적용 후 0~1 범위여야 한다."""
        df, scaler = scale_features(sample_df.copy(), exclude_columns=["target"], method="minmax")
        assert df["feature_a"].min() >= -0.01
        assert df["feature_a"].max() <= 1.01

    def test_excludes_target(self, sample_df: pd.DataFrame) -> None:
        """target 컬럼은 스케일링되지 않아야 한다."""
        original_target = sample_df["target"].copy()
        df, _ = scale_features(sample_df.copy(), exclude_columns=["target"])
        pd.testing.assert_series_equal(df["target"], original_target)

    def test_returns_scaler(self, sample_df: pd.DataFrame) -> None:
        """학습된 scaler 객체를 반환해야 한다."""
        _, scaler = scale_features(sample_df.copy(), exclude_columns=["target"])
        assert hasattr(scaler, "transform")

    def test_invalid_method(self, sample_df: pd.DataFrame) -> None:
        """잘못된 method는 ValueError."""
        with pytest.raises(ValueError):
            scale_features(sample_df.copy(), method="invalid")


class TestSplitTimeseries:
    """split_timeseries 테스트."""

    def test_default_ratios(self, sample_df: pd.DataFrame) -> None:
        """기본 비율(0.6/0.2/0.2)로 분할되어야 한다."""
        train, val, test = split_timeseries(sample_df)
        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20

    def test_no_overlap(self, sample_df: pd.DataFrame) -> None:
        """train/val/test 간 시간이 겹치지 않아야 한다."""
        train, val, test = split_timeseries(sample_df)
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()

    def test_preserves_order(self, sample_df: pd.DataFrame) -> None:
        """시간 순서가 유지되어야 한다."""
        train, val, test = split_timeseries(sample_df)
        assert train.index.is_monotonic_increasing
        assert val.index.is_monotonic_increasing
        assert test.index.is_monotonic_increasing

    def test_total_rows(self, sample_df: pd.DataFrame) -> None:
        """분할 후 전체 행 수가 보존되어야 한다."""
        train, val, test = split_timeseries(sample_df)
        assert len(train) + len(val) + len(test) == len(sample_df)

    def test_invalid_ratios(self, sample_df: pd.DataFrame) -> None:
        """비율 합이 1.0이 아니면 ValueError."""
        with pytest.raises(ValueError):
            split_timeseries(sample_df, train_ratio=0.5, val_ratio=0.2, test_ratio=0.1)

    def test_custom_ratios(self, sample_df: pd.DataFrame) -> None:
        """커스텀 비율로 분할 가능해야 한다."""
        train, val, test = split_timeseries(sample_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15


class TestRunPipelineRawOhlcv:
    """run_pipeline의 원본 OHLCV 보존 테스트."""

    @pytest.fixture
    def ohlcv_parquet(self, tmp_path: object) -> str:
        """테스트용 OHLCV Parquet 파일을 생성한다."""
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        base = 40000.0
        close = base + np.cumsum(np.random.randn(n) * 100)
        df = pd.DataFrame(
            {
                "open": close - np.abs(np.random.randn(n) * 20),
                "high": close + np.abs(np.random.randn(n) * 50),
                "low": close - np.abs(np.random.randn(n) * 50),
                "close": close,
                "volume": np.random.uniform(100, 1000, n),
            },
            index=dates,
        )
        path = tmp_path / "test_ohlcv.parquet"  # type: ignore[operator]
        df.to_parquet(path)
        return str(path)

    def test_raw_columns_exist(self, ohlcv_parquet: str, tmp_path: object) -> None:
        """run_pipeline 결과에 close_raw 등 원본 컬럼이 존재해야 한다."""
        result = run_pipeline(ohlcv_parquet, output_dir=tmp_path / "out")  # type: ignore[operator]
        for split_name in ("train", "val", "test"):
            df = result[split_name]
            assert "close_raw" in df.columns, f"{split_name}에 close_raw 없음"
            assert "open_raw" in df.columns, f"{split_name}에 open_raw 없음"
            assert "volume_raw" in df.columns, f"{split_name}에 volume_raw 없음"

    def test_raw_columns_not_scaled(self, ohlcv_parquet: str, tmp_path: object) -> None:
        """close_raw는 스케일링되지 않고 원본 가격 범위를 유지해야 한다."""
        result = run_pipeline(ohlcv_parquet, output_dir=tmp_path / "out")  # type: ignore[operator]
        for split_name in ("train", "val", "test"):
            df = result[split_name]
            # 원본 가격은 ~40000 범위여야 한다
            assert df["close_raw"].mean() > 1000.0, f"{split_name} close_raw가 스케일링됨"
            # 스케일링된 close는 평균 ~0이어야 한다
            assert abs(df["close"].mean()) < 100.0, f"{split_name} close가 스케일링 안 됨"

    def test_raw_matches_original(self, ohlcv_parquet: str, tmp_path: object) -> None:
        """close_raw와 스케일링 전 close가 동일해야 한다."""
        original = pd.read_parquet(ohlcv_parquet)
        result = run_pipeline(ohlcv_parquet, output_dir=tmp_path / "out")  # type: ignore[operator]
        # train의 close_raw는 원본 close와 같아야 한다 (피처 생성 후 결측치 제거 영향은 있음)
        train = result["train"]
        common_idx = train.index.intersection(original.index)
        if len(common_idx) > 0:
            np.testing.assert_array_almost_equal(
                train.loc[common_idx, "close_raw"].values,
                original.loc[common_idx, "close"].values,
                decimal=2,
            )
