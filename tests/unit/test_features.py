"""features 모듈 단위 테스트.

실제 OHLCV 구조의 샘플 데이터로 피처 생성 로직을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    add_custom_features,
    add_momentum_features,
    add_time_features,
    add_trend_features,
    add_volatility_features,
    add_volume_features,
    build_features,
    create_target,
    select_features,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """200행 랜덤 OHLCV 데이터를 생성한다."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")

    close = 42000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 30
    volume = np.abs(np.random.randn(n) * 100) + 50

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestTrendFeatures:
    """add_trend_features 테스트."""

    def test_adds_sma_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """SMA 컬럼이 추가되어야 한다."""
        df = add_trend_features(sample_ohlcv)
        assert "sma_7" in df.columns
        assert "sma_25" in df.columns
        assert "sma_99" in df.columns

    def test_adds_macd_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """MACD 관련 컬럼이 추가되어야 한다."""
        df = add_trend_features(sample_ohlcv)
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_histogram" in df.columns

    def test_adds_adx_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """ADX 관련 컬럼이 추가되어야 한다."""
        df = add_trend_features(sample_ohlcv)
        assert "adx" in df.columns
        assert "adx_pos" in df.columns
        assert "adx_neg" in df.columns


class TestMomentumFeatures:
    """add_momentum_features 테스트."""

    def test_adds_rsi(self, sample_ohlcv: pd.DataFrame) -> None:
        """RSI 컬럼이 추가되어야 한다."""
        df = add_momentum_features(sample_ohlcv)
        assert "rsi_14" in df.columns
        assert "rsi_7" in df.columns

    def test_rsi_range(self, sample_ohlcv: pd.DataFrame) -> None:
        """RSI는 0~100 범위여야 한다."""
        df = add_momentum_features(sample_ohlcv)
        valid = df["rsi_14"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_adds_stochastic(self, sample_ohlcv: pd.DataFrame) -> None:
        """Stochastic 컬럼이 추가되어야 한다."""
        df = add_momentum_features(sample_ohlcv)
        assert "stoch_k" in df.columns
        assert "stoch_d" in df.columns


class TestVolatilityFeatures:
    """add_volatility_features 테스트."""

    def test_adds_bollinger_bands(self, sample_ohlcv: pd.DataFrame) -> None:
        """Bollinger Bands 컬럼이 추가되어야 한다."""
        df = add_volatility_features(sample_ohlcv)
        assert "bb_upper" in df.columns
        assert "bb_lower" in df.columns
        assert "bb_width" in df.columns

    def test_bb_order(self, sample_ohlcv: pd.DataFrame) -> None:
        """upper > middle > lower 순서여야 한다."""
        df = add_volatility_features(sample_ohlcv)
        valid = df[["bb_upper", "bb_middle", "bb_lower"]].dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_adds_atr(self, sample_ohlcv: pd.DataFrame) -> None:
        """ATR 컬럼이 추가되어야 한다."""
        df = add_volatility_features(sample_ohlcv)
        assert "atr_14" in df.columns


class TestVolumeFeatures:
    """add_volume_features 테스트."""

    def test_adds_obv(self, sample_ohlcv: pd.DataFrame) -> None:
        """OBV 컬럼이 추가되어야 한다."""
        df = add_volume_features(sample_ohlcv)
        assert "obv" in df.columns

    def test_adds_mfi(self, sample_ohlcv: pd.DataFrame) -> None:
        """MFI 컬럼이 추가되어야 한다."""
        df = add_volume_features(sample_ohlcv)
        assert "mfi" in df.columns

    def test_volume_ratio(self, sample_ohlcv: pd.DataFrame) -> None:
        """volume_ratio가 양수여야 한다."""
        df = add_volume_features(sample_ohlcv)
        valid = df["volume_ratio"].dropna()
        assert (valid > 0).all()


class TestCustomFeatures:
    """add_custom_features 테스트."""

    def test_adds_returns(self, sample_ohlcv: pd.DataFrame) -> None:
        """수익률 컬럼이 추가되어야 한다."""
        df = add_trend_features(sample_ohlcv)  # sma_* 필요
        df = add_custom_features(df)
        for period in [1, 3, 5, 10, 20]:
            assert f"return_{period}" in df.columns

    def test_adds_candle_patterns(self, sample_ohlcv: pd.DataFrame) -> None:
        """캔들 패턴 컬럼이 추가되어야 한다."""
        df = add_trend_features(sample_ohlcv)
        df = add_custom_features(df)
        assert "candle_body" in df.columns
        assert "upper_shadow" in df.columns
        assert "lower_shadow" in df.columns


class TestTimeFeatures:
    """add_time_features 테스트."""

    def test_default_no_raw_time(self, sample_ohlcv: pd.DataFrame) -> None:
        """기본값은 raw 시간 피처를 생성하지 않아야 한다."""
        df = add_time_features(sample_ohlcv)
        assert "hour" not in df.columns
        assert "day_of_week" not in df.columns
        assert "month" not in df.columns
        # sin/cos는 생성되어야 함
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns
        assert "dow_sin" in df.columns
        assert "dow_cos" in df.columns

    def test_use_raw_adds_raw_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """use_raw=True면 raw 시간 피처도 추가되어야 한다."""
        df = add_time_features(sample_ohlcv, use_raw=True)
        assert "hour" in df.columns
        assert "day_of_week" in df.columns
        assert "month" in df.columns
        assert "hour_sin" in df.columns

    def test_sin_cos_range(self, sample_ohlcv: pd.DataFrame) -> None:
        """sin/cos 값은 -1~1 범위여야 한다."""
        df = add_time_features(sample_ohlcv)
        assert df["hour_sin"].min() >= -1
        assert df["hour_sin"].max() <= 1


class TestBuildFeatures:
    """build_features 통합 테스트."""

    def test_feature_count(self, sample_ohlcv: pd.DataFrame) -> None:
        """20개 이상의 피처가 생성되어야 한다."""
        df = build_features(sample_ohlcv)
        feature_count = len(df.columns) - 5  # OHLCV 제외
        assert feature_count >= 20

    def test_no_nan(self, sample_ohlcv: pd.DataFrame) -> None:
        """NaN이 없어야 한다."""
        df = build_features(sample_ohlcv)
        assert df.isna().sum().sum() == 0

    def test_index_preserved(self, sample_ohlcv: pd.DataFrame) -> None:
        """timestamp 인덱스가 유지되어야 한다."""
        df = build_features(sample_ohlcv)
        assert df.index.name == sample_ohlcv.index.name


class TestCreateTarget:
    """create_target 테스트."""

    def test_target_values(self, sample_ohlcv: pd.DataFrame) -> None:
        """타겟은 -1, 0, 1 값만 가져야 한다."""
        df = build_features(sample_ohlcv)
        df = create_target(df)
        assert set(df["target"].unique()).issubset({-1, 0, 1})

    def test_removes_tail_rows(self, sample_ohlcv: pd.DataFrame) -> None:
        """마지막 horizon행이 제거되어야 한다."""
        df = build_features(sample_ohlcv)
        len_before = len(df)
        df = create_target(df, horizon=4)
        assert len(df) == len_before - 4

    def test_custom_threshold(self, sample_ohlcv: pd.DataFrame) -> None:
        """threshold가 높으면 관망(0) 비율이 증가해야 한다."""
        df = build_features(sample_ohlcv)
        df_low = create_target(df.copy(), threshold=0.001)
        df_high = create_target(df.copy(), threshold=0.05)
        hold_ratio_low = (df_low["target"] == 0).mean()
        hold_ratio_high = (df_high["target"] == 0).mean()
        assert hold_ratio_high >= hold_ratio_low


class TestSelectFeatures:
    """select_features 테스트."""

    def test_reduces_feature_count(self, sample_ohlcv: pd.DataFrame) -> None:
        """상관관계 필터링으로 피처 수가 감소해야 한다."""
        df = build_features(sample_ohlcv)
        original_count = len(df.columns)
        df_selected, dropped = select_features(df, corr_threshold=0.90)
        assert len(df_selected.columns) < original_count
        assert len(dropped) > 0

    def test_preserves_ohlcv(self, sample_ohlcv: pd.DataFrame) -> None:
        """OHLCV 컬럼은 보존되어야 한다."""
        df = build_features(sample_ohlcv)
        df_selected, _ = select_features(df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df_selected.columns

    def test_no_nan_introduced(self, sample_ohlcv: pd.DataFrame) -> None:
        """피처 선택 후 NaN이 생기지 않아야 한다."""
        df = build_features(sample_ohlcv)
        df_selected, _ = select_features(df)
        assert df_selected.isna().sum().sum() == 0

    def test_low_variance_filter(self) -> None:
        """분산이 거의 0인 피처가 제거되어야 한다."""
        df = pd.DataFrame({
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [100.0, 200.0, 300.0],
            "constant_feat": [1.0, 1.0, 1.0],  # 분산 0
            "good_feat": [1.0, 5.0, 10.0],
        })
        df_selected, dropped = select_features(df, variance_threshold=1e-8)
        assert "constant_feat" in dropped
        assert "good_feat" in df_selected.columns

    def test_build_features_with_selection(self, sample_ohlcv: pd.DataFrame) -> None:
        """build_features에서 apply_selection=True가 작동해야 한다."""
        df_no_sel = build_features(sample_ohlcv.copy(), apply_selection=False)
        df_sel = build_features(sample_ohlcv.copy(), apply_selection=True)
        assert len(df_sel.columns) < len(df_no_sel.columns)
