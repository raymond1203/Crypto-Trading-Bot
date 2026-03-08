"""MarketRegimeDetector 단위 테스트.

합성 OHLCV 데이터로 BULL/BEAR/SIDEWAYS 레짐 감지 로직을 검증한다.
"""

import pandas as pd
import pytest
import yaml

from src.strategy.regime import MarketRegimeDetector, add_regime_features


def _make_ohlcv_with_adx(
    closes: list[float],
    adx_values: list[float],
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """close와 ADX 값을 지정한 합성 OHLCV DataFrame을 생성한다."""
    n = len(closes)
    dates = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [100.0] * n,
            "adx": adx_values,
        },
        index=dates,
    )


class TestMarketRegimeDetector:
    """MarketRegimeDetector 테스트."""

    def test_default_params(self) -> None:
        """기본 파라미터가 적용되어야 한다."""
        detector = MarketRegimeDetector()
        assert detector.config["adx_trend_threshold"] == 25
        assert detector.config["sma_slope_window"] == 10
        assert detector.config["sma_window"] == 25

    def test_custom_config(self) -> None:
        """커스텀 설정이 기본값을 오버라이드해야 한다."""
        detector = MarketRegimeDetector(config={"adx_trend_threshold": 30})
        assert detector.config["adx_trend_threshold"] == 30
        assert detector.config["sma_window"] == 25  # 기본값 유지

    def test_from_config_file(self, tmp_path: object) -> None:
        """YAML 설정 파일에서 regime 섹션을 로드해야 한다."""
        config = {"regime": {"adx_trend_threshold": 20}}
        config_path = tmp_path / "config.yaml"  # type: ignore[operator]
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        detector = MarketRegimeDetector(config_path=config_path)
        assert detector.config["adx_trend_threshold"] == 20

    def test_detect_bull(self) -> None:
        """ADX > threshold이고 가격 상승이면 BULL이어야 한다."""
        # 꾸준히 상승하는 가격 + ADX 30 (추세장)
        n = 50
        closes = [100.0 + i * 2.0 for i in range(n)]
        adx_values = [30.0] * n
        df = _make_ohlcv_with_adx(closes, adx_values)

        detector = MarketRegimeDetector()
        regimes = detector.detect(df)

        # SMA slope 계산을 위해 초반은 SIDEWAYS일 수 있으므로 후반만 검증
        assert regimes[-1] == MarketRegimeDetector.BULL

    def test_detect_bear(self) -> None:
        """ADX > threshold이고 가격 하락이면 BEAR이어야 한다."""
        n = 50
        closes = [200.0 - i * 2.0 for i in range(n)]
        adx_values = [30.0] * n
        df = _make_ohlcv_with_adx(closes, adx_values)

        detector = MarketRegimeDetector()
        regimes = detector.detect(df)

        assert regimes[-1] == MarketRegimeDetector.BEAR

    def test_detect_sideways(self) -> None:
        """ADX < threshold이면 SIDEWAYS이어야 한다."""
        n = 50
        closes = [100.0 + (i % 5) * 0.5 for i in range(n)]  # 횡보
        adx_values = [15.0] * n  # 약한 추세
        df = _make_ohlcv_with_adx(closes, adx_values)

        detector = MarketRegimeDetector()
        regimes = detector.detect(df)

        assert regimes[-1] == MarketRegimeDetector.SIDEWAYS

    def test_missing_adx_without_raw_raises(self) -> None:
        """adx와 raw 컬럼 모두 없으면 ValueError가 발생해야 한다."""
        df = pd.DataFrame({"close": [100.0]}, index=pd.date_range("2024-01-01", periods=1))
        detector = MarketRegimeDetector()

        with pytest.raises(ValueError, match="adx"):
            detector.detect(df)

    def test_missing_close_raises(self) -> None:
        """close 컬럼이 없으면 ValueError가 발생해야 한다."""
        df = pd.DataFrame({"adx": [30.0]}, index=pd.date_range("2024-01-01", periods=1))
        detector = MarketRegimeDetector()

        with pytest.raises(ValueError, match="close"):
            detector.detect(df)

    def test_detect_with_raw_columns(self) -> None:
        """raw 컬럼이 있으면 ADX를 재계산하여 레짐을 감지해야 한다."""
        n = 60
        closes = [100.0 + i * 2.0 for i in range(n)]
        dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "close": [0.0] * n,  # 스케일링된 값 (사용 안 됨)
                "close_raw": closes,
                "high_raw": [c * 1.02 for c in closes],
                "low_raw": [c * 0.98 for c in closes],
                "adx": [0.0] * n,  # 스케일링된 값 (사용 안 됨)
            },
            index=dates,
        )
        detector = MarketRegimeDetector()
        regimes = detector.detect(df)

        assert len(regimes) == n
        # raw 가격으로 ADX 재계산되므로 후반에 BULL/BEAR 감지 가능
        assert not all(r == MarketRegimeDetector.SIDEWAYS for r in regimes[-10:])


class TestRegimeFeatures:
    """add_regime_features 테스트."""

    def test_adds_regime_columns(self) -> None:
        """regime 관련 컬럼이 추가되어야 한다."""
        n = 50
        closes = [100.0 + i for i in range(n)]
        adx_values = [30.0] * n
        df = _make_ohlcv_with_adx(closes, adx_values)

        df = add_regime_features(df)
        assert "regime" in df.columns
        assert "regime_bull" in df.columns
        assert "regime_bear" in df.columns
        assert "regime_sideways" in df.columns

    def test_one_hot_exclusive(self) -> None:
        """one-hot 인코딩은 각 행에서 정확히 하나만 1이어야 한다."""
        n = 50
        closes = [100.0 + i for i in range(n)]
        adx_values = [30.0] * n
        df = _make_ohlcv_with_adx(closes, adx_values)

        df = add_regime_features(df)
        one_hot_sum = df["regime_bull"] + df["regime_bear"] + df["regime_sideways"]
        assert (one_hot_sum == 1).all()

    def test_regime_values(self) -> None:
        """regime 컬럼은 -1, 0, 1 값만 가져야 한다."""
        n = 50
        closes = [100.0 + i for i in range(n)]
        adx_values = [30.0] * n
        df = _make_ohlcv_with_adx(closes, adx_values)

        df = add_regime_features(df)
        assert set(df["regime"].unique()).issubset({-1, 0, 1})
