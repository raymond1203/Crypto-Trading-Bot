"""RiskManager 단위 테스트.

합성 OHLCV 데이터로 손절/익절/트레일링 스탑/MDD 제한/일일 한도/쿨다운
리스크 규칙의 신호 전처리 동작을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.strategy.risk import DynamicPositionSizer, RiskManager

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


def _make_ohlcv(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    """close 가격 리스트로 합성 OHLCV DataFrame을 생성한다."""
    n = len(closes)
    dates = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [100.0] * n,
        },
        index=dates,
    )


def _make_multiday_ohlcv(closes: list[float], dates: list[str]) -> pd.DataFrame:
    """여러 날짜에 걸친 OHLCV DataFrame을 생성한다."""
    index = pd.DatetimeIndex(dates, tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [100.0] * len(closes),
        },
        index=index,
    )


@pytest.fixture
def risk_manager() -> RiskManager:
    """기본 설정 RiskManager."""
    return RiskManager()


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:
    """리스크 관리기 초기화 테스트."""

    def test_default_params(self) -> None:
        """기본 파라미터가 적용되어야 한다."""
        rm = RiskManager()
        assert rm.config["stop_loss"] == 0.03
        assert rm.config["take_profit"] == 0.04
        assert rm.config["trailing_stop"] == 0.03
        assert rm.config["max_drawdown"] == 0.20
        assert rm.config["max_daily_trades"] == 10
        assert rm.config["cooldown_after_loss"] == 2
        assert rm.config["use_atr_stops"] is False

    def test_custom_config(self) -> None:
        """커스텀 설정이 기본값을 오버라이드해야 한다."""
        rm = RiskManager(config={"stop_loss": 0.05, "max_daily_trades": 5})
        assert rm.config["stop_loss"] == 0.05
        assert rm.config["max_daily_trades"] == 5
        assert rm.config["take_profit"] == 0.04  # 기본값 유지

    def test_from_config_file(self, tmp_path: object) -> None:
        """YAML 설정 파일에서 risk_management 섹션을 로드해야 한다."""
        config = {
            "risk_management": {
                "stop_loss": 0.05,
                "take_profit": 0.10,
            }
        }
        config_path = tmp_path / "config.yaml"  # type: ignore[operator]
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        rm = RiskManager(config_path=config_path)
        assert rm.config["stop_loss"] == 0.05
        assert rm.config["take_profit"] == 0.10
        assert rm.config["trailing_stop"] == 0.03  # 기본값 유지


# ---------------------------------------------------------------------------
# Stop Loss
# ---------------------------------------------------------------------------


class TestStopLoss:
    """손절 테스트."""

    def test_stop_loss_triggers_sell(self, risk_manager: RiskManager) -> None:
        """가격이 진입가 대비 3% 이상 하락하면 강제 매도해야 한다."""
        # 100에서 매수 → 96.9로 하락 (3.1% 하락, SL 트리거)
        closes = [100.0, 100.0, 96.9]
        df = _make_ohlcv(closes)
        signals = np.array([1, 0, 0])

        result = risk_manager.process_signals(df, signals)
        assert result[0] == 1  # 매수
        assert result[2] == -1  # SL 강제 매도

    def test_stop_loss_not_triggered(self, risk_manager: RiskManager) -> None:
        """가격이 SL 미만으로 하락하지 않으면 매도하지 않아야 한다."""
        # 100에서 매수 → 98.5로 하락 (1.5% 하락, SL 3% 미도달, trailing 2% 미도달)
        closes = [100.0, 100.0, 98.5]
        df = _make_ohlcv(closes)
        signals = np.array([1, 0, 0])

        result = risk_manager.process_signals(df, signals)
        assert result[0] == 1
        assert result[2] == 0  # 매도 없음

    def test_stop_loss_triggers_cooldown(self) -> None:
        """SL 발동 후 쿨다운 기간에 매수가 억제되어야 한다."""
        rm = RiskManager(config={"stop_loss": 0.03, "cooldown_after_loss": 2})
        # 매수 → SL → 쿨다운 2캔들 → 매수 가능
        closes = [100.0, 96.9, 100.0, 100.0, 100.0]
        df = _make_ohlcv(closes)
        signals = np.array([1, 0, 1, 1, 1])

        result = rm.process_signals(df, signals)
        assert result[0] == 1  # 매수
        assert result[1] == -1  # SL 매도
        assert result[2] == 0  # 쿨다운 1
        assert result[3] == 0  # 쿨다운 2
        assert result[4] == 1  # 쿨다운 종료, 매수 허용


# ---------------------------------------------------------------------------
# Take Profit
# ---------------------------------------------------------------------------


class TestTakeProfit:
    """익절 테스트."""

    def test_take_profit_triggers_sell(self, risk_manager: RiskManager) -> None:
        """가격이 진입가 대비 4% 이상 상승하면 강제 매도해야 한다."""
        # 100에서 매수 → 104.1로 상승 (4.1% 상승, TP 트리거)
        closes = [100.0, 102.0, 104.1]
        df = _make_ohlcv(closes)
        signals = np.array([1, 0, 0])

        result = risk_manager.process_signals(df, signals)
        assert result[0] == 1
        assert result[2] == -1  # TP 강제 매도

    def test_take_profit_not_triggered(self, risk_manager: RiskManager) -> None:
        """가격이 TP 미만으로 상승하면 매도하지 않아야 한다."""
        # 100에서 매수 → 103.5로 상승 (3.5%, TP 4% 미도달)
        closes = [100.0, 102.0, 103.5]
        df = _make_ohlcv(closes)
        signals = np.array([1, 0, 0])

        result = risk_manager.process_signals(df, signals)
        assert result[0] == 1
        assert result[2] == 0


# ---------------------------------------------------------------------------
# Trailing Stop
# ---------------------------------------------------------------------------


class TestTrailingStop:
    """트레일링 스탑 테스트."""

    def test_trailing_stop_after_rise(self) -> None:
        """상승 후 peak 대비 2% 하락하면 트레일링 스탑이 발동해야 한다."""
        rm = RiskManager(config={"trailing_stop": 0.02, "stop_loss": 0.10, "take_profit": 0.50})
        # 100 매수 → 110 peak → 107.7 (110 대비 2.09% 하락)
        closes = [100.0, 110.0, 107.7]
        df = _make_ohlcv(closes)
        signals = np.array([1, 0, 0])

        result = rm.process_signals(df, signals)
        assert result[0] == 1
        assert result[2] == -1  # 트레일링 스탑

    def test_trailing_stop_not_triggered_on_rise(self) -> None:
        """지속 상승 시 트레일링 스탑이 발동하지 않아야 한다."""
        rm = RiskManager(config={"trailing_stop": 0.02, "stop_loss": 0.10, "take_profit": 0.50})
        # 100 매수 → 105 → 110 (계속 상승)
        closes = [100.0, 105.0, 110.0]
        df = _make_ohlcv(closes)
        signals = np.array([1, 0, 0])

        result = rm.process_signals(df, signals)
        assert result[0] == 1
        assert result[1] == 0
        assert result[2] == 0


# ---------------------------------------------------------------------------
# Max Drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    """최대 낙폭 테스트."""

    def test_max_drawdown_suppresses_buy(self) -> None:
        """MDD 초과 시 매수 신호가 억제되어야 한다."""
        rm = RiskManager(config={"max_drawdown": 0.20, "stop_loss": 0.25})
        # 100 매수 → 74 하락 (26% 하락) → SL 발동 → equity DD > 20% → 매수 억제
        closes = [100.0, 74.0, 100.0]
        df = _make_ohlcv(closes)
        signals = np.array([1, 0, 1])

        result = rm.process_signals(df, signals)
        assert result[0] == 1
        assert result[1] == -1  # SL 매도
        # equity = 10000 * (74/100) = 7400, peak = 10000, DD = 26% > 20%
        assert result[2] == 0  # 매수 억제

    def test_max_drawdown_allows_buy_within_limit(self) -> None:
        """MDD 이내면 매수가 허용되어야 한다."""
        rm = RiskManager(
            config={
                "max_drawdown": 0.20,
                "stop_loss": 0.10,
                "cooldown_after_loss": 0,
            }
        )
        # 100 매수 → 91 하락 (9% 하락, SL 발동) → equity DD=9% < 20% → 매수 허용
        closes = [100.0, 91.0, 100.0]
        df = _make_ohlcv(closes)
        signals = np.array([1, 0, 1])

        result = rm.process_signals(df, signals)
        assert result[0] == 1
        assert result[1] == -1  # SL 매도
        assert result[2] == 1  # 매수 허용 (DD < 20%)


# ---------------------------------------------------------------------------
# Daily Limit
# ---------------------------------------------------------------------------


class TestDailyLimit:
    """일일 거래 한도 테스트."""

    def test_daily_limit_suppresses_buy(self) -> None:
        """일일 한도 초과 시 매수 신호가 억제되어야 한다."""
        rm = RiskManager(
            config={
                "max_daily_trades": 2,
                "stop_loss": 0.10,
                "take_profit": 0.50,
                "trailing_stop": 0.50,
                "cooldown_after_loss": 0,
            }
        )
        # 같은 날: 매수(1) → 매도(2) → 매수(3=한도초과)
        closes = [100.0, 100.0, 100.0, 100.0, 100.0]
        df = _make_ohlcv(closes)
        signals = np.array([1, -1, 1, -1, 1])

        result = rm.process_signals(df, signals)
        assert result[0] == 1  # 매수 (거래 1)
        assert result[1] == -1  # 매도 (거래 2)
        assert result[2] == 0  # 한도 초과 → 매수 억제

    def test_daily_limit_resets_on_new_day(self) -> None:
        """날짜 변경 시 일일 카운터가 리셋되어야 한다."""
        rm = RiskManager(
            config={
                "max_daily_trades": 2,
                "stop_loss": 0.10,
                "take_profit": 0.50,
                "trailing_stop": 0.50,
                "cooldown_after_loss": 0,
            }
        )
        dates = [
            "2024-01-01 10:00",
            "2024-01-01 11:00",
            "2024-01-01 12:00",
            "2024-01-02 10:00",  # 날짜 변경
            "2024-01-02 11:00",
        ]
        closes = [100.0, 100.0, 100.0, 100.0, 100.0]
        df = _make_multiday_ohlcv(closes, dates)
        signals = np.array([1, -1, 1, 1, -1])

        result = rm.process_signals(df, signals)
        assert result[0] == 1  # 매수 (1/1 거래 1)
        assert result[1] == -1  # 매도 (1/1 거래 2)
        assert result[2] == 0  # 1/1 한도 초과
        assert result[3] == 1  # 1/2 날짜 변경, 매수 허용
        assert result[4] == -1  # 1/2 매도


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    """쿨다운 테스트."""

    def test_cooldown_suppresses_buy(self) -> None:
        """쿨다운 기간 중 매수가 억제되어야 한다."""
        rm = RiskManager(config={"stop_loss": 0.03, "cooldown_after_loss": 3})
        closes = [100.0, 96.9, 100.0, 100.0, 100.0, 100.0]
        df = _make_ohlcv(closes)
        signals = np.array([1, 0, 1, 1, 1, 1])

        result = rm.process_signals(df, signals)
        assert result[0] == 1  # 매수
        assert result[1] == -1  # SL 매도
        assert result[2] == 0  # 쿨다운 1
        assert result[3] == 0  # 쿨다운 2
        assert result[4] == 0  # 쿨다운 3
        assert result[5] == 1  # 쿨다운 종료

    def test_cooldown_not_applied_on_normal_sell(self, risk_manager: RiskManager) -> None:
        """일반 매도 후에는 쿨다운이 적용되지 않아야 한다."""
        closes = [100.0, 100.0, 100.0]
        df = _make_ohlcv(closes)
        signals = np.array([1, -1, 1])

        result = risk_manager.process_signals(df, signals)
        assert result[0] == 1
        assert result[1] == -1
        assert result[2] == 1  # 쿨다운 없음


# ---------------------------------------------------------------------------
# Process Signals (통합)
# ---------------------------------------------------------------------------


class TestProcessSignals:
    """신호 전처리 통합 테스트."""

    def test_no_risk_signals_pass_through(self, risk_manager: RiskManager) -> None:
        """리스크 조건이 없으면 원본 신호가 그대로 통과해야 한다."""
        closes = [100.0, 101.0, 102.0, 101.5, 101.0]
        df = _make_ohlcv(closes)
        signals = np.array([1, 0, 0, 0, -1])

        result = risk_manager.process_signals(df, signals)
        assert result[0] == 1
        assert result[4] == -1
        assert all(result[1:4] == 0)

    def test_empty_signals(self, risk_manager: RiskManager) -> None:
        """빈 신호 배열을 처리할 수 있어야 한다."""
        df = _make_ohlcv([])
        signals = np.array([], dtype=int)

        result = risk_manager.process_signals(df, signals)
        assert len(result) == 0

    def test_validation_close_column(self, risk_manager: RiskManager) -> None:
        """close 컬럼이 없으면 ValueError가 발생해야 한다."""
        df = pd.DataFrame({"open": [100.0]}, index=pd.date_range("2024-01-01", periods=1))
        signals = np.array([1])

        with pytest.raises(ValueError, match="close"):
            risk_manager.process_signals(df, signals)

    def test_validation_length_mismatch(self, risk_manager: RiskManager) -> None:
        """signals 길이가 df와 다르면 ValueError가 발생해야 한다."""
        df = _make_ohlcv([100.0, 101.0])
        signals = np.array([1])

        with pytest.raises(ValueError, match="불일치"):
            risk_manager.process_signals(df, signals)


class TestAtrDynamicStops:
    """ATR 기반 동적 스탑 테스트."""

    def test_atr_stops_use_atr_values(self) -> None:
        """use_atr_stops=True이면 ATR 기반으로 스탑이 계산되어야 한다."""
        rm = RiskManager(config={
            "use_atr_stops": True,
            "atr_stop_multiplier": 2.0,
            "stop_loss": 0.03,
            "take_profit": 0.04,
            "trailing_stop": 0.03,
        })
        # ATR=1000 → atr_pct=1000/100000=0.01 → SL=0.02, TP=0.04, TS=0.02
        closes = [100000.0, 100000.0, 97500.0]  # -2.5% drop
        df = _make_ohlcv(closes)
        df["atr_14"] = [1000.0, 1000.0, 1000.0]
        signals = np.array([1, 0, 0])

        result = rm.process_signals(df, signals)
        # ATR SL = 0.01*2 = 0.02, drop is 2.5% > 2% → 스탑로스 발동
        assert result[0] == 1
        assert result[2] == -1

    def test_atr_stops_no_atr_column_falls_back(self) -> None:
        """atr_14 컬럼이 없으면 고정 스탑으로 폴백해야 한다."""
        rm = RiskManager(config={
            "use_atr_stops": True,
            "atr_stop_multiplier": 2.0,
            "stop_loss": 0.03,
            "take_profit": 0.04,
            "trailing_stop": 0.03,
        })
        closes = [100.0, 101.0, 102.0]
        df = _make_ohlcv(closes)  # atr_14 없음
        signals = np.array([1, 0, -1])

        result = rm.process_signals(df, signals)
        assert result[0] == 1  # 고정 스탑으로 정상 작동

    def test_atr_stops_disabled_uses_fixed(self) -> None:
        """use_atr_stops=False면 고정 스탑을 사용해야 한다."""
        rm = RiskManager(config={
            "use_atr_stops": False,
            "stop_loss": 0.03,
            "take_profit": 0.04,
            "trailing_stop": 0.03,
        })
        closes = [100.0, 96.5]  # -3.5% drop > SL 3%
        df = _make_ohlcv(closes)
        signals = np.array([1, 0])

        result = rm.process_signals(df, signals)
        assert result[0] == 1
        assert result[1] == -1  # 고정 SL 3% 발동


class TestDynamicPositionSizer:
    """DynamicPositionSizer 테스트."""

    def test_default_params(self) -> None:
        """기본 파라미터가 적용되어야 한다."""
        sizer = DynamicPositionSizer()
        assert sizer.config["method"] == "volatility"
        assert sizer.config["target_volatility"] == 0.02
        assert sizer.config["kelly_fraction"] == 0.25

    def test_volatility_high_atr_reduces_size(self) -> None:
        """높은 ATR(높은 변동성)이면 포지션이 축소되어야 한다."""
        sizer = DynamicPositionSizer(config={
            "method": "volatility",
            "base_size": 0.50,
            "target_volatility": 0.02,
        })
        # ATR=2000, close=100000 → current_vol=0.02 → size=0.50 (정확히 target)
        size_normal = sizer.compute(atr=2000.0, close=100000.0)
        # ATR=4000 → current_vol=0.04 → size=0.25 (절반)
        size_high = sizer.compute(atr=4000.0, close=100000.0)
        assert size_high < size_normal

    def test_volatility_low_atr_increases_size(self) -> None:
        """낮은 ATR(낮은 변동성)이면 포지션이 확대되어야 한다."""
        sizer = DynamicPositionSizer(config={
            "method": "volatility",
            "base_size": 0.50,
            "target_volatility": 0.02,
        })
        # ATR=1000 → current_vol=0.01 → size=1.0 → capped at max_size
        size = sizer.compute(atr=1000.0, close=100000.0)
        assert size == 0.95  # max_size

    def test_kelly_positive_edge(self) -> None:
        """양의 엣지가 있으면 Kelly가 양수를 반환해야 한다."""
        sizer = DynamicPositionSizer(config={
            "method": "kelly",
            "kelly_fraction": 0.25,
        })
        # WR=0.55, avg_win=0.04, avg_loss=0.03
        # kelly_f = (0.55*0.04 - 0.45*0.03) / 0.04 = (0.022-0.0135)/0.04 = 0.2125
        # size = 0.2125 * 0.25 = 0.053 → min_size 0.10
        size = sizer.compute(atr=0, close=0, win_rate=0.55, avg_win=0.04, avg_loss=0.03)
        assert size == 0.10  # min_size

    def test_kelly_negative_edge_returns_min(self) -> None:
        """음의 엣지면 min_size를 반환해야 한다."""
        sizer = DynamicPositionSizer(config={
            "method": "kelly",
            "kelly_fraction": 0.25,
        })
        # WR=0.30 → 음의 기대값
        size = sizer.compute(atr=0, close=0, win_rate=0.30, avg_win=0.02, avg_loss=0.04)
        assert size == 0.10  # min_size

    def test_size_clamped_to_range(self) -> None:
        """포지션 크기가 min/max 범위 내에 있어야 한다."""
        sizer = DynamicPositionSizer(config={
            "method": "volatility",
            "base_size": 0.50,
            "target_volatility": 0.02,
            "min_size": 0.15,
            "max_size": 0.80,
        })
        # 극단적 ATR → 매우 작은 size
        size_tiny = sizer.compute(atr=10000.0, close=100000.0)
        assert size_tiny >= 0.15
        # 극단적으로 낮은 ATR → 매우 큰 size
        size_huge = sizer.compute(atr=100.0, close=100000.0)
        assert size_huge <= 0.80
