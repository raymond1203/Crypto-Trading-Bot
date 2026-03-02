"""RiskManager 단위 테스트.

합성 OHLCV 데이터로 손절/익절/트레일링 스탑/MDD 제한/일일 한도/쿨다운
리스크 규칙의 신호 전처리 동작을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.strategy.risk import RiskManager

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
        assert rm.config["take_profit"] == 0.06
        assert rm.config["trailing_stop"] == 0.02
        assert rm.config["max_drawdown"] == 0.20
        assert rm.config["max_daily_trades"] == 10
        assert rm.config["cooldown_after_loss"] == 2

    def test_custom_config(self) -> None:
        """커스텀 설정이 기본값을 오버라이드해야 한다."""
        rm = RiskManager(config={"stop_loss": 0.05, "max_daily_trades": 5})
        assert rm.config["stop_loss"] == 0.05
        assert rm.config["max_daily_trades"] == 5
        assert rm.config["take_profit"] == 0.06  # 기본값 유지

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
        assert rm.config["trailing_stop"] == 0.02  # 기본값 유지


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
        """가격이 진입가 대비 6% 이상 상승하면 강제 매도해야 한다."""
        # 100에서 매수 → 106.1로 상승 (6.1% 상승, TP 트리거)
        closes = [100.0, 103.0, 106.1]
        df = _make_ohlcv(closes)
        signals = np.array([1, 0, 0])

        result = risk_manager.process_signals(df, signals)
        assert result[0] == 1
        assert result[2] == -1  # TP 강제 매도

    def test_take_profit_not_triggered(self, risk_manager: RiskManager) -> None:
        """가격이 TP 미만으로 상승하면 매도하지 않아야 한다."""
        # 100에서 매수 → 105로 상승 (5%, TP 미도달)
        closes = [100.0, 103.0, 105.0]
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
