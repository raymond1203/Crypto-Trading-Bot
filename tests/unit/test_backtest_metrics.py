"""BacktestMetrics 단위 테스트.

결정적 데이터로 각 성능 지표 계산을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import BacktestMetrics

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_equity_curve() -> pd.Series:
    """단조 증가 후 하락하는 자산 곡선."""
    values = [10000, 10200, 10500, 10300, 10100, 10400, 10600]
    dates = pd.date_range("2024-01-01", periods=len(values), freq="1h", tz="UTC")
    return pd.Series(values, index=dates, dtype=float)


@pytest.fixture
def sample_trades() -> pd.DataFrame:
    """승 2건, 패 1건의 거래 기록."""
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10"], utc=True),
            "exit_time": pd.to_datetime(["2024-01-03", "2024-01-08", "2024-01-12"], utc=True),
            "entry_price": [40000.0, 41000.0, 39000.0],
            "exit_price": [41000.0, 40500.0, 40000.0],
            "position_size": [0.24, 0.23, 0.25],
            "pnl": [0.025, -0.012195, 0.025641],
            "pnl_abs": [240.0, -115.0, 250.0],
            "side": ["long", "long", "long"],
            "duration": pd.to_timedelta(["2 days", "3 days", "2 days"]),
        }
    )


@pytest.fixture
def empty_trades() -> pd.DataFrame:
    """빈 거래 DataFrame."""
    return pd.DataFrame(
        columns=[
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "position_size",
            "pnl",
            "pnl_abs",
            "side",
            "duration",
        ]
    )


# ---------------------------------------------------------------------------
# Sharpe Ratio
# ---------------------------------------------------------------------------


class TestSharpeRatio:
    """Sharpe Ratio 계산 테스트."""

    def test_positive_returns(self) -> None:
        """양의 수익률은 양의 Sharpe를 반환해야 한다."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.03, 0.01])
        result = BacktestMetrics.sharpe_ratio(returns, periods=8760)
        assert result > 0

    def test_zero_std_returns_zero(self) -> None:
        """표준편차 0이면 0.0을 반환해야 한다."""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01])
        result = BacktestMetrics.sharpe_ratio(returns, periods=8760)
        assert result == 0.0

    def test_negative_returns(self) -> None:
        """음의 수익률은 음의 Sharpe를 반환해야 한다."""
        returns = pd.Series([-0.01, -0.02, -0.01, -0.03, -0.01])
        result = BacktestMetrics.sharpe_ratio(returns, periods=8760)
        assert result < 0

    def test_with_risk_free_rate(self) -> None:
        """무위험 수익률이 높으면 Sharpe가 감소해야 한다."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.03, 0.01])
        sharpe_zero = BacktestMetrics.sharpe_ratio(returns, risk_free_rate=0.0)
        sharpe_high = BacktestMetrics.sharpe_ratio(returns, risk_free_rate=0.1)
        assert sharpe_high < sharpe_zero

    def test_annualization_scaling(self) -> None:
        """periods가 다르면 Sharpe도 비례적으로 변해야 한다."""
        returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.01])
        sharpe_hourly = BacktestMetrics.sharpe_ratio(returns, periods=8760)
        sharpe_daily = BacktestMetrics.sharpe_ratio(returns, periods=252)
        assert sharpe_hourly > sharpe_daily


# ---------------------------------------------------------------------------
# Sortino Ratio
# ---------------------------------------------------------------------------


class TestSortinoRatio:
    """Sortino Ratio 계산 테스트."""

    def test_no_negative_returns(self) -> None:
        """음의 수익률이 없으면 0.0을 반환해야 한다."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.01])
        result = BacktestMetrics.sortino_ratio(returns)
        assert result == 0.0

    def test_mixed_returns(self) -> None:
        """양/음 혼합 수익률은 양의 Sortino를 반환해야 한다."""
        returns = pd.Series([0.03, -0.01, 0.02, -0.005, 0.015])
        result = BacktestMetrics.sortino_ratio(returns)
        assert result > 0

    def test_all_negative_returns(self) -> None:
        """전부 음의 수익률은 음의 Sortino를 반환해야 한다."""
        returns = pd.Series([-0.01, -0.02, -0.03, -0.01])
        result = BacktestMetrics.sortino_ratio(returns)
        assert result < 0


# ---------------------------------------------------------------------------
# Max Drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    """Maximum Drawdown 계산 테스트."""

    def test_monotonic_increase(self) -> None:
        """단조 증가 곡선은 MDD 0을 반환해야 한다."""
        equity = pd.Series([100.0, 110.0, 120.0, 130.0])
        result = BacktestMetrics.max_drawdown(equity)
        assert result == 0.0

    def test_known_drawdown(self) -> None:
        """[100, 120, 90, 110] → MDD = (90-120)/120 = -0.25."""
        equity = pd.Series([100.0, 120.0, 90.0, 110.0])
        result = BacktestMetrics.max_drawdown(equity)
        assert result == pytest.approx(-0.25, abs=1e-10)

    def test_single_value(self) -> None:
        """단일 값은 MDD 0.0을 반환해야 한다."""
        equity = pd.Series([100.0])
        result = BacktestMetrics.max_drawdown(equity)
        assert result == 0.0

    def test_always_non_positive(self, sample_equity_curve: pd.Series) -> None:
        """MDD는 항상 0 이하여야 한다."""
        result = BacktestMetrics.max_drawdown(sample_equity_curve)
        assert result <= 0.0


# ---------------------------------------------------------------------------
# Win Rate
# ---------------------------------------------------------------------------


class TestWinRate:
    """승률 계산 테스트."""

    def test_all_winners(self) -> None:
        """전부 승리하면 1.0을 반환해야 한다."""
        trades = pd.DataFrame({"pnl": [0.01, 0.02, 0.03]})
        assert BacktestMetrics.win_rate(trades) == 1.0

    def test_all_losers(self) -> None:
        """전부 패배하면 0.0을 반환해야 한다."""
        trades = pd.DataFrame({"pnl": [-0.01, -0.02, -0.03]})
        assert BacktestMetrics.win_rate(trades) == 0.0

    def test_mixed(self) -> None:
        """3승 2패면 0.6을 반환해야 한다."""
        trades = pd.DataFrame({"pnl": [0.01, -0.01, 0.02, -0.02, 0.03]})
        assert BacktestMetrics.win_rate(trades) == pytest.approx(0.6)

    def test_empty_trades(self, empty_trades: pd.DataFrame) -> None:
        """빈 거래는 0.0을 반환해야 한다."""
        assert BacktestMetrics.win_rate(empty_trades) == 0.0

    def test_known_value(self, sample_trades: pd.DataFrame) -> None:
        """승 2건, 패 1건 → 2/3."""
        result = BacktestMetrics.win_rate(sample_trades)
        assert result == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# Profit Factor
# ---------------------------------------------------------------------------


class TestProfitFactor:
    """Profit Factor 계산 테스트."""

    def test_no_losses(self) -> None:
        """손실이 없으면 inf를 반환해야 한다."""
        trades = pd.DataFrame({"pnl_abs": [100.0, 200.0, 50.0]})
        assert BacktestMetrics.profit_factor(trades) == float("inf")

    def test_no_profits(self) -> None:
        """이익이 없으면 0.0을 반환해야 한다."""
        trades = pd.DataFrame({"pnl_abs": [-100.0, -200.0]})
        assert BacktestMetrics.profit_factor(trades) == 0.0

    def test_known_value(self) -> None:
        """총이익 300 / 총손실 150 = 2.0."""
        trades = pd.DataFrame({"pnl_abs": [200.0, -100.0, 100.0, -50.0]})
        result = BacktestMetrics.profit_factor(trades)
        assert result == pytest.approx(2.0)

    def test_empty_trades(self, empty_trades: pd.DataFrame) -> None:
        """빈 거래는 0.0을 반환해야 한다."""
        assert BacktestMetrics.profit_factor(empty_trades) == 0.0

    def test_sample_trades(self, sample_trades: pd.DataFrame) -> None:
        """승 490, 패 115 → 490/115."""
        result = BacktestMetrics.profit_factor(sample_trades)
        expected = 490.0 / 115.0
        assert result == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# Calmar Ratio
# ---------------------------------------------------------------------------


class TestCalmarRatio:
    """Calmar Ratio 계산 테스트."""

    def test_no_drawdown(self) -> None:
        """MDD가 0이면 0.0을 반환해야 한다."""
        equity = pd.Series([100.0, 110.0, 120.0, 130.0])
        returns = equity.pct_change().dropna()
        result = BacktestMetrics.calmar_ratio(returns, equity)
        assert result == 0.0

    def test_known_value(self) -> None:
        """알려진 연수익률과 MDD로 계산해야 한다."""
        equity = pd.Series([100.0, 120.0, 90.0, 110.0])
        returns = equity.pct_change().dropna()
        mdd = abs(BacktestMetrics.max_drawdown(equity))  # 0.25
        annual_return = returns.mean() * 8760
        expected = annual_return / mdd
        result = BacktestMetrics.calmar_ratio(returns, equity)
        assert result == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Buy & Hold Return
# ---------------------------------------------------------------------------


class TestBuyAndHoldReturn:
    """Buy & Hold 수익률 계산 테스트."""

    def test_positive_return(self) -> None:
        """가격 상승 시 양의 수익률을 반환해야 한다."""
        prices = pd.Series([100.0, 110.0, 120.0])
        assert BacktestMetrics.buy_and_hold_return(prices) == pytest.approx(0.2)

    def test_negative_return(self) -> None:
        """가격 하락 시 음의 수익률을 반환해야 한다."""
        prices = pd.Series([100.0, 90.0, 80.0])
        assert BacktestMetrics.buy_and_hold_return(prices) == pytest.approx(-0.2)

    def test_flat(self) -> None:
        """가격 변화 없으면 0.0을 반환해야 한다."""
        prices = pd.Series([100.0, 100.0, 100.0])
        assert BacktestMetrics.buy_and_hold_return(prices) == 0.0

    def test_single_value(self) -> None:
        """단일 값은 0.0을 반환해야 한다."""
        prices = pd.Series([100.0])
        assert BacktestMetrics.buy_and_hold_return(prices) == 0.0


# ---------------------------------------------------------------------------
# Generate Report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """generate_report 통합 테스트."""

    def test_report_keys(self, sample_trades: pd.DataFrame, sample_equity_curve: pd.Series) -> None:
        """리포트에 필요한 모든 키가 있어야 한다."""
        prices = pd.Series(
            np.linspace(40000, 42000, len(sample_equity_curve)),
            index=sample_equity_curve.index,
        )
        report = BacktestMetrics.generate_report(sample_trades, sample_equity_curve, prices)
        expected_keys = {
            "total_trades",
            "winning_trades",
            "losing_trades",
            "win_rate",
            "profit_factor",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "strategy_return",
            "buy_hold_return",
            "excess_return",
            "initial_capital",
            "final_capital",
            "best_trade",
            "worst_trade",
            "avg_win",
            "avg_loss",
            "avg_trade_duration_hours",
        }
        assert expected_keys.issubset(report.keys())

    def test_report_numeric_values(self, sample_trades: pd.DataFrame, sample_equity_curve: pd.Series) -> None:
        """리포트 값은 숫자(또는 None)이어야 한다."""
        prices = pd.Series(
            np.linspace(40000, 42000, len(sample_equity_curve)),
            index=sample_equity_curve.index,
        )
        report = BacktestMetrics.generate_report(sample_trades, sample_equity_curve, prices)
        for key, value in report.items():
            assert isinstance(value, (int, float, type(None))), f"{key}의 타입이 {type(value)}입니다"

    def test_with_empty_trades(self, empty_trades: pd.DataFrame, sample_equity_curve: pd.Series) -> None:
        """빈 거래로도 리포트가 생성되어야 한다."""
        prices = pd.Series(
            np.linspace(40000, 42000, len(sample_equity_curve)),
            index=sample_equity_curve.index,
        )
        report = BacktestMetrics.generate_report(empty_trades, sample_equity_curve, prices)
        assert report["total_trades"] == 0
        assert report["win_rate"] == 0.0
        assert report["best_trade"] == 0.0
        assert report["worst_trade"] == 0.0
