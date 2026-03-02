"""백테스트 성능 지표 계산 모듈.

Sharpe Ratio, MDD, Win Rate, Profit Factor, Calmar, Sortino 등
트레이딩 전략 평가 지표를 계산한다.
"""

import numpy as np
import pandas as pd
from loguru import logger

# 1시간 캔들 기준 1년 = 24 * 365 = 8760
_DEFAULT_PERIODS = 8760


class BacktestMetrics:
    """백테스트 성능 지표 계산기.

    모든 메서드는 정적 메서드로, 인스턴스 생성 없이 호출한다.
    """

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods: int = _DEFAULT_PERIODS,
    ) -> float:
        """Sharpe Ratio를 계산한다.

        Args:
            returns: 수익률 시리즈 (각 기간별).
            risk_free_rate: 연율 무위험 수익률.
            periods: 연환산 기간 (8760 = 1h 캔들 기준 1년).

        Returns:
            Sharpe Ratio 값. 표준편차가 0이면 0.0.
        """
        excess = returns - risk_free_rate / periods
        std = excess.std()
        if std == 0 or np.isnan(std):
            return 0.0
        return float((excess.mean() / std) * np.sqrt(periods))

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods: int = _DEFAULT_PERIODS,
    ) -> float:
        """Sortino Ratio를 계산한다.

        Args:
            returns: 수익률 시리즈.
            risk_free_rate: 연율 무위험 수익률.
            periods: 연환산 기간.

        Returns:
            Sortino Ratio 값. 하방 편차가 0이면 0.0.
        """
        excess = returns - risk_free_rate / periods
        downside = returns[returns < 0]
        if len(downside) == 0:
            return 0.0
        downside_std = downside.std()
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
        return float((excess.mean() / downside_std) * np.sqrt(periods))

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """Maximum Drawdown을 계산한다.

        Args:
            equity_curve: 자산 가치 시리즈.

        Returns:
            MDD 값 (음수, e.g., -0.20 = -20%). 단조 증가 시 0.0.
        """
        if len(equity_curve) < 2:
            return 0.0
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        return float(drawdown.min())

    @staticmethod
    def win_rate(trades: pd.DataFrame) -> float:
        """승률을 계산한다.

        Args:
            trades: 거래 DataFrame (pnl 컬럼 필수).

        Returns:
            승률 (0.0~1.0). 거래가 없으면 0.0.
        """
        if trades.empty or "pnl" not in trades.columns:
            return 0.0
        total = len(trades)
        winning = (trades["pnl"] > 0).sum()
        return float(winning / total)

    @staticmethod
    def profit_factor(trades: pd.DataFrame) -> float:
        """Profit Factor를 계산한다.

        Args:
            trades: 거래 DataFrame (pnl_abs 컬럼 필수).

        Returns:
            총이익/총손실. 손실이 없으면 inf. 이익이 없으면 0.0.
        """
        if trades.empty or "pnl_abs" not in trades.columns:
            return 0.0
        gross_profit = trades.loc[trades["pnl_abs"] > 0, "pnl_abs"].sum()
        gross_loss = abs(trades.loc[trades["pnl_abs"] < 0, "pnl_abs"].sum())
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    @staticmethod
    def calmar_ratio(
        returns: pd.Series,
        equity_curve: pd.Series,
        periods: int = _DEFAULT_PERIODS,
    ) -> float:
        """Calmar Ratio를 계산한다.

        Args:
            returns: 수익률 시리즈.
            equity_curve: 자산 가치 시리즈.
            periods: 연환산 기간.

        Returns:
            연수익률/|MDD|. MDD가 0이면 0.0.
        """
        annual_return = float(returns.mean() * periods)
        mdd = abs(BacktestMetrics.max_drawdown(equity_curve))
        if mdd == 0:
            return 0.0
        return float(annual_return / mdd)

    @staticmethod
    def buy_and_hold_return(prices: pd.Series) -> float:
        """Buy & Hold 수익률을 계산한다.

        Args:
            prices: 가격(close) 시리즈.

        Returns:
            수익률 (e.g., 0.30 = +30%).
        """
        if len(prices) < 2:
            return 0.0
        return float((prices.iloc[-1] / prices.iloc[0]) - 1)

    @staticmethod
    def generate_report(
        trades: pd.DataFrame,
        equity_curve: pd.Series,
        prices: pd.Series,
        initial_capital: float = 10000.0,
        periods: int = _DEFAULT_PERIODS,
    ) -> dict:
        """전체 성능 리포트를 생성한다.

        Args:
            trades: 거래 DataFrame.
            equity_curve: 자산 가치 시리즈.
            prices: 가격(close) 시리즈.
            initial_capital: 초기 자본.
            periods: 연환산 기간.

        Returns:
            모든 성능 지표를 포함하는 딕셔너리 (raw 숫자).
        """
        returns = equity_curve.pct_change().dropna()
        strategy_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        bh_return = BacktestMetrics.buy_and_hold_return(prices)

        winning_trades = int((trades["pnl"] > 0).sum()) if not trades.empty else 0
        losing_trades = int((trades["pnl"] < 0).sum()) if not trades.empty else 0

        report: dict = {
            "total_trades": len(trades),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": BacktestMetrics.win_rate(trades),
            "profit_factor": BacktestMetrics.profit_factor(trades),
            "sharpe_ratio": BacktestMetrics.sharpe_ratio(returns, periods=periods),
            "sortino_ratio": BacktestMetrics.sortino_ratio(returns, periods=periods),
            "max_drawdown": BacktestMetrics.max_drawdown(equity_curve),
            "calmar_ratio": BacktestMetrics.calmar_ratio(returns, equity_curve, periods=periods),
            "strategy_return": float(strategy_return),
            "buy_hold_return": float(bh_return),
            "excess_return": float(strategy_return - bh_return),
            "initial_capital": float(initial_capital),
            "final_capital": float(equity_curve.iloc[-1]),
        }

        if not trades.empty and "pnl" in trades.columns:
            winners = trades.loc[trades["pnl"] > 0, "pnl"]
            losers = trades.loc[trades["pnl"] < 0, "pnl"]
            report["best_trade"] = float(trades["pnl"].max())
            report["worst_trade"] = float(trades["pnl"].min())
            report["avg_win"] = float(winners.mean()) if len(winners) > 0 else 0.0
            report["avg_loss"] = float(losers.mean()) if len(losers) > 0 else 0.0
        else:
            report["best_trade"] = 0.0
            report["worst_trade"] = 0.0
            report["avg_win"] = 0.0
            report["avg_loss"] = 0.0

        if not trades.empty and "duration" in trades.columns:
            avg_duration = trades["duration"].mean()
            report["avg_trade_duration_hours"] = (
                float(avg_duration.total_seconds() / 3600) if pd.notna(avg_duration) else None
            )
        else:
            report["avg_trade_duration_hours"] = None

        logger.info(
            f"백테스트 리포트: {report['total_trades']}건, "
            f"승률 {report['win_rate']:.2%}, "
            f"Sharpe {report['sharpe_ratio']:.2f}, "
            f"MDD {report['max_drawdown']:.2%}"
        )

        return report
