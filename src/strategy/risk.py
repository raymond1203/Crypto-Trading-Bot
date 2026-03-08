"""리스크 관리 모듈.

원본 매매 신호를 입력받아 손절/익절/트레일링 스탑/MDD 제한/쿨다운 등
리스크 규칙을 적용한 수정 신호를 반환한다.
BacktestEngine 수정 없이 신호 전처리 방식으로 동작한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger


class RiskManager:
    """리스크 관리 신호 전처리기.

    매매 신호를 bar-by-bar 시뮬레이션하며 포지션 상태를 추적하고,
    리스크 규칙에 따라 강제 매도 삽입 또는 매수 억제를 수행한다.

    Attributes:
        config: 리스크 관리 파라미터 딕셔너리.
    """

    DEFAULT_PARAMS: dict = {
        "stop_loss": 0.03,
        "take_profit": 0.04,
        "trailing_stop": 0.03,
        "max_drawdown": 0.20,
        "max_position_size": 0.95,
        "max_daily_trades": 10,
        "cooldown_after_loss": 2,
        "use_atr_stops": False,
        "atr_stop_multiplier": 2.0,
    }

    def __init__(
        self,
        config: dict | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """리스크 관리기를 초기화한다.

        Args:
            config: 리스크 파라미터 딕셔너리.
            config_path: YAML 설정 파일 경로 (config 미지정 시 사용).
        """
        if config is None and config_path is not None:
            full_config = self._load_yaml(config_path)
            config = full_config.get("risk_management", {})

        config = config or {}
        self.config: dict = {**self.DEFAULT_PARAMS, **config}
        logger.info(f"RiskManager 초기화: {self.config}")

    @staticmethod
    def _load_yaml(path: str | Path) -> dict:
        """YAML 설정 파일을 로드한다.

        Args:
            path: YAML 파일 경로.

        Returns:
            설정 딕셔너리.
        """
        with open(path) as f:
            return yaml.safe_load(f)

    def process_signals(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,
        initial_capital: float = 10000.0,
    ) -> np.ndarray:
        """리스크 규칙을 적용하여 매매 신호를 수정한다.

        bar-by-bar 시뮬레이션으로 포지션 상태를 추적하며,
        손절/익절/트레일링 스탑/MDD 제한/일일 한도/쿨다운을 적용한다.

        Args:
            df: OHLCV DataFrame (close 컬럼 필수).
            signals: 원본 매매 신호 (1=매수, 0=관망, -1=매도).
            initial_capital: 초기 자본 (MDD 계산용).

        Returns:
            리스크 규칙이 적용된 수정 신호 배열.

        Raises:
            ValueError: df에 close 컬럼이 없을 때.
            ValueError: signals 길이가 df와 불일치할 때.
        """
        if "close" not in df.columns:
            raise ValueError("DataFrame에 'close' 컬럼이 필요합니다.")

        if len(signals) != len(df):
            raise ValueError(f"signals 길이({len(signals)})가 df 길이({len(df)})와 불일치합니다.")

        close = df["close"].values
        output = np.zeros(len(df), dtype=int)

        use_atr = self.config["use_atr_stops"]
        atr_mult = self.config["atr_stop_multiplier"]
        atr_values: np.ndarray | None = None
        if use_atr and "atr_14" in df.columns:
            atr_values = df["atr_14"].values
        elif use_atr:
            logger.warning("use_atr_stops=True이지만 atr_14 컬럼 없음, 고정 스탑 사용")
            use_atr = False

        stop_loss = self.config["stop_loss"]
        take_profit = self.config["take_profit"]
        trailing_stop = self.config["trailing_stop"]
        max_drawdown = self.config["max_drawdown"]
        max_daily_trades = self.config["max_daily_trades"]
        cooldown_after_loss = self.config["cooldown_after_loss"]
        position_ratio = self.config["max_position_size"]

        in_position = False
        entry_price = 0.0
        peak_price = 0.0
        equity = initial_capital
        peak_equity = initial_capital
        cooldown_remaining = 0
        daily_trades = 0
        current_date: object = None

        dates = df.index.date if hasattr(df.index, "date") else None

        for i in range(len(df)):
            signal = int(signals[i])
            price = close[i]

            # 날짜 변경 시 일일 거래 카운터 리셋
            if dates is not None:
                bar_date = dates[i]
                if bar_date != current_date:
                    current_date = bar_date
                    daily_trades = 0

            if in_position:
                # SL/TP/Trailing Stop 체크
                if self._check_stop_loss(entry_price, price, stop_loss):
                    output[i] = -1
                    in_position = False
                    cooldown_remaining = cooldown_after_loss
                    pnl_pct = price / entry_price - 1.0
                    equity = equity * (1.0 + position_ratio * pnl_pct)
                    daily_trades += 1
                elif self._check_take_profit(entry_price, price, take_profit):
                    output[i] = -1
                    in_position = False
                    pnl_pct = price / entry_price - 1.0
                    equity = equity * (1.0 + position_ratio * pnl_pct)
                    daily_trades += 1
                elif self._check_trailing_stop(peak_price, price, trailing_stop):
                    output[i] = -1
                    in_position = False
                    cooldown_remaining = cooldown_after_loss
                    pnl_pct = price / entry_price - 1.0
                    equity = equity * (1.0 + position_ratio * pnl_pct)
                    daily_trades += 1
                else:
                    peak_price = max(peak_price, price)
                    if signal == -1:
                        output[i] = -1
                        in_position = False
                        pnl_pct = price / entry_price - 1.0
                        equity = equity * (1.0 + position_ratio * pnl_pct)
                        daily_trades += 1
                    else:
                        output[i] = 0
            else:
                if signal == 1:
                    if (
                        self._check_max_drawdown(equity, peak_equity, max_drawdown)
                        or self._check_daily_limit(daily_trades, max_daily_trades)
                        or cooldown_remaining > 0
                    ):
                        output[i] = 0
                    else:
                        output[i] = 1
                        entry_price = price
                        peak_price = price
                        in_position = True
                        daily_trades += 1
                        # ATR 기반 동적 스탑 설정
                        if use_atr and atr_values is not None:
                            atr_pct = atr_values[i] / price
                            stop_loss = max(atr_pct * atr_mult, 0.01)
                            take_profit = max(atr_pct * atr_mult * 2, 0.02)
                            trailing_stop = max(atr_pct * atr_mult, 0.01)
                else:
                    output[i] = 0

                if cooldown_remaining > 0:
                    cooldown_remaining -= 1

            # peak_equity 갱신
            if not in_position:
                peak_equity = max(peak_equity, equity)

        return output

    @staticmethod
    def _check_stop_loss(entry_price: float, current_price: float, threshold: float) -> bool:
        """손절 조건을 확인한다.

        Args:
            entry_price: 진입 가격.
            current_price: 현재 가격.
            threshold: 손절 비율 (e.g., 0.03 = 3%).

        Returns:
            손절 조건 충족 시 True.
        """
        return (entry_price - current_price) / entry_price >= threshold

    @staticmethod
    def _check_take_profit(entry_price: float, current_price: float, threshold: float) -> bool:
        """익절 조건을 확인한다.

        Args:
            entry_price: 진입 가격.
            current_price: 현재 가격.
            threshold: 익절 비율 (e.g., 0.06 = 6%).

        Returns:
            익절 조건 충족 시 True.
        """
        return (current_price - entry_price) / entry_price >= threshold

    @staticmethod
    def _check_trailing_stop(peak_price: float, current_price: float, threshold: float) -> bool:
        """트레일링 스탑 조건을 확인한다.

        Args:
            peak_price: 진입 후 최고 가격.
            current_price: 현재 가격.
            threshold: 트레일링 비율 (e.g., 0.02 = 2%).

        Returns:
            트레일링 스탑 조건 충족 시 True.
        """
        return (peak_price - current_price) / peak_price >= threshold

    @staticmethod
    def _check_max_drawdown(equity: float, peak_equity: float, threshold: float) -> bool:
        """최대 낙폭 초과 여부를 확인한다.

        Args:
            equity: 현재 자산 가치.
            peak_equity: 최고 자산 가치.
            threshold: 최대 낙폭 비율 (e.g., 0.20 = 20%).

        Returns:
            MDD 초과 시 True.
        """
        if peak_equity == 0:
            return False
        drawdown = (peak_equity - equity) / peak_equity
        return drawdown >= threshold

    @staticmethod
    def _check_daily_limit(daily_trades: int, max_daily: int) -> bool:
        """일일 거래 한도 초과 여부를 확인한다.

        Args:
            daily_trades: 당일 거래 횟수.
            max_daily: 일일 최대 거래 수.

        Returns:
            한도 초과 시 True.
        """
        return daily_trades >= max_daily
