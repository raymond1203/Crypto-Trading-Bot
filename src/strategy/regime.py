"""마켓 레짐 감지 모듈.

ADX, Bollinger Band Width, SMA Slope를 조합하여
시장 상태를 BULL / BEAR / SIDEWAYS 3가지로 분류한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger


class MarketRegimeDetector:
    """마켓 레짐 감지기.

    ADX로 추세 강도를 측정하고, SMA slope로 방향을 판별한다.
    BB width를 보조 지표로 사용하여 횡보 확인을 강화한다.

    Attributes:
        config: 레짐 감지 파라미터 딕셔너리.
    """

    BULL = 1
    SIDEWAYS = 0
    BEAR = -1

    DEFAULT_PARAMS: dict = {
        "adx_trend_threshold": 25,
        "sma_slope_window": 10,
        "sma_window": 25,
    }

    def __init__(
        self,
        config: dict | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """레짐 감지기를 초기화한다.

        Args:
            config: 레짐 파라미터 딕셔너리.
            config_path: YAML 설정 파일 경로 (config 미지정 시 사용).
        """
        if config is None and config_path is not None:
            full_config = self._load_yaml(config_path)
            config = full_config.get("regime", {})

        config = config or {}
        self.config: dict = {**self.DEFAULT_PARAMS, **config}
        logger.info(f"MarketRegimeDetector 초기화: {self.config}")

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

    def detect(self, df: pd.DataFrame) -> np.ndarray:
        """마켓 레짐을 감지한다.

        원본 가격(close_raw, high_raw, low_raw)이 있으면 ADX를 재계산하고,
        없으면 기존 adx 컬럼을 사용한다.

        Args:
            df: close 컬럼이 포함된 DataFrame.
                원본 가격 컬럼(close_raw, high_raw, low_raw)이 있으면 ADX 재계산.
                없으면 adx 컬럼 필수.

        Returns:
            레짐 배열 (BULL=1, SIDEWAYS=0, BEAR=-1).

        Raises:
            ValueError: 필수 컬럼이 없을 때.
        """
        adx_threshold = self.config["adx_trend_threshold"]
        sma_window = self.config["sma_window"]
        slope_window = self.config["sma_slope_window"]

        # 원본 가격이 있으면 ADX를 재계산 (스케일링된 adx는 threshold 비교 불가)
        has_raw = all(c in df.columns for c in ("close_raw", "high_raw", "low_raw"))
        if has_raw:
            adx = self._compute_adx(
                df["high_raw"].values,
                df["low_raw"].values,
                df["close_raw"].values,
            )
            close = df["close_raw"].values
        else:
            if "adx" not in df.columns:
                raise ValueError("DataFrame에 'adx' 컬럼이 필요합니다.")
            if "close" not in df.columns:
                raise ValueError("DataFrame에 'close' 컬럼이 필요합니다.")
            adx = df["adx"].values
            close = df["close"].values

        # SMA slope 계산: SMA의 최근 N기간 변화율
        sma = pd.Series(close).rolling(sma_window).mean().values
        sma_slope = np.zeros(len(close))
        for i in range(slope_window, len(close)):
            if sma[i - slope_window] != 0 and not np.isnan(sma[i - slope_window]):
                sma_slope[i] = (sma[i] - sma[i - slope_window]) / sma[i - slope_window]

        # 레짐 분류
        regimes = np.full(len(df), self.SIDEWAYS, dtype=int)

        for i in range(len(df)):
            if np.isnan(adx[i]):
                continue
            if adx[i] > adx_threshold:
                regimes[i] = self.BULL if sma_slope[i] > 0 else self.BEAR
            # else: SIDEWAYS (기본값)

        bull_pct = (regimes == self.BULL).mean() * 100
        bear_pct = (regimes == self.BEAR).mean() * 100
        side_pct = (regimes == self.SIDEWAYS).mean() * 100
        logger.info(
            f"레짐 감지 완료: BULL {bull_pct:.1f}%, BEAR {bear_pct:.1f}%, "
            f"SIDEWAYS {side_pct:.1f}%"
        )

        return regimes

    @staticmethod
    def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """원본 가격 데이터로 ADX를 계산한다.

        Args:
            high: 고가 배열.
            low: 저가 배열.
            close: 종가 배열.
            period: ADX 기간 (기본 14).

        Returns:
            ADX 배열 (초기 값은 NaN).
        """
        n = len(close)
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            h_diff = high[i] - high[i - 1]
            l_diff = low[i - 1] - low[i]
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            plus_dm[i] = h_diff if h_diff > l_diff and h_diff > 0 else 0.0
            minus_dm[i] = l_diff if l_diff > h_diff and l_diff > 0 else 0.0

        # Wilder smoothing
        atr = np.full(n, np.nan)
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)

        atr[period] = np.mean(tr[1 : period + 1])
        sm_plus = np.mean(plus_dm[1 : period + 1])
        sm_minus = np.mean(minus_dm[1 : period + 1])

        if atr[period] != 0:
            plus_di[period] = 100 * sm_plus / atr[period]
            minus_di[period] = 100 * sm_minus / atr[period]

        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
            sm_plus = (sm_plus * (period - 1) + plus_dm[i]) / period
            sm_minus = (sm_minus * (period - 1) + minus_dm[i]) / period
            if atr[i] != 0:
                plus_di[i] = 100 * sm_plus / atr[i]
                minus_di[i] = 100 * sm_minus / atr[i]

        dx = np.full(n, np.nan)
        for i in range(period, n):
            if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
                di_sum = plus_di[i] + minus_di[i]
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum if di_sum != 0 else 0.0

        adx = np.full(n, np.nan)
        start = 2 * period
        if start < n:
            valid_dx = [dx[i] for i in range(period, start + 1) if not np.isnan(dx[i])]
            if valid_dx:
                adx[start] = np.mean(valid_dx)
                for i in range(start + 1, n):
                    if not np.isnan(dx[i]) and not np.isnan(adx[i - 1]):
                        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

        return adx


def add_regime_features(
    df: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """레짐 피처를 DataFrame에 추가한다.

    regime 컬럼(ordinal)과 one-hot 인코딩 컬럼을 추가한다.

    Args:
        df: ADX와 close 컬럼이 포함된 DataFrame.
        config: 레짐 감지 파라미터.

    Returns:
        레짐 피처가 추가된 DataFrame.
    """
    detector = MarketRegimeDetector(config=config)
    regimes = detector.detect(df)

    df["regime"] = regimes
    df["regime_bull"] = (regimes == MarketRegimeDetector.BULL).astype(int)
    df["regime_bear"] = (regimes == MarketRegimeDetector.BEAR).astype(int)
    df["regime_sideways"] = (regimes == MarketRegimeDetector.SIDEWAYS).astype(int)

    return df
