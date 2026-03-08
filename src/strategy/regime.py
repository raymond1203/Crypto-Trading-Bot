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

        Args:
            df: ADX 컬럼(`adx`)과 close 컬럼이 포함된 DataFrame.

        Returns:
            레짐 배열 (BULL=1, SIDEWAYS=0, BEAR=-1).

        Raises:
            ValueError: 필수 컬럼이 없을 때.
        """
        if "adx" not in df.columns:
            raise ValueError("DataFrame에 'adx' 컬럼이 필요합니다.")
        if "close" not in df.columns:
            raise ValueError("DataFrame에 'close' 컬럼이 필요합니다.")

        adx_threshold = self.config["adx_trend_threshold"]
        sma_window = self.config["sma_window"]
        slope_window = self.config["sma_slope_window"]

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
