"""기술적 지표 피처 엔지니어링 모듈.

OHLCV 원본 데이터에서 트렌드, 모멘텀, 변동성, 거래량, 커스텀, 시간 피처를 생성한다.
"""

import numpy as np
import pandas as pd
import ta
from loguru import logger


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """추세 관련 기술적 지표를 추가한다.

    Args:
        df: OHLCV DataFrame.

    Returns:
        추세 지표가 추가된 DataFrame.
    """
    # SMA (Simple Moving Average)
    df["sma_7"] = ta.trend.sma_indicator(df["close"], window=7)
    df["sma_25"] = ta.trend.sma_indicator(df["close"], window=25)
    df["sma_99"] = ta.trend.sma_indicator(df["close"], window=99)

    # EMA (Exponential Moving Average)
    df["ema_12"] = ta.trend.ema_indicator(df["close"], window=12)
    df["ema_26"] = ta.trend.ema_indicator(df["close"], window=26)

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_histogram"] = macd.macd_diff()

    # ADX (Average Directional Index)
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # Ichimoku
    ichi = ta.trend.IchimokuIndicator(df["high"], df["low"])
    df["ichimoku_a"] = ichi.ichimoku_a()
    df["ichimoku_b"] = ichi.ichimoku_b()

    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """모멘텀 관련 기술적 지표를 추가한다.

    Args:
        df: OHLCV DataFrame.

    Returns:
        모멘텀 지표가 추가된 DataFrame.
    """
    # RSI
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
    df["rsi_7"] = ta.momentum.rsi(df["close"], window=7)

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Williams %R
    df["williams_r"] = ta.momentum.williams_r(df["high"], df["low"], df["close"])

    # ROC (Rate of Change)
    df["roc_10"] = ta.momentum.roc(df["close"], window=10)

    # CCI (Commodity Channel Index)
    df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"])

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """변동성 관련 기술적 지표를 추가한다.

    Args:
        df: OHLCV DataFrame.

    Returns:
        변동성 지표가 추가된 DataFrame.
    """
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct"] = bb.bollinger_pband()

    # ATR (Average True Range)
    df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])

    # Keltner Channel
    kc = ta.volatility.KeltnerChannel(df["high"], df["low"], df["close"])
    df["kc_upper"] = kc.keltner_channel_hband()
    df["kc_lower"] = kc.keltner_channel_lband()

    # 실현 변동성 (20기간)
    df["realized_vol_20"] = df["close"].pct_change().rolling(20).std() * (252**0.5)

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """거래량 관련 기술적 지표를 추가한다.

    Args:
        df: OHLCV DataFrame.

    Returns:
        거래량 지표가 추가된 DataFrame.
    """
    # OBV (On Balance Volume)
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])

    # VWAP 근사
    df["vwap"] = (
        (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()
    )

    # Volume SMA
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

    # MFI (Money Flow Index)
    df["mfi"] = ta.volume.money_flow_index(df["high"], df["low"], df["close"], df["volume"])

    # CMF (Chaikin Money Flow)
    df["cmf"] = ta.volume.chaikin_money_flow(df["high"], df["low"], df["close"], df["volume"])

    return df


def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """커스텀 파생 피처를 추가한다.

    Args:
        df: OHLCV + 기술적 지표 DataFrame.

    Returns:
        커스텀 피처가 추가된 DataFrame.
    """
    # 가격 변화율
    for period in [1, 3, 5, 10, 20]:
        df[f"return_{period}"] = df["close"].pct_change(period)

    # 로그 수익률
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # 고가-저가 비율
    df["hl_ratio"] = (df["high"] - df["low"]) / df["close"]

    # 캔들 패턴
    df["candle_body"] = (df["close"] - df["open"]) / df["open"]
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]

    # 가격 위치 (최근 N기간 대비)
    for period in [20, 50]:
        rolling_min = df["low"].rolling(period).min()
        rolling_max = df["high"].rolling(period).max()
        df[f"price_position_{period}"] = (df["close"] - rolling_min) / (rolling_max - rolling_min)

    # SMA 크로스오버 시그널
    df["sma_cross_7_25"] = (df["sma_7"] > df["sma_25"]).astype(int)
    df["sma_cross_25_99"] = (df["sma_25"] > df["sma_99"]).astype(int)

    return df


def add_time_features(df: pd.DataFrame, *, use_raw: bool = False) -> pd.DataFrame:
    """시간 관련 피처를 추가한다.

    sin/cos 인코딩만 사용하여 주기성을 표현한다.
    Raw 시간 피처(hour, day_of_week, month)는 XGBoost에서 과적합을 유발하므로
    기본적으로 비활성화한다.

    Args:
        df: timestamp 인덱스의 DataFrame.
        use_raw: True면 raw 시간 피처(hour, day_of_week, month)도 추가.

    Returns:
        시간 피처가 추가된 DataFrame.
    """
    if use_raw:
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month

    # 사인/코사인 인코딩 (주기성 반영)
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    return df


def build_features(
    df: pd.DataFrame,
    *,
    use_raw_time: bool = False,
) -> pd.DataFrame:
    """전체 피처 엔지니어링 파이프라인을 실행한다.

    Args:
        df: OHLCV DataFrame.
        use_raw_time: True면 raw 시간 피처(hour, day_of_week, month)도 추가.

    Returns:
        모든 피처가 추가된 DataFrame (NaN 행 제거됨).
    """
    initial_len = len(df)

    df = add_trend_features(df)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_custom_features(df)
    df = add_time_features(df, use_raw=use_raw_time)

    df = df.dropna()
    feature_count = len(df.columns) - 5  # OHLCV 5개 제외
    logger.info(f"피처 생성 완료: {feature_count}개 피처, {initial_len}행 → {len(df)}행 (NaN {initial_len - len(df)}행 제거)")

    return df


def create_target(
    df: pd.DataFrame,
    horizon: int = 4,
    threshold: float = 0.005,
) -> pd.DataFrame:
    """매매 신호 타겟을 생성한다.

    Args:
        df: 피처가 포함된 DataFrame.
        horizon: 예측 기간 (캔들 수).
        threshold: 매수/매도 판단 기준 수익률.

    Returns:
        target 컬럼이 추가된 DataFrame.
        - 1: 매수 (future return > threshold)
        - 0: 관망 (-threshold <= future return <= threshold)
        - -1: 매도 (future return < -threshold)
    """
    df = df.copy()
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    df["target"] = 0
    df.loc[future_return > threshold, "target"] = 1
    df.loc[future_return < -threshold, "target"] = -1

    before = len(df)
    df = df.iloc[:-horizon]  # 타겟 계산 불가능한 마지막 N행 제거
    logger.info(f"타겟 생성 완료: horizon={horizon}, threshold={threshold}, {before}행 → {len(df)}행")

    return df
