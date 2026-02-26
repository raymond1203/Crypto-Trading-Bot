"""Binance OHLCV 데이터 수집 모듈.

ccxt를 사용하여 BTC/USDT 과거 캔들 데이터를 수집하고,
Parquet 포맷으로 저장/로드/증분 업데이트한다.
"""

import time
from pathlib import Path

import ccxt
import pandas as pd
from loguru import logger

# Binance API 제한
_MAX_CANDLES_PER_REQUEST = 1000
_MAX_RETRIES = 5
_BASE_BACKOFF_SEC = 1.0


def _create_exchange() -> ccxt.binance:
    """ccxt Binance 인스턴스를 생성한다.

    Returns:
        Rate limit이 활성화된 Binance 인스턴스.
    """
    return ccxt.binance({"enableRateLimit": True})


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    since: str = "2024-01-01",
    limit: int = _MAX_CANDLES_PER_REQUEST,
) -> pd.DataFrame:
    """Binance에서 OHLCV 데이터를 수집한다.

    Args:
        symbol: 거래쌍 (e.g., "BTC/USDT").
        timeframe: 캔들 간격 (e.g., "1h", "4h", "1d").
        since: 시작 날짜 (ISO format, e.g., "2024-01-01").
        limit: 요청당 최대 캔들 수 (Binance 최대 1000).

    Returns:
        timestamp 인덱스의 OHLCV DataFrame.

    Raises:
        ccxt.BaseError: API 호출 실패 시.
    """
    exchange = _create_exchange()
    since_ts = exchange.parse8601(f"{since}T00:00:00Z")

    all_candles: list[list] = []
    request_count = 0

    logger.info(f"데이터 수집 시작: {symbol} {timeframe} (since={since})")

    while True:
        candles = _fetch_with_retry(exchange, symbol, timeframe, since_ts, limit)
        if not candles:
            break

        all_candles.extend(candles)
        request_count += 1
        since_ts = candles[-1][0] + 1

        if request_count % 50 == 0:
            logger.info(f"  {len(all_candles)}개 캔들 수집 완료...")

        if len(candles) < limit:
            break

    df = _candles_to_dataframe(all_candles)
    logger.info(f"수집 완료: {len(df)}개 캔들 ({df.index.min()} ~ {df.index.max()})")
    return df


def _fetch_with_retry(
    exchange: ccxt.binance,
    symbol: str,
    timeframe: str,
    since: int,
    limit: int,
) -> list[list]:
    """지수 백오프를 적용하여 OHLCV 데이터를 요청한다.

    Args:
        exchange: ccxt Binance 인스턴스.
        symbol: 거래쌍.
        timeframe: 캔들 간격.
        since: 시작 타임스탬프 (ms).
        limit: 요청당 최대 캔들 수.

    Returns:
        OHLCV 캔들 리스트.

    Raises:
        ccxt.BaseError: 최대 재시도 횟수 초과 시.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except ccxt.RateLimitExceeded:
            wait = _BASE_BACKOFF_SEC * (2**attempt)
            logger.warning(f"Rate limit 초과, {wait}초 대기 (시도 {attempt + 1}/{_MAX_RETRIES})")
            time.sleep(wait)
        except ccxt.NetworkError as e:
            wait = _BASE_BACKOFF_SEC * (2**attempt)
            logger.warning(f"네트워크 에러: {e}, {wait}초 대기 (시도 {attempt + 1}/{_MAX_RETRIES})")
            time.sleep(wait)

    raise ccxt.BaseError(f"{_MAX_RETRIES}회 재시도 실패: {symbol} {timeframe}")


def _candles_to_dataframe(candles: list[list]) -> pd.DataFrame:
    """캔들 리스트를 DataFrame으로 변환한다.

    Args:
        candles: [timestamp, open, high, low, close, volume] 형태의 리스트.

    Returns:
        timestamp 인덱스의 OHLCV DataFrame.
    """
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def save_to_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """DataFrame을 Parquet 포맷으로 저장한다.

    Args:
        df: 저장할 DataFrame.
        path: 저장 경로.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    logger.info(f"저장 완료: {path} ({len(df)}행)")


def load_from_parquet(path: str | Path) -> pd.DataFrame:
    """Parquet 파일을 DataFrame으로 로드한다.

    Args:
        path: 파일 경로.

    Returns:
        로드된 DataFrame.

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"로드 완료: {path} ({len(df)}행)")
    return df


def update_data(
    existing_path: str | Path,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
) -> pd.DataFrame:
    """기존 데이터에 최신 데이터를 추가한다.

    Args:
        existing_path: 기존 Parquet 파일 경로.
        symbol: 거래쌍.
        timeframe: 캔들 간격.

    Returns:
        업데이트된 DataFrame.
    """
    existing = load_from_parquet(existing_path)
    last_ts = existing.index[-1].strftime("%Y-%m-%d")

    logger.info(f"증분 업데이트: {last_ts}부터 최신 데이터 수집")
    new_data = fetch_ohlcv(symbol, timeframe, since=last_ts)

    combined = pd.concat([existing, new_data])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    logger.info(f"업데이트 완료: {len(existing)}행 → {len(combined)}행 (+{len(combined) - len(existing)})")
    return combined


def validate_data(df: pd.DataFrame, timeframe: str = "1h") -> dict[str, int]:
    """데이터 품질을 검증한다.

    Args:
        df: 검증할 OHLCV DataFrame.
        timeframe: 캔들 간격 (gap 탐지용).

    Returns:
        검증 결과 딕셔너리.
    """
    issues: dict[str, int] = {}

    # 중복 타임스탬프
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        issues["duplicate_timestamps"] = int(duplicates)

    # 가격 이상치 (0 이하)
    zero_prices = (df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
    if zero_prices > 0:
        issues["zero_or_negative_prices"] = int(zero_prices)

    # volume 0인 캔들
    zero_volume = (df["volume"] == 0).sum()
    if zero_volume > 0:
        issues["zero_volume"] = int(zero_volume)

    # 빠진 캔들 (gap) 탐지
    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
    if timeframe in freq_map:
        expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq_map[timeframe])
        missing = len(expected_index) - len(df)
        if missing > 0:
            issues["missing_candles"] = missing

    if issues:
        logger.warning(f"데이터 검증 이슈: {issues}")
    else:
        logger.info("데이터 검증 통과: 이슈 없음")

    return issues
