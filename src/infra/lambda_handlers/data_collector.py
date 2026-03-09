"""Lambda 데이터 수집 핸들러.

EventBridge 스케줄로 매시간 트리거되어 Binance BTC/USDT OHLCV 데이터를
S3에서 로드 → 최신 데이터 수집 → S3에 저장한다.
"""

import json
import os
import tempfile
from pathlib import Path

import boto3
from loguru import logger
from pandas import DataFrame

from src.data.collector import (
    fetch_ohlcv,
    load_from_parquet,
    save_to_parquet,
    update_data,
    validate_data,
)

S3_DATA_BUCKET = os.environ.get("S3_DATA_BUCKET", "")
S3_OHLCV_KEY = os.environ.get("S3_OHLCV_KEY", "ohlcv/BTC_USDT_1h.parquet")
SYMBOL = os.environ.get("SYMBOL", "BTC/USDT")
TIMEFRAME = os.environ.get("TIMEFRAME", "1h")

s3_client = boto3.client("s3")


def handler(event: dict, context: object) -> dict:
    """Lambda 엔트리포인트.

    Args:
        event: EventBridge 이벤트.
        context: Lambda 컨텍스트.

    Returns:
        실행 결과 딕셔너리.
    """
    logger.info(f"데이터 수집 시작: {SYMBOL} {TIMEFRAME}")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / "ohlcv.parquet"

        # S3에서 기존 데이터 다운로드
        existing_df = _download_from_s3(local_path)

        if existing_df is not None:
            # 기존 데이터에 최신 캔들 추가
            logger.info(f"기존 데이터: {len(existing_df)}행, 마지막: {existing_df.index[-1]}")
            updated_df = update_data(local_path, symbol=SYMBOL, timeframe=TIMEFRAME)
        else:
            # 최초 수집: 최근 30일
            logger.info("기존 데이터 없음, 최초 수집 (30일)")
            updated_df = fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, since="30d")

        new_rows = len(updated_df) - (len(existing_df) if existing_df is not None else 0)
        logger.info(f"수집 완료: {len(updated_df)}행 (신규 {new_rows}행)")

        # 데이터 검증
        issues = validate_data(updated_df)
        if any(issues.values()):
            logger.warning(f"데이터 품질 이슈: {json.dumps(issues, default=str)}")

        # S3에 업로드
        save_to_parquet(updated_df, local_path)
        _upload_to_s3(local_path)

    return {
        "statusCode": 200,
        "body": {
            "total_rows": len(updated_df),
            "new_rows": new_rows,
            "last_timestamp": str(updated_df.index[-1]),
            "issues": {k: len(v) if isinstance(v, list) else v for k, v in issues.items()},
        },
    }


def _download_from_s3(local_path: Path) -> DataFrame | None:
    """S3에서 Parquet 파일을 다운로드한다.

    Returns:
        DataFrame 또는 파일 미존재 시 None.
    """
    try:
        s3_client.download_file(S3_DATA_BUCKET, S3_OHLCV_KEY, str(local_path))
        return load_from_parquet(local_path)
    except s3_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logger.info("S3에 기존 데이터 없음")
            return None
        raise


def _upload_to_s3(local_path: Path) -> None:
    """Parquet 파일을 S3에 업로드한다."""
    s3_client.upload_file(str(local_path), S3_DATA_BUCKET, S3_OHLCV_KEY)
    logger.info(f"S3 업로드 완료: s3://{S3_DATA_BUCKET}/{S3_OHLCV_KEY}")
