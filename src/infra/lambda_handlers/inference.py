"""Lambda 추론 핸들러.

EventBridge 스케줄로 매시간 트리거되어 앙상블 모델로 매매 신호를 생성한다.
S3에서 모델 로드 → 최신 OHLCV 피처 생성 → 추론 → 리스크 관리 → DynamoDB 기록.
Phase 4에서는 dry-run 모드로 동작 (신호 생성 + 기록만, 실제 주문 미실행).
"""

import os
import tempfile
from pathlib import Path

import boto3
from loguru import logger

from src.data.collector import load_from_parquet
from src.data.features import build_features
from src.data.preprocessor import scale_features
from src.models.ensemble import EnsembleModel
from src.models.lstm_model import LSTMSignalModel
from src.models.xgboost_model import XGBoostSignalModel
from src.strategy.portfolio import PortfolioManager
from src.strategy.risk import RiskManager

S3_DATA_BUCKET = os.environ.get("S3_DATA_BUCKET", "")
S3_MODEL_BUCKET = os.environ.get("S3_MODEL_BUCKET", "")
S3_OHLCV_KEY = os.environ.get("S3_OHLCV_KEY", "ohlcv/BTC_USDT_1h.parquet")
S3_MODEL_PREFIX = os.environ.get("S3_MODEL_PREFIX", "models/latest/")
SYMBOL = os.environ.get("SYMBOL", "BTC/USDT")
TABLE_PREFIX = os.environ.get("TABLE_PREFIX", "cryptosentinel")
ENV_NAME = os.environ.get("ENV_NAME", "dev")

MODEL_CACHE_DIR = Path("/tmp/models")  # noqa: S108 — Lambda /tmp 캐시

s3_client = boto3.client("s3")

# 모델 캐시 (warm start 시 재사용)
_cached_models: dict | None = None


def handler(event: dict, context: object) -> dict:
    """Lambda 엔트리포인트.

    Args:
        event: EventBridge 이벤트.
        context: Lambda 컨텍스트.

    Returns:
        추론 결과 딕셔너리.
    """
    logger.info(f"추론 시작: {SYMBOL}")

    portfolio = PortfolioManager(
        mode="dynamodb",
        table_prefix=TABLE_PREFIX,
        env_name=ENV_NAME,
    )

    # dry-run 모드 확인
    trading_mode = portfolio.get_bot_state("trading_mode") or "dry-run"
    bot_running = portfolio.get_bot_state("running") or "true"

    if bot_running != "true":
        logger.info("봇 중지 상태, 추론 스킵")
        return {"statusCode": 200, "body": {"status": "skipped", "reason": "bot_stopped"}}

    # 1. S3에서 OHLCV 데이터 로드
    with tempfile.TemporaryDirectory() as tmpdir:
        ohlcv_path = Path(tmpdir) / "ohlcv.parquet"
        s3_client.download_file(S3_DATA_BUCKET, S3_OHLCV_KEY, str(ohlcv_path))
        df = load_from_parquet(ohlcv_path)

    logger.info(f"OHLCV 데이터 로드: {len(df)}행, 마지막: {df.index[-1]}")

    # 2. 피처 엔지니어링
    df_features = build_features(df)
    logger.info(f"피처 생성 완료: {df_features.shape[1]}개 컬럼")

    # 3. 모델 로드 (캐시 활용)
    models = _load_models()
    xgb_model, lstm_model, ensemble_model, risk_manager = (
        models["xgboost"],
        models["lstm"],
        models["ensemble"],
        models["risk_manager"],
    )

    # 4. 스케일링 (원본 가격 보존)
    df_scaled = df_features.copy()
    if "close" in df_scaled.columns:
        df_scaled["close_raw"] = df_scaled["close"].copy()
    df_scaled, _ = scale_features(df_scaled, exclude_columns=["target", "close_raw"])

    # 5. 개별 모델 추론
    xgb_proba = xgb_model.predict_proba(df_scaled)
    lstm_proba = lstm_model.predict_proba(df_scaled)

    # LSTM seq_length 정렬
    seq_length = lstm_model.config.get("seq_length", 60)
    n_lstm = len(lstm_proba)
    df_aligned = df_scaled.iloc[seq_length - 1 : seq_length - 1 + n_lstm]
    xgb_aligned = xgb_proba[seq_length - 1 : seq_length - 1 + n_lstm]

    # 6. 앙상블 추론
    base_predictions = {"xgboost": xgb_aligned, "lstm": lstm_proba}
    signals = ensemble_model.predict(base_predictions)
    logger.info(f"앙상블 추론 완료: {n_lstm}개 bar")

    # 7. 리스크 관리
    adjusted_signals = risk_manager.process_signals(df_aligned, signals)

    # 8. 최신 신호 확인 및 포지션 관리
    latest_signal = int(adjusted_signals[-1])
    latest_price = float(df_aligned.iloc[-1]["close_raw"])
    latest_time = str(df_aligned.index[-1])

    result = _process_signal(
        portfolio=portfolio,
        signal=latest_signal,
        price=latest_price,
        timestamp=latest_time,
        trading_mode=trading_mode,
    )

    logger.info(f"추론 완료: signal={latest_signal}, price={latest_price}, mode={trading_mode}")

    return {
        "statusCode": 200,
        "body": {
            "signal": latest_signal,
            "price": latest_price,
            "timestamp": latest_time,
            "trading_mode": trading_mode,
            **result,
        },
    }


def _load_models() -> dict:
    """모델을 로드한다 (cold start 시 S3에서 다운로드, warm start 시 캐시 사용).

    Returns:
        모델 딕셔너리.
    """
    global _cached_models  # noqa: PLW0603

    if _cached_models is not None:
        logger.info("캐시된 모델 사용 (warm start)")
        return _cached_models

    logger.info("S3에서 모델 다운로드 (cold start)")
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _download_models_from_s3(MODEL_CACHE_DIR)

    xgb_model = XGBoostSignalModel.load(MODEL_CACHE_DIR)
    lstm_model = LSTMSignalModel.load(MODEL_CACHE_DIR)
    ensemble_model = EnsembleModel.load(MODEL_CACHE_DIR)
    risk_manager = RiskManager(config_path="configs/trading_config.yaml")

    _cached_models = {
        "xgboost": xgb_model,
        "lstm": lstm_model,
        "ensemble": ensemble_model,
        "risk_manager": risk_manager,
    }

    logger.info("모델 로드 완료")
    return _cached_models


def _download_models_from_s3(model_dir: Path) -> None:
    """S3에서 모델 파일을 다운로드한다.

    Args:
        model_dir: 로컬 저장 경로.
    """
    model_files = [
        "xgboost_signal_model.joblib",
        "lstm_signal_model.pt",
        "lstm_config.json",
        "ensemble_meta.joblib",
        "ensemble_meta.json",
    ]

    for filename in model_files:
        local_path = model_dir / filename
        if local_path.exists():
            continue
        s3_key = f"{S3_MODEL_PREFIX}{filename}"
        logger.info(f"다운로드: s3://{S3_MODEL_BUCKET}/{s3_key}")
        s3_client.download_file(S3_MODEL_BUCKET, s3_key, str(local_path))


def _process_signal(
    portfolio: PortfolioManager,
    signal: int,
    price: float,
    timestamp: str,
    trading_mode: str,
) -> dict:
    """신호에 따라 포지션을 관리한다.

    Args:
        portfolio: 포지션 관리자.
        signal: 매매 신호 (1=매수, -1=매도, 0=관망).
        price: 현재 가격.
        timestamp: 현재 시각.
        trading_mode: 거래 모드 ("dry-run" | "live").

    Returns:
        처리 결과 딕셔너리.
    """
    current_position = portfolio.get_current_position(SYMBOL)

    if signal == 1 and current_position is None:
        # 매수 신호 + 포지션 없음 → 포지션 오픈
        position = portfolio.open_position(SYMBOL, "long", price, 0.5)
        action = "open_long"
        logger.info(f"[{trading_mode}] 롱 포지션 오픈 @ {price}")

        if trading_mode == "live":
            # TODO: Binance 실제 주문 실행
            pass

        return {"action": action, "position": str(position)}

    if signal == -1 and current_position is not None:
        # 매도 신호 + 포지션 있음 → 포지션 청산
        trade = portfolio.close_position(SYMBOL, price)
        action = "close_position"
        logger.info(f"[{trading_mode}] 포지션 청산 @ {price}, PnL={trade.pnl_pct:+.2f}%")

        if trading_mode == "live":
            # TODO: Binance 실제 주문 실행
            pass

        return {"action": action, "pnl_pct": trade.pnl_pct}

    return {"action": "hold", "has_position": current_position is not None}
