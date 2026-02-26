"""데이터 전처리 파이프라인 모듈.

OHLCV + 피처 데이터를 모델 학습에 적합한 형태로 변환한다.
결측치/이상치 처리, 스케일링, 시계열 분할을 포함한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.data.collector import load_from_parquet, save_to_parquet
from src.data.features import build_features, create_target


def handle_missing_values(df: pd.DataFrame, method: str = "drop") -> pd.DataFrame:
    """결측치를 처리한다.

    Args:
        df: 입력 DataFrame.
        method: 처리 방법 ("drop", "ffill", "interpolate").

    Returns:
        결측치가 처리된 DataFrame.
    """
    missing_count = df.isna().sum().sum()
    if missing_count == 0:
        return df

    if method == "drop":
        df = df.dropna()
    elif method == "ffill":
        df = df.ffill().dropna()
    elif method == "interpolate":
        df = df.interpolate(method="linear").dropna()

    logger.info(f"결측치 처리 완료: {missing_count}개 ({method})")
    return df


def handle_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    z_threshold: float = 5.0,
) -> pd.DataFrame:
    """Z-score 기반으로 이상치를 클리핑한다.

    Args:
        df: 입력 DataFrame.
        columns: 이상치 처리 대상 컬럼 (None이면 수치형 전체).
        z_threshold: Z-score 임계값.

    Returns:
        이상치가 클리핑된 DataFrame.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_count = 0
    for col in columns:
        if col in ("target",):
            continue
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            continue
        z_scores = (df[col] - mean) / std
        mask = z_scores.abs() > z_threshold
        outlier_count += mask.sum()
        lower = mean - z_threshold * std
        upper = mean + z_threshold * std
        df[col] = df[col].clip(lower=lower, upper=upper)

    if outlier_count > 0:
        logger.info(f"이상치 클리핑 완료: {outlier_count}개 (z_threshold={z_threshold})")
    return df


def scale_features(
    df: pd.DataFrame,
    exclude_columns: list[str] | None = None,
    method: str = "standard",
) -> tuple[pd.DataFrame, StandardScaler | MinMaxScaler]:
    """피처를 스케일링한다.

    Args:
        df: 입력 DataFrame.
        exclude_columns: 스케일링 제외 컬럼 (e.g., ["target"]).
        method: 스케일링 방법 ("standard", "minmax").

    Returns:
        (스케일링된 DataFrame, 학습된 scaler).
    """
    if exclude_columns is None:
        exclude_columns = []

    feature_cols = [c for c in df.columns if c not in exclude_columns]

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"지원하지 않는 스케일링 방법: {method}")

    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    logger.info(f"스케일링 완료: {len(feature_cols)}개 컬럼 ({method})")
    return df, scaler


def split_timeseries(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """시계열 기반으로 train/validation/test를 분할한다.

    시간순으로 분할하여 look-ahead bias를 방지한다.

    Args:
        df: 입력 DataFrame (시간순 정렬 필수).
        train_ratio: 학습 데이터 비율.
        val_ratio: 검증 데이터 비율.
        test_ratio: 테스트 데이터 비율.

    Returns:
        (train, validation, test) DataFrame 튜플.

    Raises:
        ValueError: 비율 합이 1.0이 아닐 때.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"비율 합이 1.0이어야 합니다: {total}")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    logger.info(
        f"시계열 분할 완료: train={len(train)}행 ({train.index.min()} ~ {train.index.max()}), "
        f"val={len(val)}행, test={len(test)}행"
    )
    return train, val, test


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path = "data/processed",
    target_horizon: int = 4,
    target_threshold: float = 0.005,
    scale_method: str = "standard",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> dict[str, pd.DataFrame]:
    """전체 전처리 파이프라인을 실행한다.

    raw OHLCV → 피처 생성 → 타겟 생성 → 결측치 처리 → 이상치 처리 →
    시계열 분할 → 스케일링 → Parquet 저장.

    Args:
        input_path: raw OHLCV Parquet 파일 경로.
        output_dir: 전처리 결과 저장 디렉토리.
        target_horizon: 예측 기간 (캔들 수).
        target_threshold: 매수/매도 판단 기준 수익률.
        scale_method: 스케일링 방법 ("standard", "minmax").
        train_ratio: 학습 데이터 비율.
        val_ratio: 검증 데이터 비율.
        test_ratio: 테스트 데이터 비율.

    Returns:
        {"train": DataFrame, "val": DataFrame, "test": DataFrame}.
    """
    output_dir = Path(output_dir)
    logger.info(f"전처리 파이프라인 시작: {input_path}")

    # 1. 데이터 로드
    df = load_from_parquet(input_path)
    logger.info(f"원본 데이터: {len(df)}행, {len(df.columns)}개 컬럼")

    # 2. 피처 생성
    df = build_features(df)

    # 3. 타겟 생성
    df = create_target(df, horizon=target_horizon, threshold=target_threshold)

    # 4. 결측치 처리
    df = handle_missing_values(df, method="drop")

    # 5. 이상치 처리
    df = handle_outliers(df)

    # 6. 시계열 분할 (스케일링 전에 분할해야 data leakage 방지)
    train, val, test = split_timeseries(df, train_ratio, val_ratio, test_ratio)

    # 7. 스케일링 (train 기준으로 fit, val/test에 transform)
    exclude = ["target"]
    feature_cols = [c for c in train.columns if c not in exclude]

    if scale_method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    train = train.copy()
    val = val.copy()
    test = test.copy()

    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    val[feature_cols] = scaler.transform(val[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])

    logger.info(f"스케일링 완료: train 기준 fit → val/test transform ({scale_method})")

    # 8. 저장
    save_to_parquet(train, output_dir / "train.parquet")
    save_to_parquet(val, output_dir / "val.parquet")
    save_to_parquet(test, output_dir / "test.parquet")

    logger.info("전처리 파이프라인 완료")
    return {"train": train, "val": val, "test": test}
