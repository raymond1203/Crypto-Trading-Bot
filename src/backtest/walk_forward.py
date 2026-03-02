"""Walk-forward 검증 모듈.

과거 N개 데이터로 학습 → 미래 M개 데이터로 테스트 → 윈도우 이동 → 반복.
구간별 성능 일관성으로 모델 과적합 여부를 판단한다.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.backtest.engine import BacktestEngine, BacktestResult


@dataclass
class WindowResult:
    """개별 윈도우 검증 결과.

    Attributes:
        window_id: 윈도우 번호 (0-based).
        train_start: 학습 구간 시작 시간.
        train_end: 학습 구간 종료 시간.
        test_start: 테스트 구간 시작 시간.
        test_end: 테스트 구간 종료 시간.
        backtest_result: 테스트 구간 백테스트 결과.
    """

    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    backtest_result: BacktestResult


@dataclass
class WalkForwardResult:
    """Walk-forward 검증 전체 결과.

    Attributes:
        window_results: 윈도우별 결과 리스트.
        summary: 윈도우별 지표 평균/표준편차 딕셔너리.
        equity_curve: 테스트 구간을 이어붙인 연속 equity 시리즈.
        config: 사용된 설정 딕셔너리.
    """

    window_results: list[WindowResult]
    summary: dict
    equity_curve: pd.Series
    config: dict


class WalkForwardValidator:
    """Walk-forward 검증기.

    signal_fn 콜백을 받아 윈도우별로 학습 → 예측 → 백테스트를 반복한다.
    모델에 의존하지 않으며, 콜백 내부에서 학습/예측을 자유롭게 처리한다.

    Attributes:
        config: walk-forward 파라미터 딕셔너리.
        backtest_config: BacktestEngine에 전달할 설정.
    """

    DEFAULT_PARAMS: dict = {
        "train_window": 5000,
        "test_window": 1000,
        "step": 500,
    }

    def __init__(
        self,
        config: dict | None = None,
        config_path: str | Path | None = None,
        backtest_config: dict | None = None,
    ) -> None:
        """Walk-forward 검증기를 초기화한다.

        Args:
            config: walk-forward 파라미터 딕셔너리.
            config_path: YAML 설정 파일 경로 (config 미지정 시 사용).
            backtest_config: BacktestEngine에 전달할 설정.
        """
        if config is None and config_path is not None:
            full_config = self._load_yaml(config_path)
            config = full_config.get("backtest", {}).get("walk_forward", {})
            if backtest_config is None:
                bt_cfg = full_config.get("backtest", {})
                backtest_config = {k: v for k, v in bt_cfg.items() if k != "walk_forward"}

        config = config or {}
        self.config: dict = {**self.DEFAULT_PARAMS, **config}
        self.backtest_config: dict | None = backtest_config
        logger.info(f"WalkForwardValidator 초기화: {self.config}")

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

    def run(
        self,
        df: pd.DataFrame,
        signal_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    ) -> WalkForwardResult:
        """Walk-forward 검증을 실행한다.

        Args:
            df: OHLCV + 피처 DataFrame (close 컬럼 필수).
            signal_fn: 학습/예측 콜백.
                       (train_df, test_df) → test 구간 매매 신호 배열.

        Returns:
            WalkForwardResult (window_results, summary, equity_curve, config).

        Raises:
            ValueError: df에 close 컬럼이 없을 때.
            ValueError: 데이터가 최소 1개 윈도우에 부족할 때.
        """
        if "close" not in df.columns:
            raise ValueError("DataFrame에 'close' 컬럼이 필요합니다.")

        windows = self._generate_windows(len(df))
        if not windows:
            train_w = self.config["train_window"]
            test_w = self.config["test_window"]
            raise ValueError(
                f"데이터 길이({len(df)})가 최소 윈도우 크기"
                f"(train={train_w} + test={test_w} = {train_w + test_w})보다 작습니다."
            )

        engine = BacktestEngine(config=self.backtest_config)
        window_results: list[WindowResult] = []

        for i, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
            train_df = df.iloc[tr_start:tr_end]
            test_df = df.iloc[te_start:te_end]

            logger.info(
                f"윈도우 {i + 1}/{len(windows)}: "
                f"train [{train_df.index[0]} ~ {train_df.index[-1]}], "
                f"test [{test_df.index[0]} ~ {test_df.index[-1]}]"
            )

            signals = signal_fn(train_df, test_df)
            result = engine.run(test_df, signals)

            window_results.append(
                WindowResult(
                    window_id=i,
                    train_start=train_df.index[0],
                    train_end=train_df.index[-1],
                    test_start=test_df.index[0],
                    test_end=test_df.index[-1],
                    backtest_result=result,
                )
            )

        summary = self._build_summary(window_results)
        equity_curve = self._build_equity_curve(window_results)

        logger.info(f"Walk-forward 완료: {len(window_results)}개 윈도우, 총 {summary['total_trades']}건 거래")

        return WalkForwardResult(
            window_results=window_results,
            summary=summary,
            equity_curve=equity_curve,
            config=self.config,
        )

    def _generate_windows(self, data_length: int) -> list[tuple[int, int, int, int]]:
        """윈도우 인덱스를 생성한다.

        Args:
            data_length: 전체 데이터 길이.

        Returns:
            (train_start, train_end, test_start, test_end) 튜플 리스트.
            인덱스는 iloc 슬라이싱 기준 (end는 exclusive).
        """
        train_w = self.config["train_window"]
        test_w = self.config["test_window"]
        step = self.config["step"]

        windows: list[tuple[int, int, int, int]] = []
        start = 0

        while start + train_w + test_w <= data_length:
            tr_start = start
            tr_end = start + train_w
            te_start = tr_end
            te_end = tr_end + test_w
            windows.append((tr_start, tr_end, te_start, te_end))
            start += step

        return windows

    @staticmethod
    def _build_summary(window_results: list[WindowResult]) -> dict:
        """윈도우별 지표의 평균/표준편차를 계산한다.

        Args:
            window_results: 윈도우별 결과 리스트.

        Returns:
            요약 딕셔너리 (n_windows, total_trades, 지표별 mean/std).
        """
        aggregate_keys = [
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "profit_factor",
            "strategy_return",
        ]

        summary: dict = {
            "n_windows": len(window_results),
            "total_trades": sum(wr.backtest_result.metrics["total_trades"] for wr in window_results),
        }

        for key in aggregate_keys:
            values = []
            for wr in window_results:
                val = wr.backtest_result.metrics.get(key)
                if val is not None and np.isfinite(val):
                    values.append(val)

            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
            else:
                summary[f"{key}_mean"] = 0.0
                summary[f"{key}_std"] = 0.0

        return summary

    @staticmethod
    def _build_equity_curve(
        window_results: list[WindowResult],
    ) -> pd.Series:
        """테스트 구간 equity를 이어붙여 연속 equity curve를 생성한다.

        각 윈도우의 시작 equity를 직전 윈도우 마지막 equity에 맞춰 스케일링한다.

        Args:
            window_results: 윈도우별 결과 리스트.

        Returns:
            연속 equity 시리즈 (테스트 구간만 포함).
        """
        if not window_results:
            return pd.Series(dtype=float, name="equity")

        segments: list[pd.Series] = []
        carry_capital = window_results[0].backtest_result.equity_curve.iloc[0]

        for wr in window_results:
            eq = wr.backtest_result.equity_curve
            window_start = eq.iloc[0]

            scale = carry_capital / window_start if window_start != 0 else 1.0

            scaled_eq = eq * scale
            segments.append(scaled_eq)
            carry_capital = scaled_eq.iloc[-1]

        combined = pd.concat(segments)
        combined.name = "equity"
        return combined
