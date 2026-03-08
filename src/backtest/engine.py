"""시뮬레이션 기반 백테스트 엔진.

매매 신호와 OHLCV 데이터로 트레이딩을 시뮬레이션하고,
수수료/슬리피지를 반영한 현실적 성능 지표를 계산한다.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.backtest.metrics import BacktestMetrics


@dataclass
class Trade:
    """개별 거래 기록.

    Attributes:
        entry_time: 진입 시간.
        exit_time: 청산 시간.
        entry_price: 진입 가격 (슬리피지 포함).
        exit_price: 청산 가격 (슬리피지 포함).
        position_size: 포지션 크기 (BTC).
        pnl: 수익률 (e.g., 0.03 = +3%).
        pnl_abs: 절대 수익 (USD).
        side: 거래 방향.
    """

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_abs: float
    side: str = "long"


@dataclass
class BacktestResult:
    """백테스트 실행 결과.

    Attributes:
        trades: 거래 DataFrame.
        equity_curve: 자산 가치 시리즈.
        metrics: 성능 지표 딕셔너리.
        config: 사용된 설정 딕셔너리.
    """

    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: dict
    config: dict


class BacktestEngine:
    """시뮬레이션 기반 백테스트 엔진.

    매매 신호(1=매수, 0=관망, -1=매도)를 받아
    수수료/슬리피지를 반영한 트레이딩 시뮬레이션을 실행한다.

    Attributes:
        config: 백테스트 파라미터 딕셔너리.
    """

    DEFAULT_PARAMS: dict = {
        "initial_capital": 10000.0,
        "commission": 0.001,
        "slippage": 0.0005,
        "position_size": 0.95,
    }

    def __init__(
        self,
        config: dict | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """백테스트 엔진을 초기화한다.

        Args:
            config: 백테스트 파라미터 딕셔너리 (backtest 섹션).
            config_path: YAML 설정 파일 경로 (config 미지정 시 사용).
        """
        if config is None and config_path is not None:
            full_config = self._load_yaml(config_path)
            config = full_config.get("backtest", {})

        config = config or {}
        self.config: dict = {**self.DEFAULT_PARAMS, **config}

        ps = self.config["position_size"]
        comm = self.config["commission"]
        if ps * (1 + comm) > 1.0:
            raise ValueError(
                f"position_size({ps}) × (1 + commission({comm})) = {ps * (1 + comm):.4f} > 1.0 — "
                f"매수 시 capital이 음수가 됩니다."
            )

        logger.info(f"BacktestEngine 초기화: {self.config}")

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
        signals: np.ndarray | pd.Series,
    ) -> BacktestResult:
        """백테스트를 실행한다.

        Args:
            df: OHLCV + 피처 DataFrame (close 컬럼 필수).
            signals: 매매 신호 배열 (1=매수, 0=관망, -1=매도).
                     df와 길이가 같아야 한다.

        Returns:
            BacktestResult (trades, equity_curve, metrics, config).

        Raises:
            ValueError: signals 길이가 df와 불일치할 때.
            ValueError: df에 close 컬럼이 없을 때.
        """
        # 원본 가격 컬럼 우선 사용 (스케일링된 가격으로 시뮬레이션 방지)
        price_col = "close_raw" if "close_raw" in df.columns else "close"
        if price_col == "close_raw":
            logger.info("원본 가격(close_raw) 사용하여 백테스트 실행")

        if "close" not in df.columns and "close_raw" not in df.columns:
            raise ValueError("DataFrame에 'close' 또는 'close_raw' 컬럼이 필요합니다.")

        if isinstance(signals, pd.Series):
            signals = signals.values

        if len(signals) != len(df):
            raise ValueError(f"signals 길이({len(signals)})가 df 길이({len(df)})와 불일치합니다.")

        initial_capital = self.config["initial_capital"]
        commission = self.config["commission"]
        slippage = self.config["slippage"]
        position_size_ratio = self.config["position_size"]

        capital = initial_capital
        position = 0.0
        entry_price = 0.0
        entry_time: pd.Timestamp | None = None
        trades: list[Trade] = []
        equity: list[float] = []

        close_prices = df[price_col].values
        timestamps = df.index

        for i in range(len(df)):
            price = close_prices[i]
            signal = int(signals[i])
            timestamp = timestamps[i]

            if signal == 1 and position == 0:
                buy_price = price * (1 + slippage)
                invest_amount = capital * position_size_ratio
                btc_amount = invest_amount / buy_price
                cost = btc_amount * buy_price * (1 + commission)
                capital -= cost
                position = btc_amount
                entry_price = buy_price
                entry_time = timestamp

            elif signal == -1 and position > 0:
                sell_price = price * (1 - slippage)
                revenue = position * sell_price * (1 - commission)
                pnl = (sell_price - entry_price) / entry_price
                entry_cost = position * entry_price * (1 + commission)
                pnl_abs = revenue - entry_cost

                trades.append(
                    Trade(
                        entry_time=entry_time,  # type: ignore[arg-type]
                        exit_time=timestamp,
                        entry_price=entry_price,
                        exit_price=sell_price,
                        position_size=position,
                        pnl=pnl,
                        pnl_abs=pnl_abs,
                        side="long",
                    )
                )
                capital += revenue
                position = 0.0

            current_equity = capital + (position * price)
            equity.append(current_equity)

        # 미청산 포지션 강제 청산
        if position > 0:
            last_price = close_prices[-1]
            sell_price = last_price * (1 - slippage)
            revenue = position * sell_price * (1 - commission)
            pnl = (sell_price - entry_price) / entry_price
            entry_cost = position * entry_price * (1 + commission)
            pnl_abs = revenue - entry_cost

            trades.append(
                Trade(
                    entry_time=entry_time,  # type: ignore[arg-type]
                    exit_time=timestamps[-1],
                    entry_price=entry_price,
                    exit_price=sell_price,
                    position_size=position,
                    pnl=pnl,
                    pnl_abs=pnl_abs,
                    side="long",
                )
            )
            capital += revenue
            equity[-1] = capital

        equity_curve = pd.Series(equity, index=df.index, name="equity")
        trades_df = self._build_trades_dataframe(trades)

        metrics = BacktestMetrics.generate_report(
            trades=trades_df,
            equity_curve=equity_curve,
            prices=df[price_col],
            initial_capital=initial_capital,
        )

        logger.info(f"백테스트 완료: {len(trades_df)}건 거래, 최종 자본 ${equity_curve.iloc[-1]:,.2f}")

        return BacktestResult(
            trades=trades_df,
            equity_curve=equity_curve,
            metrics=metrics,
            config=self.config,
        )

    @staticmethod
    def _build_trades_dataframe(trades: list[Trade]) -> pd.DataFrame:
        """Trade 객체 리스트를 DataFrame으로 변환한다.

        Args:
            trades: Trade 객체 리스트.

        Returns:
            거래 DataFrame. 비어있으면 올바른 스키마의 빈 DataFrame.
        """
        if not trades:
            return pd.DataFrame(
                columns=[
                    "entry_time",
                    "exit_time",
                    "entry_price",
                    "exit_price",
                    "position_size",
                    "pnl",
                    "pnl_abs",
                    "side",
                    "duration",
                ]
            )

        records = [
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "position_size": t.position_size,
                "pnl": t.pnl,
                "pnl_abs": t.pnl_abs,
                "side": t.side,
            }
            for t in trades
        ]
        trades_df = pd.DataFrame(records)
        trades_df["duration"] = trades_df["exit_time"] - trades_df["entry_time"]
        return trades_df


def run_backtest_cli() -> None:
    """CLI 엔트리포인트: 백테스트를 실행한다."""
    parser = argparse.ArgumentParser(description="CryptoSentinel 백테스트")
    parser.add_argument(
        "--config",
        default="configs/trading_config.yaml",
        help="설정 파일 경로",
    )
    parser.add_argument(
        "--data",
        default="data/processed/btc_usdt_features_1h.parquet",
        help="피처 데이터 Parquet 경로",
    )
    parser.add_argument(
        "--output",
        default="data/models/backtest_results.json",
        help="결과 JSON 출력 경로",
    )
    parser.add_argument(
        "--model-dir",
        default="data/models",
        help="모델 디렉토리",
    )
    args = parser.parse_args()

    from src.data.collector import load_from_parquet
    from src.models.ensemble import EnsembleModel
    from src.models.lstm_model import LSTMSignalModel
    from src.models.xgboost_model import XGBoostSignalModel

    logger.info(f"데이터 로드: {args.data}")
    df = load_from_parquet(args.data)

    logger.info(f"모델 로드: {args.model_dir}")
    xgb_model = XGBoostSignalModel.load(args.model_dir)
    lstm_model = LSTMSignalModel.load(args.model_dir)
    ensemble = EnsembleModel.load(args.model_dir)

    logger.info("앙상블 신호 생성")
    xgb_proba = xgb_model.predict_proba(df)
    lstm_proba = lstm_model.predict_proba(df)

    # LSTM 출력은 seq_length만큼 짧으므로 정렬 (윈도우 마지막 시점 기준)
    seq_length = lstm_model.config.get("seq_length", 60)
    n_lstm = len(lstm_proba)
    df_aligned = df.iloc[seq_length - 1 : seq_length - 1 + n_lstm]
    xgb_proba_aligned = xgb_proba[seq_length - 1 : seq_length - 1 + n_lstm]

    base_predictions = {"xgboost": xgb_proba_aligned, "lstm": lstm_proba}
    signals = ensemble.predict(base_predictions)

    logger.info("백테스트 실행")
    engine = BacktestEngine(config_path=args.config)
    result = engine.run(df_aligned, signals)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.metrics, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"백테스트 결과 저장: {output_path}")


if __name__ == "__main__":
    run_backtest_cli()
