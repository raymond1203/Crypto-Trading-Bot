"""BacktestEngine 단위 테스트.

합성 OHLCV 데이터와 매매 신호로 백테스트 시뮬레이션 로직을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest.engine import BacktestEngine, BacktestResult

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """20-bar 합성 OHLCV 데이터."""
    np.random.seed(42)
    n = 20
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    base = 40000.0
    close = base + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame(
        {
            "open": close - np.abs(np.random.randn(n) * 20),
            "high": close + np.abs(np.random.randn(n) * 50),
            "low": close - np.abs(np.random.randn(n) * 50),
            "close": close,
            "volume": np.random.uniform(100, 1000, n),
        },
        index=dates,
    )


@pytest.fixture
def engine() -> BacktestEngine:
    """기본 설정 BacktestEngine."""
    return BacktestEngine()


@pytest.fixture
def zero_cost_engine() -> BacktestEngine:
    """수수료/슬리피지 없는 BacktestEngine (검증 용이)."""
    return BacktestEngine(config={"commission": 0.0, "slippage": 0.0, "position_size": 1.0})


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestBacktestEngineInit:
    """엔진 초기화 테스트."""

    def test_default_params(self) -> None:
        """기본 파라미터가 적용되어야 한다."""
        engine = BacktestEngine()
        assert engine.config["initial_capital"] == 10000.0
        assert engine.config["commission"] == 0.001
        assert engine.config["slippage"] == 0.0005
        assert engine.config["position_size"] == 0.95

    def test_custom_params(self) -> None:
        """커스텀 설정이 기본값을 오버라이드해야 한다."""
        engine = BacktestEngine(config={"initial_capital": 50000.0, "commission": 0.002})
        assert engine.config["initial_capital"] == 50000.0
        assert engine.config["commission"] == 0.002
        assert engine.config["slippage"] == 0.0005  # default 유지

    def test_from_config_file(self, tmp_path: object) -> None:
        """YAML 설정 파일에서 backtest 섹션을 로드해야 한다."""
        config = {
            "backtest": {
                "initial_capital": 20000.0,
                "commission": 0.002,
            }
        }
        config_path = tmp_path / "config.yaml"  # type: ignore[operator]
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        engine = BacktestEngine(config_path=config_path)
        assert engine.config["initial_capital"] == 20000.0
        assert engine.config["commission"] == 0.002

    def test_invalid_position_size_raises(self) -> None:
        """position_size × (1+commission) > 1이면 ValueError가 발생해야 한다."""
        with pytest.raises(ValueError, match="capital이 음수"):
            BacktestEngine(config={"position_size": 1.0, "commission": 0.01})

    def test_boundary_position_size_ok(self) -> None:
        """position_size × (1+commission) == 1.0이면 정상 생성되어야 한다."""
        engine = BacktestEngine(config={"position_size": 1.0, "commission": 0.0})
        assert engine.config["position_size"] == 1.0


# ---------------------------------------------------------------------------
# Run — 기본 동작
# ---------------------------------------------------------------------------


class TestRun:
    """run() 메서드 기본 동작 테스트."""

    def test_basic_buy_sell_cycle(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """매수→매도 1사이클은 정확히 1건의 거래를 생성해야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        signals[2] = 1  # 매수
        signals[8] = -1  # 매도
        result = engine.run(sample_ohlcv, signals)
        assert len(result.trades) == 1

    def test_equity_curve_length(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """equity_curve 길이는 df 길이와 같아야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        result = engine.run(sample_ohlcv, signals)
        assert len(result.equity_curve) == len(sample_ohlcv)

    def test_equity_curve_index_matches_df(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """equity_curve 인덱스가 df 인덱스와 동일해야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        result = engine.run(sample_ohlcv, signals)
        pd.testing.assert_index_equal(result.equity_curve.index, sample_ohlcv.index)

    def test_result_type(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """반환 타입이 BacktestResult여야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        result = engine.run(sample_ohlcv, signals)
        assert isinstance(result, BacktestResult)

    def test_result_has_metrics(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """결과 metrics에 필수 키가 포함되어야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        signals[2] = 1
        signals[8] = -1
        result = engine.run(sample_ohlcv, signals)
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics
        assert "win_rate" in result.metrics
        assert "profit_factor" in result.metrics
        assert "buy_hold_return" in result.metrics

    def test_result_has_config(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """결과 config가 엔진 config와 동일해야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        result = engine.run(sample_ohlcv, signals)
        assert result.config == engine.config

    def test_commission_reduces_profit(self, sample_ohlcv: pd.DataFrame) -> None:
        """수수료가 높을수록 최종 자본이 낮아야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        signals[2] = 1
        signals[8] = -1

        engine_low = BacktestEngine(config={"commission": 0.001})
        engine_high = BacktestEngine(config={"commission": 0.01})

        result_low = engine_low.run(sample_ohlcv, signals)
        result_high = engine_high.run(sample_ohlcv, signals)

        assert result_low.equity_curve.iloc[-1] > result_high.equity_curve.iloc[-1]

    def test_slippage_reduces_profit(self, sample_ohlcv: pd.DataFrame) -> None:
        """슬리피지가 높을수록 최종 자본이 낮아야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        signals[2] = 1
        signals[8] = -1

        engine_low = BacktestEngine(config={"slippage": 0.0005})
        engine_high = BacktestEngine(config={"slippage": 0.01})

        result_low = engine_low.run(sample_ohlcv, signals)
        result_high = engine_high.run(sample_ohlcv, signals)

        assert result_low.equity_curve.iloc[-1] > result_high.equity_curve.iloc[-1]


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


class TestSignalHandling:
    """매매 신호 처리 테스트."""

    def test_consecutive_buy_ignored(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """이미 포지션 보유 중 매수 신호는 무시되어야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        signals[2] = 1  # 매수
        signals[3] = 1  # 중복 매수 (무시됨)
        signals[4] = 1  # 중복 매수 (무시됨)
        signals[8] = -1  # 매도
        result = engine.run(sample_ohlcv, signals)
        assert len(result.trades) == 1

    def test_sell_without_position_ignored(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """포지션 없는데 매도 신호는 무시되어야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        signals[0] = -1  # 매도 (포지션 없음, 무시)
        signals[5] = -1  # 매도 (포지션 없음, 무시)
        result = engine.run(sample_ohlcv, signals)
        assert len(result.trades) == 0

    def test_all_hold_no_trades(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """모든 신호가 0이면 거래가 없어야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        result = engine.run(sample_ohlcv, signals)
        assert len(result.trades) == 0
        assert result.equity_curve.iloc[-1] == pytest.approx(10000.0)

    def test_force_close_at_end(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """마지막까지 미청산 포지션은 강제 청산되어야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        signals[2] = 1  # 매수만 하고 매도 안 함
        result = engine.run(sample_ohlcv, signals)
        assert len(result.trades) == 1  # 강제 청산 포함

    def test_signals_as_series(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """pd.Series 신호도 정상 동작해야 한다."""
        signals = pd.Series(np.zeros(len(sample_ohlcv), dtype=int))
        signals.iloc[2] = 1
        signals.iloc[8] = -1
        result = engine.run(sample_ohlcv, signals)
        assert len(result.trades) == 1

    def test_signal_length_mismatch_raises(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """신호 길이 불일치 시 ValueError가 발생해야 한다."""
        signals = np.zeros(10, dtype=int)  # df는 20행
        with pytest.raises(ValueError, match="불일치"):
            engine.run(sample_ohlcv, signals)

    def test_missing_close_column_raises(self, engine: BacktestEngine) -> None:
        """close 컬럼 없으면 ValueError가 발생해야 한다."""
        df = pd.DataFrame({"open": [100.0], "high": [110.0]})
        with pytest.raises(ValueError, match="close"):
            engine.run(df, np.array([0]))


# ---------------------------------------------------------------------------
# Trade recording
# ---------------------------------------------------------------------------


class TestTradeRecording:
    """거래 기록 테스트."""

    def test_trade_columns(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """거래 DataFrame에 필수 컬럼이 있어야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        signals[2] = 1
        signals[8] = -1
        result = engine.run(sample_ohlcv, signals)
        expected_cols = {
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "position_size",
            "pnl",
            "pnl_abs",
            "side",
            "duration",
        }
        assert expected_cols.issubset(set(result.trades.columns))

    def test_trade_pnl_positive_on_price_increase(self, zero_cost_engine: BacktestEngine) -> None:
        """가격 상승 시 pnl이 양수여야 한다."""
        dates = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 110.0, 110.0, 110.0],
                "high": [105.0, 105.0, 115.0, 115.0, 115.0],
                "low": [95.0, 95.0, 105.0, 105.0, 105.0],
                "close": [100.0, 100.0, 110.0, 110.0, 110.0],
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        signals = np.array([0, 1, 0, -1, 0])
        result = zero_cost_engine.run(df, signals)
        assert result.trades.iloc[0]["pnl"] > 0

    def test_trade_pnl_negative_on_price_decrease(self, zero_cost_engine: BacktestEngine) -> None:
        """가격 하락 시 pnl이 음수여야 한다."""
        dates = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 90.0, 90.0, 90.0],
                "high": [105.0, 105.0, 95.0, 95.0, 95.0],
                "low": [95.0, 95.0, 85.0, 85.0, 85.0],
                "close": [100.0, 100.0, 90.0, 90.0, 90.0],
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        signals = np.array([0, 1, 0, -1, 0])
        result = zero_cost_engine.run(df, signals)
        assert result.trades.iloc[0]["pnl"] < 0

    def test_trade_entry_exit_times(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """거래의 entry/exit 시간이 신호 위치와 일치해야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        signals[2] = 1
        signals[8] = -1
        result = engine.run(sample_ohlcv, signals)
        trade = result.trades.iloc[0]
        assert trade["entry_time"] == sample_ohlcv.index[2]
        assert trade["exit_time"] == sample_ohlcv.index[8]

    def test_trade_duration_computed(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """duration이 exit_time - entry_time으로 계산되어야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        signals[2] = 1
        signals[8] = -1
        result = engine.run(sample_ohlcv, signals)
        trade = result.trades.iloc[0]
        expected_duration = trade["exit_time"] - trade["entry_time"]
        assert trade["duration"] == expected_duration

    def test_multiple_trades(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """여러 매수/매도 사이클이 올바른 거래 수를 생성해야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        signals[1] = 1
        signals[5] = -1
        signals[8] = 1
        signals[12] = -1
        signals[15] = 1
        signals[18] = -1
        result = engine.run(sample_ohlcv, signals)
        assert len(result.trades) == 3

    def test_empty_trades_have_correct_schema(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """거래 0건이어도 올바른 스키마의 DataFrame이어야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        result = engine.run(sample_ohlcv, signals)
        assert result.trades.empty
        expected_cols = {
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "position_size",
            "pnl",
            "pnl_abs",
            "side",
            "duration",
        }
        assert expected_cols.issubset(set(result.trades.columns))


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------


class TestBenchmarkComparison:
    """벤치마크 비교 테스트."""

    def test_buy_hold_in_metrics(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """metrics에 buy_hold_return이 포함되어야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        result = engine.run(sample_ohlcv, signals)
        assert "buy_hold_return" in result.metrics

    def test_excess_return_computed(self, engine: BacktestEngine, sample_ohlcv: pd.DataFrame) -> None:
        """excess_return = strategy_return - buy_hold_return이어야 한다."""
        signals = np.zeros(len(sample_ohlcv), dtype=int)
        signals[2] = 1
        signals[8] = -1
        result = engine.run(sample_ohlcv, signals)
        expected = result.metrics["strategy_return"] - result.metrics["buy_hold_return"]
        assert result.metrics["excess_return"] == pytest.approx(expected, abs=1e-10)
