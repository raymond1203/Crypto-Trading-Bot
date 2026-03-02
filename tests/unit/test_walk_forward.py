"""WalkForwardValidator 단위 테스트.

합성 OHLCV 데이터와 더미 signal_fn으로 walk-forward 검증 로직을 테스트한다.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest.engine import BacktestResult
from src.backtest.walk_forward import (
    WalkForwardResult,
    WalkForwardValidator,
    WindowResult,
)

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def large_ohlcv() -> pd.DataFrame:
    """200-bar 합성 OHLCV 데이터 (윈도우 테스트용)."""
    np.random.seed(42)
    n = 200
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


def dummy_signal_fn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """더미 신호 생성 콜백: 첫 bar 매수, 마지막 bar 매도."""
    signals = np.zeros(len(test_df), dtype=int)
    if len(test_df) >= 2:
        signals[0] = 1
        signals[-1] = -1
    return signals


def hold_signal_fn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """관망 신호 콜백: 모든 bar 0."""
    return np.zeros(len(test_df), dtype=int)


@pytest.fixture
def validator() -> WalkForwardValidator:
    """소규모 윈도우 설정 검증기 (train=50, test=30, step=20)."""
    return WalkForwardValidator(
        config={"train_window": 50, "test_window": 30, "step": 20},
        backtest_config={"initial_capital": 10000.0, "commission": 0.001, "slippage": 0.0005},
    )


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestWalkForwardInit:
    """검증기 초기화 테스트."""

    def test_default_params(self) -> None:
        """기본 파라미터가 적용되어야 한다."""
        v = WalkForwardValidator()
        assert v.config["train_window"] == 5000
        assert v.config["test_window"] == 1000
        assert v.config["step"] == 500

    def test_custom_params(self) -> None:
        """커스텀 설정이 기본값을 오버라이드해야 한다."""
        v = WalkForwardValidator(config={"train_window": 100, "step": 10})
        assert v.config["train_window"] == 100
        assert v.config["step"] == 10
        assert v.config["test_window"] == 1000  # default 유지

    def test_from_config_file(self, tmp_path: object) -> None:
        """YAML 설정 파일에서 walk_forward 섹션을 로드해야 한다."""
        config = {
            "backtest": {
                "initial_capital": 20000.0,
                "commission": 0.002,
                "walk_forward": {
                    "train_window": 3000,
                    "test_window": 500,
                    "step": 250,
                },
            }
        }
        config_path = tmp_path / "config.yaml"  # type: ignore[operator]
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        v = WalkForwardValidator(config_path=config_path)
        assert v.config["train_window"] == 3000
        assert v.config["test_window"] == 500
        assert v.config["step"] == 250
        # backtest_config에서 walk_forward 키가 제외되어야 한다
        assert v.backtest_config is not None
        assert "walk_forward" not in v.backtest_config
        assert v.backtest_config["initial_capital"] == 20000.0


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------


class TestGenerateWindows:
    """윈도우 생성 테스트."""

    def test_window_count(self, validator: WalkForwardValidator) -> None:
        """200-bar 데이터에서 예상 윈도우 수를 생성해야 한다.

        train=50, test=30, step=20, data=200
        start=0: 0+50+30=80 <= 200 ✓
        start=20: 20+50+30=100 <= 200 ✓
        ...
        start=120: 120+50+30=200 <= 200 ✓
        start=140: 140+50+30=220 > 200 ✗
        → 7개 윈도우
        """
        windows = validator._generate_windows(200)
        assert len(windows) == 7

    def test_window_indices_correct(self, validator: WalkForwardValidator) -> None:
        """첫 윈도우의 인덱스가 정확해야 한다."""
        windows = validator._generate_windows(200)
        tr_start, tr_end, te_start, te_end = windows[0]
        assert tr_start == 0
        assert tr_end == 50
        assert te_start == 50
        assert te_end == 80

    def test_step_offset(self, validator: WalkForwardValidator) -> None:
        """두 번째 윈도우는 step만큼 이동해야 한다."""
        windows = validator._generate_windows(200)
        tr_start, tr_end, te_start, te_end = windows[1]
        assert tr_start == 20
        assert tr_end == 70
        assert te_start == 70
        assert te_end == 100

    def test_insufficient_data_empty(self) -> None:
        """데이터 부족 시 빈 리스트를 반환해야 한다."""
        v = WalkForwardValidator(config={"train_window": 100, "test_window": 50})
        windows = v._generate_windows(100)  # 100 < 100+50
        assert windows == []


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class TestRun:
    """run() 메서드 테스트."""

    def test_result_type(self, validator: WalkForwardValidator, large_ohlcv: pd.DataFrame) -> None:
        """반환 타입이 WalkForwardResult여야 한다."""
        result = validator.run(large_ohlcv, dummy_signal_fn)
        assert isinstance(result, WalkForwardResult)

    def test_window_results_count(self, validator: WalkForwardValidator, large_ohlcv: pd.DataFrame) -> None:
        """window_results 개수가 윈도우 수와 일치해야 한다."""
        result = validator.run(large_ohlcv, dummy_signal_fn)
        expected_windows = len(validator._generate_windows(len(large_ohlcv)))
        assert len(result.window_results) == expected_windows

    def test_window_result_type(self, validator: WalkForwardValidator, large_ohlcv: pd.DataFrame) -> None:
        """각 window_result가 WindowResult 타입이어야 한다."""
        result = validator.run(large_ohlcv, dummy_signal_fn)
        for wr in result.window_results:
            assert isinstance(wr, WindowResult)
            assert isinstance(wr.backtest_result, BacktestResult)

    def test_config_preserved(self, validator: WalkForwardValidator, large_ohlcv: pd.DataFrame) -> None:
        """결과 config가 검증기 config와 동일해야 한다."""
        result = validator.run(large_ohlcv, dummy_signal_fn)
        assert result.config == validator.config

    def test_missing_close_raises(self, validator: WalkForwardValidator) -> None:
        """close 컬럼 없으면 ValueError가 발생해야 한다."""
        df = pd.DataFrame({"open": [100.0] * 200, "high": [110.0] * 200})
        with pytest.raises(ValueError, match="close"):
            validator.run(df, dummy_signal_fn)

    def test_insufficient_data_raises(self) -> None:
        """데이터가 최소 윈도우보다 짧으면 ValueError가 발생해야 한다."""
        v = WalkForwardValidator(config={"train_window": 100, "test_window": 50})
        dates = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {"close": np.ones(50), "open": np.ones(50), "high": np.ones(50), "low": np.ones(50), "volume": np.ones(50)},
            index=dates,
        )
        with pytest.raises(ValueError, match="데이터 길이"):
            v.run(df, dummy_signal_fn)

    def test_equity_curve_not_empty(self, validator: WalkForwardValidator, large_ohlcv: pd.DataFrame) -> None:
        """equity_curve가 비어있지 않아야 한다."""
        result = validator.run(large_ohlcv, dummy_signal_fn)
        assert len(result.equity_curve) > 0

    def test_hold_signal_preserves_capital(self, validator: WalkForwardValidator, large_ohlcv: pd.DataFrame) -> None:
        """관망 신호만 주면 자본이 10000으로 유지되어야 한다."""
        result = validator.run(large_ohlcv, hold_signal_fn)
        # 모든 윈도우에서 거래가 없으므로 equity = initial_capital
        for wr in result.window_results:
            assert wr.backtest_result.equity_curve.iloc[-1] == pytest.approx(10000.0)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    """요약 통계 테스트."""

    def test_summary_has_required_keys(self, validator: WalkForwardValidator, large_ohlcv: pd.DataFrame) -> None:
        """summary에 필수 키가 포함되어야 한다."""
        result = validator.run(large_ohlcv, dummy_signal_fn)
        assert "n_windows" in result.summary
        assert "total_trades" in result.summary
        assert "sharpe_ratio_mean" in result.summary
        assert "sharpe_ratio_std" in result.summary
        assert "max_drawdown_mean" in result.summary
        assert "win_rate_mean" in result.summary
        assert "profit_factor_mean" in result.summary

    def test_n_windows_matches(self, validator: WalkForwardValidator, large_ohlcv: pd.DataFrame) -> None:
        """n_windows가 실제 윈도우 수와 일치해야 한다."""
        result = validator.run(large_ohlcv, dummy_signal_fn)
        assert result.summary["n_windows"] == len(result.window_results)

    def test_total_trades_sum(self, validator: WalkForwardValidator, large_ohlcv: pd.DataFrame) -> None:
        """total_trades가 윈도우별 거래 수 합계와 일치해야 한다."""
        result = validator.run(large_ohlcv, dummy_signal_fn)
        expected_total = sum(wr.backtest_result.metrics["total_trades"] for wr in result.window_results)
        assert result.summary["total_trades"] == expected_total


# ---------------------------------------------------------------------------
# Equity curve stitching
# ---------------------------------------------------------------------------


class TestEquityCurve:
    """Equity curve 연결 테스트."""

    def test_equity_curve_continuous(self, validator: WalkForwardValidator, large_ohlcv: pd.DataFrame) -> None:
        """equity_curve에 NaN이 없어야 한다."""
        result = validator.run(large_ohlcv, dummy_signal_fn)
        assert not result.equity_curve.isna().any()

    def test_equity_curve_starts_at_initial_capital(
        self, validator: WalkForwardValidator, large_ohlcv: pd.DataFrame
    ) -> None:
        """equity_curve 시작값이 initial_capital이어야 한다."""
        result = validator.run(large_ohlcv, hold_signal_fn)
        assert result.equity_curve.iloc[0] == pytest.approx(10000.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """경계 조건 테스트."""

    def test_single_window(self, large_ohlcv: pd.DataFrame) -> None:
        """윈도우가 1개만 생성되는 설정에서도 정상 동작해야 한다."""
        v = WalkForwardValidator(
            config={"train_window": 150, "test_window": 50, "step": 200},
            backtest_config={"initial_capital": 10000.0},
        )
        result = v.run(large_ohlcv, dummy_signal_fn)
        assert len(result.window_results) == 1
        assert result.summary["n_windows"] == 1

    def test_step_larger_than_test_window(self, large_ohlcv: pd.DataFrame) -> None:
        """step > test_window일 때도 정상 동작해야 한다 (갭 있는 윈도우)."""
        v = WalkForwardValidator(
            config={"train_window": 50, "test_window": 20, "step": 40},
            backtest_config={"initial_capital": 10000.0},
        )
        result = v.run(large_ohlcv, dummy_signal_fn)
        assert len(result.window_results) > 0
        # 윈도우 간 테스트 구간이 겹치지 않아야 한다
        for i in range(1, len(result.window_results)):
            prev_end = result.window_results[i - 1].test_end
            curr_start = result.window_results[i].test_start
            assert curr_start >= prev_end

    def test_window_ids_sequential(self, validator: WalkForwardValidator, large_ohlcv: pd.DataFrame) -> None:
        """window_id가 0부터 순차적이어야 한다."""
        result = validator.run(large_ohlcv, dummy_signal_fn)
        for i, wr in enumerate(result.window_results):
            assert wr.window_id == i
