"""BacktestReportGenerator 단위 테스트.

합성 BacktestResult와 WalkForwardResult로 리포트 생성 로직을 검증한다.
"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest.engine import BacktestResult
from src.backtest.report import BacktestReportGenerator
from src.backtest.walk_forward import WalkForwardResult, WindowResult

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


def _make_metrics(**overrides: float) -> dict:
    """19개 키를 가진 metrics dict를 생성한다."""
    base = {
        "total_trades": 10,
        "winning_trades": 6,
        "losing_trades": 4,
        "win_rate": 0.6,
        "profit_factor": 1.8,
        "sharpe_ratio": 1.8,
        "sortino_ratio": 2.1,
        "max_drawdown": -0.15,
        "calmar_ratio": 1.5,
        "strategy_return": 0.12,
        "buy_hold_return": 0.08,
        "excess_return": 0.04,
        "initial_capital": 10000.0,
        "final_capital": 11200.0,
        "best_trade": 0.05,
        "worst_trade": -0.03,
        "avg_win": 0.025,
        "avg_loss": -0.015,
        "avg_trade_duration_hours": 6.5,
    }
    base.update(overrides)
    return base


def _make_backtest_result(**metric_overrides: float) -> BacktestResult:
    """BacktestResult 객체를 생성한다."""
    metrics = _make_metrics(**metric_overrides)
    n = 20
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    equity = pd.Series(np.linspace(10000, metrics["final_capital"], n), index=dates, name="equity")
    trades = pd.DataFrame(
        {
            "entry_time": [dates[2]],
            "exit_time": [dates[8]],
            "entry_price": [40000.0],
            "exit_price": [41000.0],
            "position_size": [0.25],
            "pnl": [0.025],
            "pnl_abs": [250.0],
            "side": ["long"],
            "duration": [dates[8] - dates[2]],
        }
    )
    return BacktestResult(trades=trades, equity_curve=equity, metrics=metrics, config={"initial_capital": 10000.0})


def _make_empty_backtest_result() -> BacktestResult:
    """거래 0건인 BacktestResult를 생성한다."""
    metrics = _make_metrics(
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        profit_factor=0.0,
        best_trade=0.0,
        worst_trade=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        avg_trade_duration_hours=None,
        strategy_return=0.0,
        excess_return=-0.08,
        final_capital=10000.0,
    )
    n = 20
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    equity = pd.Series(np.full(n, 10000.0), index=dates, name="equity")
    empty_trades = pd.DataFrame(
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
    return BacktestResult(
        trades=empty_trades, equity_curve=equity, metrics=metrics, config={"initial_capital": 10000.0}
    )


def _make_walk_forward_result() -> WalkForwardResult:
    """WalkForwardResult 객체를 생성한다."""
    window_results = []
    for i in range(3):
        br = _make_backtest_result()
        dates = pd.date_range(f"2024-01-0{i + 1}", periods=5, freq="1h", tz="UTC")
        wr = WindowResult(
            window_id=i,
            train_start=dates[0],
            train_end=dates[2],
            test_start=dates[3],
            test_end=dates[4],
            backtest_result=br,
        )
        window_results.append(wr)

    summary = {
        "n_windows": 3,
        "total_trades": 30,
        "sharpe_ratio_mean": 1.8,
        "sharpe_ratio_std": 0.3,
        "sortino_ratio_mean": 2.1,
        "sortino_ratio_std": 0.4,
        "max_drawdown_mean": -0.15,
        "max_drawdown_std": 0.03,
        "calmar_ratio_mean": 1.5,
        "calmar_ratio_std": 0.2,
        "win_rate_mean": 0.6,
        "win_rate_std": 0.05,
        "profit_factor_mean": 1.8,
        "profit_factor_std": 0.3,
        "strategy_return_mean": 0.12,
        "strategy_return_std": 0.02,
    }

    equity = pd.Series([10000.0] * 15, name="equity")

    return WalkForwardResult(
        window_results=window_results,
        summary=summary,
        equity_curve=equity,
        config={"train_window": 50, "test_window": 30, "step": 20},
    )


@pytest.fixture
def generator() -> BacktestReportGenerator:
    """기본 설정 리포트 생성기."""
    return BacktestReportGenerator()


@pytest.fixture
def sample_result() -> BacktestResult:
    """전부 KPI 통과하는 BacktestResult."""
    return _make_backtest_result()


@pytest.fixture
def sample_wf_result() -> WalkForwardResult:
    """WalkForwardResult 객체."""
    return _make_walk_forward_result()


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:
    """리포트 생성기 초기화 테스트."""

    def test_default_kpi_targets(self) -> None:
        """기본 KPI 목표값이 적용되어야 한다."""
        gen = BacktestReportGenerator()
        assert gen.kpi_targets["sharpe_ratio"] == 1.5
        assert gen.kpi_targets["max_drawdown"] == -0.20
        assert gen.kpi_targets["win_rate"] == 0.55

    def test_custom_kpi_targets(self) -> None:
        """커스텀 KPI가 기본값을 오버라이드해야 한다."""
        gen = BacktestReportGenerator(kpi_targets={"sharpe_ratio": 2.0})
        assert gen.kpi_targets["sharpe_ratio"] == 2.0
        assert gen.kpi_targets["win_rate"] == 0.55  # default 유지

    def test_from_config_file(self, tmp_path: object) -> None:
        """YAML 설정 파일에서 kpi_targets를 로드해야 한다."""
        config = {"kpi_targets": {"sharpe_ratio": 2.0, "win_rate": 0.60}}
        config_path = tmp_path / "config.yaml"  # type: ignore[operator]
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        gen = BacktestReportGenerator(config_path=config_path)
        assert gen.kpi_targets["sharpe_ratio"] == 2.0
        assert gen.kpi_targets["win_rate"] == 0.60


# ---------------------------------------------------------------------------
# KPI Comparison
# ---------------------------------------------------------------------------


class TestCompareKPI:
    """KPI 비교 테스트."""

    def test_all_pass(self, generator: BacktestReportGenerator) -> None:
        """모든 KPI가 통과해야 한다."""
        metrics = _make_metrics()  # 기본값은 전부 pass
        results = generator.compare_kpi(metrics)
        assert all(r["passed"] for r in results)
        assert len(results) == 5

    def test_all_fail(self, generator: BacktestReportGenerator) -> None:
        """모든 KPI가 실패해야 한다."""
        metrics = _make_metrics(
            sharpe_ratio=0.5,
            max_drawdown=-0.30,
            win_rate=0.40,
            profit_factor=0.8,
            calmar_ratio=0.3,
        )
        results = generator.compare_kpi(metrics)
        assert all(not r["passed"] for r in results)

    def test_mixed(self, generator: BacktestReportGenerator) -> None:
        """일부 pass, 일부 fail 혼합."""
        metrics = _make_metrics(sharpe_ratio=2.0, win_rate=0.40)
        results = generator.compare_kpi(metrics)
        result_map = {r["metric"]: r["passed"] for r in results}
        assert result_map["sharpe_ratio"] is True
        assert result_map["win_rate"] is False

    def test_max_drawdown_direction(self, generator: BacktestReportGenerator) -> None:
        """max_drawdown은 actual >= target이 pass이다.

        actual=-0.15 >= target=-0.20 → pass (덜 손실)
        actual=-0.25 >= target=-0.20 → fail (더 손실)
        """
        # 덜 손실 (pass)
        metrics_pass = _make_metrics(max_drawdown=-0.15)
        results_pass = generator.compare_kpi(metrics_pass)
        mdd_result = next(r for r in results_pass if r["metric"] == "max_drawdown")
        assert mdd_result["passed"] is True

        # 더 손실 (fail)
        metrics_fail = _make_metrics(max_drawdown=-0.25)
        results_fail = generator.compare_kpi(metrics_fail)
        mdd_result = next(r for r in results_fail if r["metric"] == "max_drawdown")
        assert mdd_result["passed"] is False

    def test_missing_metric_skipped(self, generator: BacktestReportGenerator) -> None:
        """metrics에 없는 KPI 지표는 결과에서 제외되어야 한다."""
        metrics = {"sharpe_ratio": 2.0}  # 다른 지표 없음
        results = generator.compare_kpi(metrics)
        assert len(results) == 1
        assert results[0]["metric"] == "sharpe_ratio"


# ---------------------------------------------------------------------------
# Generate text report
# ---------------------------------------------------------------------------


class TestGenerate:
    """텍스트 리포트 생성 테스트."""

    def test_contains_section_headers(self, generator: BacktestReportGenerator, sample_result: BacktestResult) -> None:
        """리포트에 3개 섹션 헤더가 포함되어야 한다."""
        report = generator.generate(sample_result)
        assert "=== Performance Summary ===" in report
        assert "=== KPI Assessment ===" in report
        assert "=== Trade Statistics ===" in report

    def test_contains_kpi_results(self, generator: BacktestReportGenerator, sample_result: BacktestResult) -> None:
        """리포트에 KPI PASS/FAIL 결과가 포함되어야 한다."""
        report = generator.generate(sample_result)
        assert "PASS" in report
        assert "Result:" in report

    def test_contains_trade_stats(self, generator: BacktestReportGenerator, sample_result: BacktestResult) -> None:
        """리포트에 거래 통계가 포함되어야 한다."""
        report = generator.generate(sample_result)
        assert "Total Trades:" in report
        assert "Win Rate:" in report
        assert "Profit Factor:" in report


# ---------------------------------------------------------------------------
# Generate walk-forward report
# ---------------------------------------------------------------------------


class TestGenerateWalkForward:
    """Walk-forward 리포트 생성 테스트."""

    def test_contains_walk_forward_section(
        self, generator: BacktestReportGenerator, sample_wf_result: WalkForwardResult
    ) -> None:
        """리포트에 walk-forward 섹션이 포함되어야 한다."""
        report = generator.generate_walk_forward(sample_wf_result)
        assert "=== Walk-Forward Summary ===" in report
        assert "Windows:" in report

    def test_contains_mean_std_format(
        self, generator: BacktestReportGenerator, sample_wf_result: WalkForwardResult
    ) -> None:
        """리포트에 mean +/- std 포맷이 포함되어야 한다."""
        report = generator.generate_walk_forward(sample_wf_result)
        assert "+/-" in report


# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------


class TestSaveJson:
    """JSON 저장 테스트."""

    def test_creates_file(
        self, generator: BacktestReportGenerator, sample_result: BacktestResult, tmp_path: object
    ) -> None:
        """JSON 파일이 생성되어야 한다."""
        path = tmp_path / "report.json"  # type: ignore[operator]
        result_path = generator.save_json(sample_result, path)
        assert result_path.exists()

    def test_json_structure(
        self, generator: BacktestReportGenerator, sample_result: BacktestResult, tmp_path: object
    ) -> None:
        """JSON에 필수 키가 포함되어야 한다."""
        path = tmp_path / "report.json"  # type: ignore[operator]
        generator.save_json(sample_result, path)

        with open(path) as f:
            data = json.load(f)

        assert "metrics" in data
        assert "kpi_assessment" in data
        assert "config" in data

    def test_json_roundtrip(
        self, generator: BacktestReportGenerator, sample_result: BacktestResult, tmp_path: object
    ) -> None:
        """저장한 JSON을 다시 로드할 수 있어야 한다."""
        path = tmp_path / "report.json"  # type: ignore[operator]
        generator.save_json(sample_result, path)

        with open(path) as f:
            data = json.load(f)

        assert data["metrics"]["sharpe_ratio"] == sample_result.metrics["sharpe_ratio"]
        assert len(data["kpi_assessment"]) == 5

    def test_walk_forward_json_structure(
        self, generator: BacktestReportGenerator, sample_wf_result: WalkForwardResult, tmp_path: object
    ) -> None:
        """Walk-forward JSON에 필수 키가 포함되어야 한다."""
        path = tmp_path / "wf_report.json"  # type: ignore[operator]
        generator.save_walk_forward_json(sample_wf_result, path)

        with open(path) as f:
            data = json.load(f)

        assert "summary" in data
        assert "kpi_assessment" in data
        assert "windows" in data
        assert "config" in data
        assert len(data["windows"]) == 3


# ---------------------------------------------------------------------------
# Save trades CSV
# ---------------------------------------------------------------------------


class TestSaveTradesCsv:
    """CSV 저장 테스트."""

    def test_creates_file(self, sample_result: BacktestResult, tmp_path: object) -> None:
        """CSV 파일이 생성되어야 한다."""
        path = tmp_path / "trades.csv"  # type: ignore[operator]
        result_path = BacktestReportGenerator.save_trades_csv(sample_result.trades, path)
        assert result_path.exists()

    def test_columns_preserved(self, sample_result: BacktestResult, tmp_path: object) -> None:
        """CSV 컬럼이 원본과 일치해야 한다."""
        path = tmp_path / "trades.csv"  # type: ignore[operator]
        BacktestReportGenerator.save_trades_csv(sample_result.trades, path)
        loaded = pd.read_csv(path)
        assert set(sample_result.trades.columns).issubset(set(loaded.columns))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """경계 조건 테스트."""

    def test_zero_trades(self, generator: BacktestReportGenerator) -> None:
        """거래 0건일 때 리포트가 정상 생성되어야 한다."""
        result = _make_empty_backtest_result()
        report = generator.generate(result)
        assert "Total Trades:       0" in report
        assert "N/A" in report  # avg_trade_duration_hours=None

    def test_empty_trades_csv(self, tmp_path: object) -> None:
        """빈 trades DataFrame도 CSV 저장 가능해야 한다."""
        result = _make_empty_backtest_result()
        path = tmp_path / "empty.csv"  # type: ignore[operator]
        result_path = BacktestReportGenerator.save_trades_csv(result.trades, path)
        assert result_path.exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == 0
