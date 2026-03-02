"""백테스트 성능 리포트 생성기.

BacktestResult / WalkForwardResult의 metrics를 포맷팅하고,
KPI 목표 대비 달성 여부를 비교하여 텍스트/JSON/CSV로 출력한다.
"""

import json
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from src.backtest.engine import BacktestResult
from src.backtest.walk_forward import WalkForwardResult


class BacktestReportGenerator:
    """백테스트 성능 리포트 생성기.

    BacktestResult 또는 WalkForwardResult의 metrics를 포맷팅하고,
    KPI 목표 대비 달성 여부를 비교한다.

    Attributes:
        kpi_targets: KPI 목표값 딕셔너리.
    """

    DEFAULT_KPI_TARGETS: dict = {
        "sharpe_ratio": 1.5,
        "max_drawdown": -0.20,
        "win_rate": 0.55,
        "profit_factor": 1.5,
        "calmar_ratio": 1.0,
    }

    def __init__(
        self,
        kpi_targets: dict | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """리포트 생성기를 초기화한다.

        Args:
            kpi_targets: KPI 목표값 딕셔너리.
            config_path: YAML 설정 파일 경로 (kpi_targets 미지정 시 사용).
        """
        if kpi_targets is None and config_path is not None:
            full_config = self._load_yaml(config_path)
            kpi_targets = full_config.get("kpi_targets", {})

        kpi_targets = kpi_targets or {}
        self.kpi_targets: dict = {**self.DEFAULT_KPI_TARGETS, **kpi_targets}
        logger.info(f"BacktestReportGenerator 초기화: KPI targets={self.kpi_targets}")

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

    def generate(self, result: BacktestResult) -> str:
        """BacktestResult에서 텍스트 성능 요약을 생성한다.

        Args:
            result: 백테스트 결과.

        Returns:
            포맷팅된 텍스트 리포트 문자열.
        """
        m = result.metrics
        lines: list[str] = []

        # Performance Summary
        lines.append("=== Performance Summary ===")
        lines.append(f"Strategy Return:    {self._format_pct(m['strategy_return'])}")
        lines.append(f"Buy & Hold Return:  {self._format_pct(m['buy_hold_return'])}")
        lines.append(f"Excess Return:      {self._format_pct(m['excess_return'])}")
        lines.append(f"Initial Capital:    {self._format_currency(m['initial_capital'])}")
        lines.append(f"Final Capital:      {self._format_currency(m['final_capital'])}")
        lines.append(f"Sharpe Ratio:       {m['sharpe_ratio']:.2f}")
        lines.append(f"Sortino Ratio:      {m['sortino_ratio']:.2f}")
        lines.append(f"Max Drawdown:       {self._format_pct(m['max_drawdown'])}")
        lines.append(f"Calmar Ratio:       {m['calmar_ratio']:.2f}")
        lines.append("")

        # KPI Assessment
        kpi_results = self.compare_kpi(m)
        passed_count = sum(1 for r in kpi_results if r["passed"])
        total_count = len(kpi_results)

        lines.append("=== KPI Assessment ===")
        for r in kpi_results:
            actual_str = self._format_kpi_value(r["metric"], r["actual"])
            target_str = self._format_kpi_target(r["metric"], r["target"])
            status = "PASS" if r["passed"] else "FAIL"
            label = self._kpi_display_name(r["metric"])
            lines.append(f"{label:<18s}{actual_str} (target: {target_str}) {status}")
        lines.append(f"Result: {passed_count}/{total_count} PASSED")
        lines.append("")

        # Trade Statistics
        lines.append("=== Trade Statistics ===")
        lines.append(f"Total Trades:       {m['total_trades']}")
        lines.append(f"Winning Trades:     {m['winning_trades']}")
        lines.append(f"Losing Trades:      {m['losing_trades']}")
        lines.append(f"Win Rate:           {self._format_pct(m['win_rate'])}")
        lines.append(f"Profit Factor:      {m['profit_factor']:.2f}")
        lines.append(f"Best Trade:         {self._format_pct(m['best_trade'])}")
        lines.append(f"Worst Trade:        {self._format_pct(m['worst_trade'])}")
        lines.append(f"Avg Win:            {self._format_pct(m['avg_win'])}")
        lines.append(f"Avg Loss:           {self._format_pct(m['avg_loss'])}")
        duration = m.get("avg_trade_duration_hours")
        duration_str = f"{duration:.1f}h" if duration is not None else "N/A"
        lines.append(f"Avg Duration:       {duration_str}")

        report = "\n".join(lines)
        logger.info(f"리포트 생성 완료: {total_count}개 KPI 중 {passed_count}개 통과")
        return report

    def generate_walk_forward(self, result: WalkForwardResult) -> str:
        """WalkForwardResult에서 walk-forward 포함 텍스트 요약을 생성한다.

        Args:
            result: walk-forward 검증 결과.

        Returns:
            포맷팅된 텍스트 리포트 문자열.
        """
        s = result.summary
        lines: list[str] = []

        lines.append("=== Walk-Forward Summary ===")
        lines.append(f"Windows:            {s['n_windows']}")
        lines.append(f"Total Trades:       {s['total_trades']}")

        wf_metrics = [
            ("Sharpe Ratio", "sharpe_ratio", False),
            ("Sortino Ratio", "sortino_ratio", False),
            ("Max Drawdown", "max_drawdown", True),
            ("Calmar Ratio", "calmar_ratio", False),
            ("Win Rate", "win_rate", True),
            ("Profit Factor", "profit_factor", False),
            ("Strategy Return", "strategy_return", True),
        ]

        for label, key, is_pct in wf_metrics:
            mean_val = s.get(f"{key}_mean", 0.0)
            std_val = s.get(f"{key}_std", 0.0)
            if is_pct:
                lines.append(f"{label:<20s}{self._format_pct(mean_val)} +/- {self._format_pct(std_val)}")
            else:
                lines.append(f"{label:<20s}{mean_val:.2f} +/- {std_val:.2f}")
        lines.append("")

        # KPI Assessment (mean 기준)
        mean_metrics = {}
        for key in self.kpi_targets:
            mean_key = f"{key}_mean"
            if mean_key in s:
                mean_metrics[key] = s[mean_key]

        if mean_metrics:
            kpi_results = self.compare_kpi(mean_metrics)
            passed_count = sum(1 for r in kpi_results if r["passed"])
            total_count = len(kpi_results)

            lines.append("=== KPI Assessment (Walk-Forward Mean) ===")
            for r in kpi_results:
                actual_str = self._format_kpi_value(r["metric"], r["actual"])
                target_str = self._format_kpi_target(r["metric"], r["target"])
                status = "PASS" if r["passed"] else "FAIL"
                label = self._kpi_display_name(r["metric"])
                lines.append(f"{label:<18s}{actual_str} (target: {target_str}) {status}")
            lines.append(f"Result: {passed_count}/{total_count} PASSED")

        report = "\n".join(lines)
        logger.info(f"Walk-forward 리포트 생성 완료: {s['n_windows']}개 윈도우")
        return report

    def compare_kpi(self, metrics: dict) -> list[dict]:
        """KPI 목표 대비 달성 여부를 비교한다.

        Args:
            metrics: 성능 지표 딕셔너리 (BacktestResult.metrics 또는 부분 딕셔너리).

        Returns:
            KPI 비교 결과 리스트.
            각 항목: {"metric", "actual", "target", "passed"}.
        """
        results: list[dict] = []
        for metric, target in self.kpi_targets.items():
            if metric not in metrics:
                continue
            actual = metrics[metric]
            passed = actual >= target
            results.append(
                {
                    "metric": metric,
                    "actual": actual,
                    "target": target,
                    "passed": passed,
                }
            )
        return results

    def save_json(self, result: BacktestResult, path: str | Path) -> Path:
        """metrics + KPI 비교 + config를 JSON으로 저장한다.

        Args:
            result: 백테스트 결과.
            path: 저장 경로.

        Returns:
            저장된 파일 경로.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metrics": result.metrics,
            "kpi_assessment": self.compare_kpi(result.metrics),
            "config": result.config,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"JSON 리포트 저장: {output_path}")
        return output_path

    def save_walk_forward_json(self, result: WalkForwardResult, path: str | Path) -> Path:
        """walk-forward 결과를 JSON으로 저장한다.

        Args:
            result: walk-forward 검증 결과.
            path: 저장 경로.

        Returns:
            저장된 파일 경로.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mean_metrics = {}
        for key in self.kpi_targets:
            mean_key = f"{key}_mean"
            if mean_key in result.summary:
                mean_metrics[key] = result.summary[mean_key]

        windows_data = []
        for wr in result.window_results:
            windows_data.append(
                {
                    "window_id": wr.window_id,
                    "train_start": str(wr.train_start),
                    "train_end": str(wr.train_end),
                    "test_start": str(wr.test_start),
                    "test_end": str(wr.test_end),
                    "metrics": wr.backtest_result.metrics,
                }
            )

        data = {
            "summary": result.summary,
            "kpi_assessment": self.compare_kpi(mean_metrics),
            "windows": windows_data,
            "config": result.config,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Walk-forward JSON 리포트 저장: {output_path}")
        return output_path

    @staticmethod
    def save_trades_csv(trades: pd.DataFrame, path: str | Path) -> Path:
        """거래 내역을 CSV로 저장한다.

        Args:
            trades: 거래 DataFrame.
            path: 저장 경로.

        Returns:
            저장된 파일 경로.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(output_path, index=False)
        logger.info(f"거래 내역 CSV 저장: {output_path} ({len(trades)}건)")
        return output_path

    @staticmethod
    def _format_pct(value: float) -> str:
        """소수를 퍼센트 문자열로 변환한다.

        Args:
            value: 소수 값 (e.g., 0.1234).

        Returns:
            포맷팅된 문자열 (e.g., "+12.34%").
        """
        pct = value * 100
        sign = "+" if pct > 0 else ""
        return f"{sign}{pct:.2f}%"

    @staticmethod
    def _format_currency(value: float) -> str:
        """달러 포맷 문자열로 변환한다.

        Args:
            value: 달러 값.

        Returns:
            포맷팅된 문자열 (e.g., "$10,000.00").
        """
        return f"${value:,.2f}"

    @staticmethod
    def _format_kpi_value(metric: str, value: float) -> str:
        """KPI 지표 값을 포맷팅한다.

        Args:
            metric: 지표 이름.
            value: 지표 값.

        Returns:
            포맷팅된 문자열.
        """
        if metric in ("max_drawdown", "win_rate"):
            pct = value * 100
            sign = "+" if pct > 0 else ""
            return f"{sign}{pct:.2f}%"
        return f"{value:.2f}"

    @staticmethod
    def _format_kpi_target(metric: str, value: float) -> str:
        """KPI 목표값을 포맷팅한다.

        Args:
            metric: 지표 이름.
            value: 목표값.

        Returns:
            포맷팅된 문자열 (e.g., "≥1.50").
        """
        if metric in ("max_drawdown", "win_rate"):
            pct = value * 100
            sign = "+" if pct > 0 else ""
            return f"\u2265{sign}{pct:.2f}%"
        return f"\u2265{value:.2f}"

    @staticmethod
    def _kpi_display_name(metric: str) -> str:
        """KPI 지표의 표시 이름을 반환한다.

        Args:
            metric: 지표 키.

        Returns:
            표시용 이름.
        """
        names = {
            "sharpe_ratio": "Sharpe Ratio:",
            "max_drawdown": "Max Drawdown:",
            "win_rate": "Win Rate:",
            "profit_factor": "Profit Factor:",
            "calmar_ratio": "Calmar Ratio:",
        }
        return names.get(metric, f"{metric}:")
