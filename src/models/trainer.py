"""Optuna 기반 하이퍼파라미터 튜닝 모듈.

XGBoost, LSTM 모델의 하이퍼파라미터를 자동 탐색하고,
최적 파라미터를 YAML 설정 파일에 저장한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import optuna
import pandas as pd
import yaml
from loguru import logger

from src.data.collector import load_from_parquet
from src.models.lstm_model import LSTMSignalModel
from src.models.xgboost_model import XGBoostSignalModel

# XGBoost에서 튜닝 대상인 파라미터 키
_XGBOOST_TUNABLE_KEYS = [
    "max_depth",
    "learning_rate",
    "n_estimators",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "min_child_weight",
]

# LSTM에서 튜닝 대상인 파라미터 키
_LSTM_TUNABLE_KEYS = [
    "hidden_size",
    "num_layers",
    "dropout",
    "seq_length",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "grad_clip",
]


class HyperparameterTuner:
    """Optuna 기반 하이퍼파라미터 튜닝 오케스트레이터.

    XGBoost, LSTM 모델의 하이퍼파라미터를 자동 탐색하고,
    최적 파라미터를 YAML 설정 파일에 저장한다.

    Attributes:
        config_path: YAML 설정 파일 경로.
        data_dir: train/val/test Parquet 파일 디렉토리.
        output_dir: 튜닝 결과 저장 디렉토리.
        seed: 재현성을 위한 랜덤 시드.
    """

    def __init__(
        self,
        config_path: str | Path = "configs/model_config.yaml",
        data_dir: str | Path = "data/processed",
        output_dir: str | Path = "data/models",
        seed: int = 42,
    ) -> None:
        """튜너를 초기화한다.

        Args:
            config_path: YAML 설정 파일 경로.
            data_dir: train/val/test Parquet 파일 디렉토리.
            output_dir: 튜닝 결과 저장 디렉토리.
            seed: 재현성을 위한 랜덤 시드.
        """
        self.config_path = Path(config_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """train/val/test Parquet 파일을 로드한다.

        Returns:
            (train_df, val_df, test_df) 튜플.
        """
        train_df = load_from_parquet(self.data_dir / "train.parquet")
        val_df = load_from_parquet(self.data_dir / "val.parquet")
        test_df = load_from_parquet(self.data_dir / "test.parquet")
        logger.info(f"데이터 로드 완료: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df

    def _xgboost_objective(
        self,
        trial: optuna.Trial,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> float:
        """XGBoost 하이퍼파라미터 탐색 objective 함수.

        Args:
            trial: Optuna trial 객체.
            train_df: 학습 DataFrame.
            val_df: 검증 DataFrame.

        Returns:
            검증 세트 f1_weighted 점수.
        """
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "early_stopping_rounds": 50,
        }

        model = XGBoostSignalModel(config=params)
        model.train(train_df, val_df)
        metrics = model.evaluate(val_df)
        return float(metrics["f1_weighted"])

    def _lstm_objective(
        self,
        trial: optuna.Trial,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> float:
        """LSTM 하이퍼파라미터 탐색 objective 함수.

        Args:
            trial: Optuna trial 객체.
            train_df: 학습 DataFrame.
            val_df: 검증 DataFrame.

        Returns:
            검증 세트 f1_weighted 점수.
        """
        params = {
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "seq_length": trial.suggest_categorical("seq_length", [30, 60, 90, 120]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "grad_clip": trial.suggest_float("grad_clip", 0.5, 2.0),
            "epochs": 30,
            "patience": 10,
            "num_classes": 3,
        }

        model = LSTMSignalModel(config=params)
        model.train(train_df, val_df)
        metrics = model.evaluate(val_df)
        return float(metrics["f1_weighted"])

    def tune_xgboost(self, n_trials: int = 50, timeout: int | None = None) -> optuna.Study:
        """XGBoost 하이퍼파라미터를 튜닝한다.

        Args:
            n_trials: 탐색 횟수.
            timeout: 제한 시간(초).

        Returns:
            완료된 Optuna Study 객체.
        """
        train_df, val_df, _ = self._load_data()

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name="xgboost_tuning",
        )

        logger.info(f"=== XGBoost 튜닝 시작 (n_trials={n_trials}) ===")
        study.optimize(
            lambda trial: self._xgboost_objective(trial, train_df, val_df),
            n_trials=n_trials,
            timeout=timeout,
        )

        logger.info(f"XGBoost 튜닝 완료: best f1_weighted={study.best_value:.4f}")
        logger.info(f"최적 파라미터: {study.best_params}")
        return study

    def tune_lstm(self, n_trials: int = 30, timeout: int | None = None) -> optuna.Study:
        """LSTM 하이퍼파라미터를 튜닝한다.

        Args:
            n_trials: 탐색 횟수.
            timeout: 제한 시간(초).

        Returns:
            완료된 Optuna Study 객체.
        """
        train_df, val_df, _ = self._load_data()

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name="lstm_tuning",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )

        logger.info(f"=== LSTM 튜닝 시작 (n_trials={n_trials}) ===")
        study.optimize(
            lambda trial: self._lstm_objective(trial, train_df, val_df),
            n_trials=n_trials,
            timeout=timeout,
        )

        logger.info(f"LSTM 튜닝 완료: best f1_weighted={study.best_value:.4f}")
        logger.info(f"최적 파라미터: {study.best_params}")
        return study

    def tune_all(
        self,
        xgboost_trials: int = 50,
        lstm_trials: int = 30,
        timeout: int | None = None,
    ) -> dict[str, optuna.Study]:
        """XGBoost와 LSTM을 순차적으로 튜닝한다.

        Args:
            xgboost_trials: XGBoost 탐색 횟수.
            lstm_trials: LSTM 탐색 횟수.
            timeout: 모델별 제한 시간(초).

        Returns:
            {"xgboost": Study, "lstm": Study} 딕셔너리.
        """
        xgb_study = self.tune_xgboost(n_trials=xgboost_trials, timeout=timeout)
        lstm_study = self.tune_lstm(n_trials=lstm_trials, timeout=timeout)
        return {"xgboost": xgb_study, "lstm": lstm_study}

    def _update_config_yaml(self, model_type: str, best_params: dict) -> None:
        """최적 파라미터를 YAML 설정 파일에 저장한다.

        해당 모델 섹션의 튜닝 대상 파라미터만 업데이트하고,
        다른 섹션(general, ensemble, sentiment, features)은 보존한다.

        Args:
            model_type: 모델 유형 ("xgboost" 또는 "lstm").
            best_params: Optuna가 찾은 최적 파라미터 딕셔너리.
        """
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        tunable_keys = _XGBOOST_TUNABLE_KEYS if model_type == "xgboost" else _LSTM_TUNABLE_KEYS
        section = config.get(model_type, {})

        for key in tunable_keys:
            if key in best_params:
                section[key] = best_params[key]

        config[model_type] = section

        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        logger.info(f"{model_type} 최적 파라미터 저장 완료: {self.config_path}")

    def save_study(self, study: optuna.Study, model_type: str) -> Path:
        """Optuna Study를 저장한다.

        pickle 파일과 JSON 요약을 생성한다.

        Args:
            study: 저장할 Optuna Study 객체.
            model_type: 모델 유형 ("xgboost" 또는 "lstm").

        Returns:
            pickle 파일 경로.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # pickle 저장
        pkl_path = self.output_dir / f"{model_type}_study.pkl"
        joblib.dump(study, pkl_path)

        # JSON 요약 저장
        completed_trials = [t for t in study.trials if t.value is not None]
        sorted_trials = sorted(completed_trials, key=lambda t: t.value or 0.0, reverse=True)

        summary = {
            "study_name": study.study_name,
            "best_trial_number": study.best_trial.number,
            "best_f1_weighted": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "top_5_trials": [{"number": t.number, "value": t.value, "params": t.params} for t in sorted_trials[:5]],
        }
        json_path = self.output_dir / f"{model_type}_study_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Study 저장 완료: {pkl_path}, {json_path}")
        return pkl_path

    def generate_report(self, studies: dict[str, optuna.Study]) -> dict:
        """튜닝 결과 리포트를 생성한다.

        Args:
            studies: {"xgboost": Study, "lstm": Study} 딕셔너리.

        Returns:
            모델별 튜닝 결과를 포함하는 딕셔너리.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        report: dict = {}
        for model_type, study in studies.items():
            completed_trials = [t for t in study.trials if t.value is not None]
            sorted_trials = sorted(completed_trials, key=lambda t: t.value or 0.0, reverse=True)

            report[model_type] = {
                "best_trial_number": study.best_trial.number,
                "best_f1_weighted": study.best_value,
                "best_params": study.best_params,
                "n_trials": len(study.trials),
                "top_5_trials": [{"number": t.number, "value": t.value, "params": t.params} for t in sorted_trials[:5]],
            }

            logger.info(f"=== {model_type} 튜닝 결과 ===")
            logger.info(f"최적 f1_weighted: {study.best_value:.4f}")
            logger.info(f"최적 파라미터: {study.best_params}")

        report_path = self.output_dir / "tuning_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"튜닝 리포트 저장: {report_path}")

        return report


def tune_from_cli() -> None:
    """CLI 엔트리포인트: 모델 학습 또는 하이퍼파라미터 튜닝."""
    parser = argparse.ArgumentParser(description="CryptoSentinel 모델 학습/튜닝")
    parser.add_argument("--model", choices=["xgboost", "lstm", "all"], required=True, help="모델 유형")
    parser.add_argument("--mode", choices=["train", "tune"], default="train", help="실행 모드")
    parser.add_argument("--config", default="configs/model_config.yaml", help="설정 파일 경로")
    parser.add_argument("--data", default="data/processed", help="데이터 디렉토리")
    parser.add_argument("--output", default="data/models", help="출력 디렉토리")
    parser.add_argument("--n-trials", type=int, default=50, help="튜닝 trial 수")
    parser.add_argument("--timeout", type=int, default=None, help="튜닝 제한 시간(초)")
    parser.add_argument("--save-config", action="store_true", help="최적 파라미터를 YAML에 저장")
    args = parser.parse_args()

    if args.mode == "train":
        if args.model in ("xgboost", "all"):
            from src.models.xgboost_model import train_from_parquet as xgb_train

            xgb_train(data_dir=args.data, config_path=args.config, output_dir=args.output)

        if args.model in ("lstm", "all"):
            from src.models.lstm_model import train_from_parquet as lstm_train

            lstm_train(data_dir=args.data, config_path=args.config, output_dir=args.output)

    elif args.mode == "tune":
        tuner = HyperparameterTuner(
            config_path=args.config,
            data_dir=args.data,
            output_dir=args.output,
        )

        studies: dict[str, optuna.Study] = {}

        if args.model == "xgboost":
            study = tuner.tune_xgboost(n_trials=args.n_trials, timeout=args.timeout)
            studies["xgboost"] = study
            tuner.save_study(study, "xgboost")

        elif args.model == "lstm":
            study = tuner.tune_lstm(n_trials=args.n_trials, timeout=args.timeout)
            studies["lstm"] = study
            tuner.save_study(study, "lstm")

        elif args.model == "all":
            lstm_trials = max(args.n_trials // 2, 10)
            studies = tuner.tune_all(
                xgboost_trials=args.n_trials,
                lstm_trials=lstm_trials,
                timeout=args.timeout,
            )
            for model_type, study in studies.items():
                tuner.save_study(study, model_type)

        if args.save_config:
            for model_type, study in studies.items():
                tuner._update_config_yaml(model_type, study.best_params)

        tuner.generate_report(studies)


if __name__ == "__main__":
    tune_from_cli()
