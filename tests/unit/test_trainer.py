"""하이퍼파라미터 튜닝 모듈 단위 테스트.

합성 데이터로 Optuna 튜닝 로직을 검증한다.
"""

import argparse
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import optuna
import pandas as pd
import pytest
import yaml

from src.models.trainer import _XGBOOST_TUNABLE_KEYS, HyperparameterTuner

# Optuna 로그 억제
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_synthetic_df(n: int, start_date: str) -> pd.DataFrame:
    """테스트용 합성 데이터를 생성한다."""
    np.random.seed(42)
    dates = pd.date_range(start_date, periods=n, freq="1h", tz="UTC")
    features = {
        "sma_7": np.random.randn(n),
        "rsi_14": np.random.uniform(20, 80, n),
        "macd": np.random.randn(n) * 0.5,
        "bb_width": np.random.uniform(0.01, 0.05, n),
        "volume_ratio": np.random.uniform(0.5, 2.0, n),
        "atr_14": np.random.uniform(100, 500, n),
        "adx": np.random.uniform(10, 50, n),
        "return_1": np.random.randn(n) * 0.01,
        "hour_sin": np.sin(2 * np.pi * np.arange(n) / 24),
        "hour_cos": np.cos(2 * np.pi * np.arange(n) / 24),
    }
    df = pd.DataFrame(features, index=dates)
    df["target"] = np.random.choice([-1, 0, 1], size=n, p=[0.25, 0.50, 0.25])
    return df


_SAMPLE_CONFIG = {
    "general": {"random_seed": 42, "target_horizon": 4, "target_threshold": 0.005},
    "xgboost": {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 5,
        "early_stopping_rounds": 50,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
    },
    "lstm": {
        "input_size": None,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "num_classes": 3,
        "seq_length": 60,
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 100,
        "patience": 15,
        "weight_decay": 0.00001,
        "grad_clip": 1.0,
    },
    "ensemble": {"method": "logistic_regression", "weights": None},
    "sentiment": {"provider": "local"},
}


@pytest.fixture
def sample_config_yaml(tmp_path: Path) -> Path:
    """테스트용 YAML 설정 파일을 생성한다."""
    config_path = tmp_path / "model_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(_SAMPLE_CONFIG, f, default_flow_style=False, sort_keys=False)
    return config_path


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """합성 데이터로 train/val/test Parquet를 생성한다."""
    data_dir = tmp_path / "processed"
    data_dir.mkdir()

    train_df = _make_synthetic_df(300, "2024-01-01")
    val_df = _make_synthetic_df(100, "2024-03-01")
    test_df = _make_synthetic_df(100, "2024-04-01")

    train_df.to_parquet(data_dir / "train.parquet")
    val_df.to_parquet(data_dir / "val.parquet")
    test_df.to_parquet(data_dir / "test.parquet")
    return data_dir


@pytest.fixture
def tuner(sample_config_yaml: Path, sample_data_dir: Path, tmp_path: Path) -> HyperparameterTuner:
    """테스트용 HyperparameterTuner 인스턴스를 반환한다."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return HyperparameterTuner(
        config_path=sample_config_yaml,
        data_dir=sample_data_dir,
        output_dir=output_dir,
    )


class TestHyperparameterTunerInit:
    """HyperparameterTuner 초기화 테스트."""

    def test_default_init(self) -> None:
        """기본 파라미터로 초기화되어야 한다."""
        tuner = HyperparameterTuner()
        assert tuner.config_path == Path("configs/model_config.yaml")
        assert tuner.data_dir == Path("data/processed")
        assert tuner.output_dir == Path("data/models")
        assert tuner.seed == 42

    def test_custom_paths(self, tmp_path: Path) -> None:
        """커스텀 경로가 적용되어야 한다."""
        tuner = HyperparameterTuner(
            config_path=tmp_path / "config.yaml",
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "out",
            seed=123,
        )
        assert tuner.config_path == tmp_path / "config.yaml"
        assert tuner.data_dir == tmp_path / "data"
        assert tuner.output_dir == tmp_path / "out"
        assert tuner.seed == 123


class TestLoadData:
    """데이터 로드 테스트."""

    def test_load_data_returns_three_dataframes(self, tuner: HyperparameterTuner) -> None:
        """train/val/test 세 개의 DataFrame을 반환해야 한다."""
        train_df, val_df, test_df = tuner._load_data()
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert len(train_df) == 300
        assert len(val_df) == 100
        assert len(test_df) == 100


class TestXGBoostObjective:
    """XGBoost objective 함수 테스트."""

    def test_returns_float(self, tuner: HyperparameterTuner) -> None:
        """objective가 float를 반환해야 한다."""
        train_df, val_df, _ = tuner._load_data()
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: tuner._xgboost_objective(trial, train_df, val_df),
            n_trials=1,
        )
        assert isinstance(study.best_value, float)

    def test_returns_valid_range(self, tuner: HyperparameterTuner) -> None:
        """f1_weighted 값이 0~1 범위여야 한다."""
        train_df, val_df, _ = tuner._load_data()
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: tuner._xgboost_objective(trial, train_df, val_df),
            n_trials=1,
        )
        assert 0.0 <= study.best_value <= 1.0


class TestLSTMObjective:
    """LSTM objective 함수 테스트."""

    def test_returns_float(self, tuner: HyperparameterTuner) -> None:
        """objective가 float를 반환해야 한다."""
        train_df, val_df, _ = tuner._load_data()

        # 소규모 config로 오버라이드
        def fast_lstm_objective(trial: optuna.Trial) -> float:
            params = {
                "hidden_size": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "seq_length": 5,
                "batch_size": 16,
                "learning_rate": 0.01,
                "weight_decay": 1e-5,
                "grad_clip": 1.0,
                "epochs": 2,
                "patience": 5,
                "num_classes": 3,
            }
            from src.models.lstm_model import LSTMSignalModel

            model = LSTMSignalModel(config=params)
            model.train(train_df, val_df)
            metrics = model.evaluate(val_df)
            return metrics["f1_weighted"]

        study = optuna.create_study(direction="maximize")
        study.optimize(fast_lstm_objective, n_trials=1)
        assert isinstance(study.best_value, float)

    def test_returns_valid_range(self, tuner: HyperparameterTuner) -> None:
        """f1_weighted 값이 0~1 범위여야 한다."""
        train_df, val_df, _ = tuner._load_data()

        def fast_lstm_objective(trial: optuna.Trial) -> float:
            params = {
                "hidden_size": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "seq_length": 5,
                "batch_size": 16,
                "learning_rate": 0.01,
                "weight_decay": 1e-5,
                "grad_clip": 1.0,
                "epochs": 2,
                "patience": 5,
                "num_classes": 3,
            }
            from src.models.lstm_model import LSTMSignalModel

            model = LSTMSignalModel(config=params)
            model.train(train_df, val_df)
            metrics = model.evaluate(val_df)
            return metrics["f1_weighted"]

        study = optuna.create_study(direction="maximize")
        study.optimize(fast_lstm_objective, n_trials=1)
        assert 0.0 <= study.best_value <= 1.0


class TestTuneXGBoost:
    """XGBoost 튜닝 테스트."""

    def test_returns_study(self, tuner: HyperparameterTuner) -> None:
        """optuna.Study를 반환해야 한다."""
        study = tuner.tune_xgboost(n_trials=2)
        assert isinstance(study, optuna.Study)

    def test_study_has_trials(self, tuner: HyperparameterTuner) -> None:
        """study에 지정된 수의 trial이 있어야 한다."""
        study = tuner.tune_xgboost(n_trials=3)
        assert len(study.trials) == 3

    def test_best_value_is_valid(self, tuner: HyperparameterTuner) -> None:
        """best_value가 0~1 범위여야 한다."""
        study = tuner.tune_xgboost(n_trials=2)
        assert 0.0 <= study.best_value <= 1.0

    def test_best_params_contain_keys(self, tuner: HyperparameterTuner) -> None:
        """best_params에 튜닝 대상 키가 포함되어야 한다."""
        study = tuner.tune_xgboost(n_trials=2)
        for key in _XGBOOST_TUNABLE_KEYS:
            assert key in study.best_params


class TestTuneLSTM:
    """LSTM 튜닝 테스트."""

    def test_returns_study(self, tuner: HyperparameterTuner) -> None:
        """optuna.Study를 반환해야 한다."""
        # LSTM은 느리므로 objective를 모킹
        with patch.object(tuner, "_lstm_objective", return_value=0.5):
            study = tuner.tune_lstm(n_trials=2)
        assert isinstance(study, optuna.Study)

    def test_study_direction_is_maximize(self, tuner: HyperparameterTuner) -> None:
        """study 방향이 maximize여야 한다."""
        with patch.object(tuner, "_lstm_objective", return_value=0.5):
            study = tuner.tune_lstm(n_trials=1)
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE


class TestTuneAll:
    """전체 모델 튜닝 테스트."""

    def test_returns_dict(self, tuner: HyperparameterTuner) -> None:
        """dict를 반환해야 한다."""
        with (
            patch.object(tuner, "_xgboost_objective", return_value=0.5),
            patch.object(tuner, "_lstm_objective", return_value=0.5),
        ):
            studies = tuner.tune_all(xgboost_trials=1, lstm_trials=1)
        assert isinstance(studies, dict)

    def test_contains_both_models(self, tuner: HyperparameterTuner) -> None:
        """결과에 xgboost와 lstm 키가 포함되어야 한다."""
        with (
            patch.object(tuner, "_xgboost_objective", return_value=0.5),
            patch.object(tuner, "_lstm_objective", return_value=0.5),
        ):
            studies = tuner.tune_all(xgboost_trials=1, lstm_trials=1)
        assert "xgboost" in studies
        assert "lstm" in studies
        assert isinstance(studies["xgboost"], optuna.Study)
        assert isinstance(studies["lstm"], optuna.Study)


class TestUpdateConfigYaml:
    """YAML 설정 업데이트 테스트."""

    def test_updates_xgboost_section(self, tuner: HyperparameterTuner) -> None:
        """XGBoost 섹션만 업데이트되어야 한다."""
        best_params = {"max_depth": 8, "learning_rate": 0.1, "n_estimators": 300}
        tuner._update_config_yaml("xgboost", best_params)

        with open(tuner.config_path) as f:
            config = yaml.safe_load(f)

        assert config["xgboost"]["max_depth"] == 8
        assert config["xgboost"]["learning_rate"] == 0.1
        assert config["xgboost"]["n_estimators"] == 300

    def test_preserves_other_sections(self, tuner: HyperparameterTuner) -> None:
        """다른 섹션(general, lstm, ensemble)이 보존되어야 한다."""
        best_params = {"max_depth": 8}
        tuner._update_config_yaml("xgboost", best_params)

        with open(tuner.config_path) as f:
            config = yaml.safe_load(f)

        assert config["general"]["random_seed"] == 42
        assert config["lstm"]["hidden_size"] == 128
        assert config["ensemble"]["method"] == "logistic_regression"
        assert config["sentiment"]["provider"] == "local"

    def test_preserves_fixed_params(self, tuner: HyperparameterTuner) -> None:
        """고정 파라미터(objective, num_class)가 변경되지 않아야 한다."""
        best_params = {"max_depth": 8, "objective": "binary:logistic"}
        tuner._update_config_yaml("xgboost", best_params)

        with open(tuner.config_path) as f:
            config = yaml.safe_load(f)

        # objective는 튜닝 대상이 아니므로 변경되지 않아야 함
        assert config["xgboost"]["objective"] == "multi:softprob"
        assert config["xgboost"]["num_class"] == 3

    def test_updates_lstm_section(self, tuner: HyperparameterTuner) -> None:
        """LSTM 섹션이 업데이트되어야 한다."""
        best_params = {"hidden_size": 256, "dropout": 0.2, "seq_length": 90}
        tuner._update_config_yaml("lstm", best_params)

        with open(tuner.config_path) as f:
            config = yaml.safe_load(f)

        assert config["lstm"]["hidden_size"] == 256
        assert config["lstm"]["dropout"] == 0.2
        assert config["lstm"]["seq_length"] == 90
        # 고정 파라미터 보존
        assert config["lstm"]["num_classes"] == 3


class TestSaveStudy:
    """Study 저장 테스트."""

    def _create_dummy_study(self) -> optuna.Study:
        """테스트용 더미 study를 생성한다."""
        study = optuna.create_study(direction="maximize", study_name="test_study")
        study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=3)
        return study

    def test_saves_pickle_file(self, tuner: HyperparameterTuner) -> None:
        """pickle 파일이 생성되어야 한다."""
        study = self._create_dummy_study()
        pkl_path = tuner.save_study(study, "xgboost")
        assert pkl_path.exists()
        assert pkl_path.name == "xgboost_study.pkl"

    def test_saves_json_summary(self, tuner: HyperparameterTuner) -> None:
        """JSON 요약 파일이 생성되어야 한다."""
        study = self._create_dummy_study()
        tuner.save_study(study, "xgboost")
        json_path = tuner.output_dir / "xgboost_study_summary.json"
        assert json_path.exists()

        import json

        with open(json_path) as f:
            summary = json.load(f)
        assert "best_f1_weighted" in summary
        assert "best_params" in summary
        assert "top_5_trials" in summary

    def test_load_roundtrip(self, tuner: HyperparameterTuner) -> None:
        """저장/로드 후 study가 복원되어야 한다."""
        study = self._create_dummy_study()
        pkl_path = tuner.save_study(study, "lstm")

        loaded = joblib.load(pkl_path)
        assert loaded.best_value == study.best_value
        assert loaded.best_params == study.best_params
        assert len(loaded.trials) == len(study.trials)


class TestGenerateReport:
    """튜닝 리포트 생성 테스트."""

    def _create_dummy_studies(self) -> dict[str, optuna.Study]:
        """테스트용 더미 studies를 생성한다."""
        studies = {}
        for name in ("xgboost", "lstm"):
            study = optuna.create_study(direction="maximize", study_name=f"{name}_test")
            study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=3)
            studies[name] = study
        return studies

    def test_report_structure(self, tuner: HyperparameterTuner) -> None:
        """리포트에 모델별 결과가 포함되어야 한다."""
        studies = self._create_dummy_studies()
        report = tuner.generate_report(studies)
        assert "xgboost" in report
        assert "lstm" in report
        assert "best_f1_weighted" in report["xgboost"]
        assert "best_params" in report["xgboost"]

    def test_report_saved_as_json(self, tuner: HyperparameterTuner) -> None:
        """JSON 파일이 생성되어야 한다."""
        studies = self._create_dummy_studies()
        tuner.generate_report(studies)
        report_path = tuner.output_dir / "tuning_report.json"
        assert report_path.exists()

    def test_top_5_trials(self, tuner: HyperparameterTuner) -> None:
        """상위 5개 trial이 포함되어야 한다."""
        studies = self._create_dummy_studies()
        report = tuner.generate_report(studies)
        assert len(report["xgboost"]["top_5_trials"]) == 3  # n_trials=3이므로 최대 3개
        assert len(report["lstm"]["top_5_trials"]) == 3


class TestCLI:
    """CLI 엔트리포인트 테스트."""

    def test_train_mode_xgboost(self) -> None:
        """train 모드에서 XGBoost 학습 함수가 호출되어야 한다."""
        with (
            patch("src.models.trainer.argparse.ArgumentParser.parse_args") as mock_args,
            patch("src.models.xgboost_model.train_from_parquet") as mock_train,
        ):
            mock_args.return_value = argparse.Namespace(
                model="xgboost",
                mode="train",
                config="configs/model_config.yaml",
                data="data/processed",
                output="data/models",
                n_trials=50,
                timeout=None,
                save_config=False,
            )
            from src.models.trainer import tune_from_cli

            tune_from_cli()
            mock_train.assert_called_once()

    def test_tune_mode_xgboost(self) -> None:
        """tune 모드에서 튜닝이 실행되어야 한다."""
        with (
            patch("src.models.trainer.argparse.ArgumentParser.parse_args") as mock_args,
            patch.object(HyperparameterTuner, "tune_xgboost") as mock_tune,
            patch.object(HyperparameterTuner, "save_study"),
            patch.object(HyperparameterTuner, "generate_report"),
        ):
            mock_study = optuna.create_study(direction="maximize")
            mock_study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=1)
            mock_tune.return_value = mock_study

            mock_args.return_value = argparse.Namespace(
                model="xgboost",
                mode="tune",
                config="configs/model_config.yaml",
                data="data/processed",
                output="data/models",
                n_trials=10,
                timeout=None,
                save_config=False,
            )
            from src.models.trainer import tune_from_cli

            tune_from_cli()
            mock_tune.assert_called_once_with(n_trials=10, timeout=None)
