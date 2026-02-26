"""LSTM 모델 단위 테스트.

합성 데이터로 학습/예측/평가/저장/로드 로직을 검증한다.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.lstm_model import (
    LSTMPredictor,
    LSTMSignalModel,
    TimeSeriesDataset,
    _decode_labels,
    _encode_labels,
)

# 테스트에서 빠른 실행을 위한 소규모 설정
_TEST_CONFIG = {
    "hidden_size": 32,
    "num_layers": 1,
    "dropout": 0.0,
    "num_classes": 3,
    "seq_length": 10,
    "batch_size": 16,
    "learning_rate": 0.01,
    "epochs": 3,
    "patience": 5,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
}


@pytest.fixture
def sample_train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    """학습/검증용 합성 데이터를 생성한다.

    seq_length=10을 고려하여 충분한 행을 생성한다.
    """
    np.random.seed(42)
    n_train, n_val = 200, 80
    n_features = 5

    def _make_df(n: int, start_date: str) -> pd.DataFrame:
        dates = pd.date_range(start_date, periods=n, freq="1h", tz="UTC")
        data = {f"feat_{i}": np.random.randn(n) for i in range(n_features)}
        df = pd.DataFrame(data, index=dates)
        df["target"] = np.random.choice([-1, 0, 1], size=n, p=[0.25, 0.50, 0.25])
        return df

    train = _make_df(n_train, "2024-01-01")
    val = _make_df(n_val, "2024-03-01")
    return train, val


@pytest.fixture
def trained_model(sample_train_val: tuple[pd.DataFrame, pd.DataFrame]) -> LSTMSignalModel:
    """학습 완료된 모델을 반환한다."""
    train, val = sample_train_val
    model = LSTMSignalModel(config=_TEST_CONFIG)
    model.train(train, val)
    return model


class TestLabelEncoding:
    """레이블 인코딩/디코딩 테스트."""

    def test_encode_labels(self) -> None:
        """(-1,0,1) → (0,1,2) 변환이 정확해야 한다."""
        y = np.array([-1, 0, 1, 0, -1])
        encoded = _encode_labels(y)
        np.testing.assert_array_equal(encoded, [0, 1, 2, 1, 0])

    def test_decode_labels(self) -> None:
        """(0,1,2) → (-1,0,1) 변환이 정확해야 한다."""
        y = np.array([0, 1, 2, 1, 0])
        decoded = _decode_labels(y)
        np.testing.assert_array_equal(decoded, [-1, 0, 1, 0, -1])

    def test_roundtrip(self) -> None:
        """encode → decode 왕복 시 원본이 보존되어야 한다."""
        original = np.array([-1, 0, 1, 1, -1, 0])
        result = _decode_labels(_encode_labels(original))
        np.testing.assert_array_equal(result, original)


class TestTimeSeriesDataset:
    """TimeSeriesDataset 테스트."""

    def test_length(self) -> None:
        """데이터셋 길이는 n - seq_length여야 한다."""
        features = np.random.randn(100, 5)
        targets = np.random.randint(0, 3, 100)
        ds = TimeSeriesDataset(features, targets, seq_length=10)
        assert len(ds) == 90

    def test_getitem_shapes(self) -> None:
        """__getitem__이 올바른 shape을 반환해야 한다."""
        features = np.random.randn(50, 8)
        targets = np.random.randint(0, 3, 50)
        ds = TimeSeriesDataset(features, targets, seq_length=10)
        x, y = ds[0]
        assert x.shape == (10, 8)
        assert y.shape == ()

    def test_target_alignment(self) -> None:
        """타겟이 시퀀스 다음 시점의 값이어야 한다."""
        features = np.ones((20, 3))
        targets = np.arange(20)
        ds = TimeSeriesDataset(features, targets, seq_length=5)
        _, y = ds[0]
        assert y.item() == 5  # idx=0, seq_length=5 → target[5]


class TestLSTMPredictor:
    """LSTMPredictor 모듈 테스트."""

    def test_forward_shape(self) -> None:
        """forward 출력 shape이 (batch, num_classes)여야 한다."""
        model = LSTMPredictor(input_size=10, hidden_size=32, num_layers=1, dropout=0.0, num_classes=3)
        x = torch.randn(4, 20, 10)  # (batch=4, seq=20, features=10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_different_seq_lengths(self) -> None:
        """다양한 시퀀스 길이에서 동작해야 한다."""
        model = LSTMPredictor(input_size=5, hidden_size=16, num_layers=1, dropout=0.0)
        for seq_len in [10, 30, 60]:
            x = torch.randn(2, seq_len, 5)
            out = model(x)
            assert out.shape == (2, 3)

    def test_gpu_cpu_compat(self) -> None:
        """CPU에서 정상 동작해야 한다."""
        model = LSTMPredictor(input_size=5, hidden_size=16, num_layers=1, dropout=0.0)
        model = model.to("cpu")
        x = torch.randn(2, 10, 5)
        out = model(x)
        assert out.device.type == "cpu"


class TestLSTMSignalModelInit:
    """모델 초기화 테스트."""

    def test_default_config(self) -> None:
        """기본 설정으로 초기화되어야 한다."""
        model = LSTMSignalModel()
        assert model.model is None
        assert model.feature_names == []

    def test_custom_config(self) -> None:
        """커스텀 설정이 적용되어야 한다."""
        model = LSTMSignalModel(config={"hidden_size": 64, "seq_length": 30})
        assert model.config["hidden_size"] == 64
        assert model.config["seq_length"] == 30

    def test_from_config_file(self, tmp_path: object) -> None:
        """YAML 설정 파일에서 로드되어야 한다."""
        config = {
            "general": {"random_seed": 123},
            "lstm": {"hidden_size": 256, "num_layers": 3},
        }
        config_path = tmp_path / "config.yaml"  # type: ignore[operator]
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        model = LSTMSignalModel(config_path=config_path)
        assert model.config["hidden_size"] == 256
        assert model.config["num_layers"] == 3
        assert model._seed == 123


class TestTrain:
    """모델 학습 테스트."""

    def test_train_returns_history(self, sample_train_val: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """학습 후 history를 반환해야 한다."""
        train, val = sample_train_val
        model = LSTMSignalModel(config=_TEST_CONFIG)
        history = model.train(train, val)
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) > 0

    def test_model_is_built(self, trained_model: LSTMSignalModel) -> None:
        """학습 후 model이 LSTMPredictor 인스턴스여야 한다."""
        assert trained_model.model is not None
        assert isinstance(trained_model.model, LSTMPredictor)

    def test_feature_names_stored(
        self,
        trained_model: LSTMSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """학습 시 피처명이 저장되어야 한다."""
        train, _ = sample_train_val
        expected = [c for c in train.columns if c != "target"]
        assert trained_model.feature_names == expected

    def test_target_excluded_from_features(self, trained_model: LSTMSignalModel) -> None:
        """target이 피처에 포함되지 않아야 한다."""
        assert "target" not in trained_model.feature_names


class TestPredict:
    """예측 테스트."""

    def test_predict_values(
        self,
        trained_model: LSTMSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """예측값은 -1, 0, 1만 포함해야 한다."""
        _, val = sample_train_val
        preds = trained_model.predict(val)
        assert set(preds).issubset({-1, 0, 1})

    def test_predict_length(
        self,
        trained_model: LSTMSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """예측 배열 길이는 len(df) - seq_length 이하여야 한다."""
        _, val = sample_train_val
        preds = trained_model.predict(val)
        seq_length = _TEST_CONFIG["seq_length"]
        assert len(preds) <= len(val) - seq_length

    def test_predict_proba_shape(
        self,
        trained_model: LSTMSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """확률 배열은 (n, 3) shape이어야 한다."""
        _, val = sample_train_val
        proba = trained_model.predict_proba(val)
        assert proba.shape[1] == 3

    def test_predict_proba_sums_to_one(
        self,
        trained_model: LSTMSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """각 행의 확률 합이 1이어야 한다."""
        _, val = sample_train_val
        proba = trained_model.predict_proba(val)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_predict_before_train_raises(self) -> None:
        """학습 전 predict 호출 시 RuntimeError."""
        model = LSTMSignalModel(config=_TEST_CONFIG)
        dummy = pd.DataFrame({"feat_0": range(20), "target": [0] * 20})
        with pytest.raises(RuntimeError, match="학습되지 않았습니다"):
            model.predict(dummy)

    def test_predict_proba_before_train_raises(self) -> None:
        """학습 전 predict_proba 호출 시 RuntimeError."""
        model = LSTMSignalModel(config=_TEST_CONFIG)
        dummy = pd.DataFrame({"feat_0": range(20), "target": [0] * 20})
        with pytest.raises(RuntimeError, match="학습되지 않았습니다"):
            model.predict_proba(dummy)


class TestEvaluate:
    """평가 테스트."""

    def test_evaluate_returns_metrics(
        self,
        trained_model: LSTMSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """평가 결과에 핵심 지표가 포함되어야 한다."""
        _, val = sample_train_val
        metrics = trained_model.evaluate(val)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert "confusion_matrix" in metrics

    def test_accuracy_range(
        self,
        trained_model: LSTMSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """accuracy는 0~1 범위여야 한다."""
        _, val = sample_train_val
        metrics = trained_model.evaluate(val)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_confusion_matrix_shape(
        self,
        trained_model: LSTMSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """혼동 행렬은 3x3이어야 한다."""
        _, val = sample_train_val
        metrics = trained_model.evaluate(val)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)


class TestSaveLoad:
    """모델 저장/로드 테스트."""

    def test_save_creates_files(
        self,
        trained_model: LSTMSignalModel,
        tmp_path: object,
    ) -> None:
        """저장 시 .pt와 meta JSON이 생성되어야 한다."""
        model_dir = tmp_path / "model"  # type: ignore[operator]
        trained_model.save(model_dir)
        assert (model_dir / "lstm_model.pt").exists()
        assert (model_dir / "lstm_meta.json").exists()

    def test_meta_contains_feature_names(
        self,
        trained_model: LSTMSignalModel,
        tmp_path: object,
    ) -> None:
        """메타데이터에 피처명이 저장되어야 한다."""
        model_dir = tmp_path / "model"  # type: ignore[operator]
        trained_model.save(model_dir)

        with open(model_dir / "lstm_meta.json") as f:
            meta = json.load(f)
        assert meta["feature_names"] == trained_model.feature_names
        assert meta["input_size"] == len(trained_model.feature_names)

    def test_load_restores_predictions(
        self,
        trained_model: LSTMSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
        tmp_path: object,
    ) -> None:
        """로드된 모델의 예측이 원본과 동일해야 한다."""
        _, val = sample_train_val
        model_dir = tmp_path / "model"  # type: ignore[operator]
        trained_model.save(model_dir)

        loaded = LSTMSignalModel.load(model_dir)
        original_preds = trained_model.predict(val)
        loaded_preds = loaded.predict(val)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_load_restores_proba(
        self,
        trained_model: LSTMSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
        tmp_path: object,
    ) -> None:
        """로드된 모델의 확률 예측이 원본과 동일해야 한다."""
        _, val = sample_train_val
        model_dir = tmp_path / "model"  # type: ignore[operator]
        trained_model.save(model_dir)

        loaded = LSTMSignalModel.load(model_dir)
        original_proba = trained_model.predict_proba(val)
        loaded_proba = loaded.predict_proba(val)
        np.testing.assert_allclose(original_proba, loaded_proba, atol=1e-5)

    def test_save_before_train_raises(self, tmp_path: object) -> None:
        """학습 전 저장 시 RuntimeError."""
        model = LSTMSignalModel(config=_TEST_CONFIG)
        with pytest.raises(RuntimeError):
            model.save(tmp_path / "model")  # type: ignore[operator]
