"""XGBoost 모델 단위 테스트.

합성 데이터로 학습/예측/평가/저장/로드 로직을 검증한다.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.models.xgboost_model import (
    XGBoostSignalModel,
    _decode_labels,
    _encode_labels,
)


@pytest.fixture
def sample_train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    """학습/검증용 합성 데이터를 생성한다.

    3-class 분류가 가능한 수준의 피처와 타겟을 포함한다.
    """
    np.random.seed(42)
    n_train, n_val = 500, 150

    def _make_df(n: int, start_date: str) -> pd.DataFrame:
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

    train = _make_df(n_train, "2024-01-01")
    val = _make_df(n_val, "2024-03-01")
    return train, val


@pytest.fixture
def trained_model(sample_train_val: tuple[pd.DataFrame, pd.DataFrame]) -> XGBoostSignalModel:
    """학습 완료된 모델을 반환한다."""
    train, val = sample_train_val
    model = XGBoostSignalModel(config={"n_estimators": 20})
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


class TestXGBoostSignalModelInit:
    """모델 초기화 테스트."""

    def test_default_params(self) -> None:
        """기본 하이퍼파라미터가 적용되어야 한다."""
        model = XGBoostSignalModel()
        assert model.config["objective"] == "multi:softprob"
        assert model.config["num_class"] == 3
        assert model.config["max_depth"] == 6

    def test_custom_params_override(self) -> None:
        """커스텀 파라미터가 기본값을 오버라이드해야 한다."""
        model = XGBoostSignalModel(config={"max_depth": 10, "learning_rate": 0.1})
        assert model.config["max_depth"] == 10
        assert model.config["learning_rate"] == 0.1
        # 나머지는 기본값 유지
        assert model.config["subsample"] == 0.8

    def test_from_config_file(self, tmp_path: object) -> None:
        """YAML 설정 파일에서 로드되어야 한다."""
        config = {
            "general": {"random_seed": 123},
            "xgboost": {"max_depth": 8, "n_estimators": 100},
        }
        config_path = tmp_path / "config.yaml"  # type: ignore[operator]
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        model = XGBoostSignalModel(config_path=config_path)
        assert model.config["max_depth"] == 8
        assert model.config["n_estimators"] == 100
        assert model.config["random_state"] == 123

    def test_not_trained_initially(self) -> None:
        """초기 상태에서 model은 None이어야 한다."""
        model = XGBoostSignalModel()
        assert model.model is None
        assert model.feature_names == []


class TestTrain:
    """모델 학습 테스트."""

    def test_train_returns_evals_result(self, sample_train_val: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """학습 후 evals_result를 반환해야 한다."""
        train, val = sample_train_val
        model = XGBoostSignalModel(config={"n_estimators": 10})
        result = model.train(train, val)
        assert "validation_0" in result or "validation_1" in result

    def test_model_is_fitted(self, trained_model: XGBoostSignalModel) -> None:
        """학습 후 model이 XGBClassifier 인스턴스여야 한다."""
        assert trained_model.model is not None

    def test_feature_names_stored(
        self,
        trained_model: XGBoostSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """학습 시 피처명이 저장되어야 한다."""
        train, _ = sample_train_val
        expected = [c for c in train.columns if c != "target"]
        assert trained_model.feature_names == expected

    def test_target_excluded_from_features(self, trained_model: XGBoostSignalModel) -> None:
        """target이 피처에 포함되지 않아야 한다."""
        assert "target" not in trained_model.feature_names


class TestPredict:
    """예측 테스트."""

    def test_predict_shape(
        self,
        trained_model: XGBoostSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """예측 배열 길이가 입력과 같아야 한다."""
        _, val = sample_train_val
        preds = trained_model.predict(val)
        assert len(preds) == len(val)

    def test_predict_values(
        self,
        trained_model: XGBoostSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """예측값은 -1, 0, 1만 포함해야 한다."""
        _, val = sample_train_val
        preds = trained_model.predict(val)
        assert set(preds).issubset({-1, 0, 1})

    def test_predict_proba_shape(
        self,
        trained_model: XGBoostSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """확률 배열은 (n, 3) shape이어야 한다."""
        _, val = sample_train_val
        proba = trained_model.predict_proba(val)
        assert proba.shape == (len(val), 3)

    def test_predict_proba_sums_to_one(
        self,
        trained_model: XGBoostSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """각 행의 확률 합이 1이어야 한다."""
        _, val = sample_train_val
        proba = trained_model.predict_proba(val)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_predict_before_train_raises(self) -> None:
        """학습 전 predict 호출 시 RuntimeError."""
        model = XGBoostSignalModel()
        dummy = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(RuntimeError, match="학습되지 않았습니다"):
            model.predict(dummy)

    def test_predict_proba_before_train_raises(self) -> None:
        """학습 전 predict_proba 호출 시 RuntimeError."""
        model = XGBoostSignalModel()
        dummy = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(RuntimeError, match="학습되지 않았습니다"):
            model.predict_proba(dummy)


class TestEvaluate:
    """평가 테스트."""

    def test_evaluate_returns_metrics(
        self,
        trained_model: XGBoostSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """평가 결과에 핵심 지표가 포함되어야 한다."""
        _, val = sample_train_val
        metrics = trained_model.evaluate(val)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert "classification_report" in metrics
        assert "confusion_matrix" in metrics

    def test_accuracy_range(
        self,
        trained_model: XGBoostSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """accuracy는 0~1 범위여야 한다."""
        _, val = sample_train_val
        metrics = trained_model.evaluate(val)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_confusion_matrix_shape(
        self,
        trained_model: XGBoostSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """혼동 행렬은 3x3이어야 한다."""
        _, val = sample_train_val
        metrics = trained_model.evaluate(val)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)


class TestFeatureImportance:
    """피처 중요도 테스트."""

    def test_returns_dataframe(self, trained_model: XGBoostSignalModel) -> None:
        """피처 중요도가 DataFrame으로 반환되어야 한다."""
        importance = trained_model.feature_importance()
        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_sorted_descending(self, trained_model: XGBoostSignalModel) -> None:
        """중요도 내림차순으로 정렬되어야 한다."""
        importance = trained_model.feature_importance()
        values = importance["importance"].tolist()
        assert values == sorted(values, reverse=True)

    def test_top_n(self, trained_model: XGBoostSignalModel) -> None:
        """top_n으로 반환 개수를 제한할 수 있어야 한다."""
        importance = trained_model.feature_importance(top_n=5)
        assert len(importance) <= 5

    def test_feature_names_mapped(self, trained_model: XGBoostSignalModel) -> None:
        """피처명이 실제 이름으로 매핑되어야 한다."""
        importance = trained_model.feature_importance()
        # f0, f1 같은 내부 이름이 아닌 실제 이름이어야 함
        for name in importance["feature"]:
            assert not name.startswith("f") or name in trained_model.feature_names

    def test_before_train_raises(self) -> None:
        """학습 전 호출 시 RuntimeError."""
        model = XGBoostSignalModel()
        with pytest.raises(RuntimeError):
            model.feature_importance()


class TestSaveLoad:
    """모델 저장/로드 테스트."""

    def test_save_creates_files(
        self,
        trained_model: XGBoostSignalModel,
        tmp_path: object,
    ) -> None:
        """저장 시 model JSON과 meta JSON이 생성되어야 한다."""
        model_dir = tmp_path / "model"  # type: ignore[operator]
        trained_model.save(model_dir)
        assert (model_dir / "xgboost_model.json").exists()
        assert (model_dir / "xgboost_meta.json").exists()

    def test_meta_contains_feature_names(
        self,
        trained_model: XGBoostSignalModel,
        tmp_path: object,
    ) -> None:
        """메타데이터에 피처명이 저장되어야 한다."""
        model_dir = tmp_path / "model"  # type: ignore[operator]
        trained_model.save(model_dir)

        with open(model_dir / "xgboost_meta.json") as f:
            meta = json.load(f)
        assert meta["feature_names"] == trained_model.feature_names

    def test_load_restores_model(
        self,
        trained_model: XGBoostSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
        tmp_path: object,
    ) -> None:
        """로드된 모델의 예측이 원본과 동일해야 한다."""
        _, val = sample_train_val
        model_dir = tmp_path / "model"  # type: ignore[operator]
        trained_model.save(model_dir)

        loaded = XGBoostSignalModel.load(model_dir)
        original_preds = trained_model.predict(val)
        loaded_preds = loaded.predict(val)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_load_restores_proba(
        self,
        trained_model: XGBoostSignalModel,
        sample_train_val: tuple[pd.DataFrame, pd.DataFrame],
        tmp_path: object,
    ) -> None:
        """로드된 모델의 확률 예측이 원본과 동일해야 한다."""
        _, val = sample_train_val
        model_dir = tmp_path / "model"  # type: ignore[operator]
        trained_model.save(model_dir)

        loaded = XGBoostSignalModel.load(model_dir)
        original_proba = trained_model.predict_proba(val)
        loaded_proba = loaded.predict_proba(val)
        np.testing.assert_allclose(original_proba, loaded_proba, atol=1e-6)

    def test_save_before_train_raises(self, tmp_path: object) -> None:
        """학습 전 저장 시 RuntimeError."""
        model = XGBoostSignalModel()
        with pytest.raises(RuntimeError):
            model.save(tmp_path / "model")  # type: ignore[operator]
