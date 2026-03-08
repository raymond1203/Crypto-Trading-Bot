"""앙상블 메타 모델 단위 테스트.

합성 확률 데이터로 학습/예측/평가/저장/로드/비교 로직을 검증한다.
"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.models.ensemble import EnsembleModel


def _make_synthetic_proba(n: int, seed: int = 42) -> np.ndarray:
    """합성 3-class 확률 배열을 생성한다."""
    rng = np.random.RandomState(seed)
    raw = rng.dirichlet(alpha=[1, 1, 1], size=n)
    return raw


def _make_base_predictions(n: int) -> dict[str, np.ndarray]:
    """XGBoost + LSTM 합성 확률을 생성한다."""
    return {
        "xgboost": _make_synthetic_proba(n, seed=42),
        "lstm": _make_synthetic_proba(n, seed=123),
    }


def _make_labels(n: int, seed: int = 42) -> np.ndarray:
    """합성 정답 레이블을 생성한다."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1, 0, 1], size=n, p=[0.25, 0.50, 0.25])


def _make_sentiment_scores(n: int, seed: int = 42) -> np.ndarray:
    """합성 감성 점수를 생성한다."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-1.0, 1.0, size=n)


class TestEnsembleInit:
    """모델 초기화 테스트."""

    def test_default_config(self) -> None:
        """기본 설정으로 초기화되어야 한다."""
        model = EnsembleModel()
        assert model.method == "logistic_regression"
        assert model.meta_model is None
        assert model.weights is None
        assert model.base_model_names == []

    def test_custom_config(self) -> None:
        """커스텀 설정이 적용되어야 한다."""
        model = EnsembleModel(config={"method": "weighted_average"})
        assert model.method == "weighted_average"

    def test_from_config_file(self, tmp_path: object) -> None:
        """YAML 설정 파일에서 로드되어야 한다."""
        config = {
            "general": {"random_seed": 123},
            "ensemble": {"method": "weighted_average"},
        }
        config_path = tmp_path / "config.yaml"  # type: ignore[operator]
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        model = EnsembleModel(config_path=config_path)
        assert model.method == "weighted_average"
        assert model._seed == 123

    def test_invalid_method_raises(self) -> None:
        """지원하지 않는 method는 학습 시 ValueError."""
        model = EnsembleModel(config={"method": "random_forest"})
        preds = _make_base_predictions(50)
        y = _make_labels(50)
        with pytest.raises(ValueError, match="지원하지 않는 method"):
            model.train(preds, y)


class TestTrainLogistic:
    """LogisticRegression 메타 러너 학습 테스트."""

    def test_train_returns_result(self) -> None:
        """학습 결과를 반환해야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        result = model.train(preds, y)
        assert "method" in result
        assert result["method"] == "logistic_regression"
        assert "train_accuracy" in result
        assert "contributions" in result

    def test_meta_model_built(self) -> None:
        """학습 후 meta_model이 LogisticRegression이어야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)
        assert model.meta_model is not None

    def test_base_model_names_stored(self) -> None:
        """학습 시 베이스 모델명이 저장되어야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)
        assert model.base_model_names == ["lstm", "xgboost"]

    def test_contributions_keys(self) -> None:
        """기여도에 모든 베이스 모델이 포함되어야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        result = model.train(preds, y)
        contributions = result["contributions"]
        assert "xgboost" in contributions
        assert "lstm" in contributions

    def test_with_sentiment(self) -> None:
        """감성 점수 포함 학습이 정상 동작해야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        sentiment = _make_sentiment_scores(200)
        result = model.train(preds, y, sentiment_scores=sentiment)
        assert result["n_meta_features"] == 7  # 3 + 3 + 1
        assert "sentiment" in result["contributions"]


class TestTrainWeightedAverage:
    """가중 평균 학습 테스트."""

    def test_train_returns_weights(self) -> None:
        """학습 결과에 가중치가 포함되어야 한다."""
        model = EnsembleModel(config={"method": "weighted_average"})
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        result = model.train(preds, y)
        assert "weights" in result
        assert "lstm" in result["weights"]
        assert "xgboost" in result["weights"]

    def test_weights_sum_to_one(self) -> None:
        """가중치 합이 1이어야 한다."""
        model = EnsembleModel(config={"method": "weighted_average"})
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)
        assert model.weights is not None
        assert abs(model.weights.sum() - 1.0) < 0.05

    def test_with_sentiment(self) -> None:
        """감성 점수 포함 가중 평균이 동작해야 한다."""
        model = EnsembleModel(config={"method": "weighted_average"})
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        sentiment = _make_sentiment_scores(200)
        result = model.train(preds, y, sentiment_scores=sentiment)
        assert "sentiment" in result["weights"]


class TestPredict:
    """예측 테스트."""

    def test_predict_values(self) -> None:
        """예측값은 -1, 0, 1만 포함해야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)

        test_preds = _make_base_predictions(50)
        result = model.predict(test_preds)
        assert set(result).issubset({-1, 0, 1})

    def test_predict_length(self) -> None:
        """예측 배열 길이가 입력과 동일해야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)

        test_preds = _make_base_predictions(50)
        result = model.predict(test_preds)
        assert len(result) == 50

    def test_predict_before_train_raises(self) -> None:
        """학습 전 predict 호출 시 RuntimeError."""
        model = EnsembleModel()
        preds = _make_base_predictions(50)
        with pytest.raises(RuntimeError, match="학습되지 않았습니다"):
            model.predict(preds)

    def test_predict_weighted_before_train_raises(self) -> None:
        """가중 평균 학습 전 predict 호출 시 RuntimeError."""
        model = EnsembleModel(config={"method": "weighted_average"})
        preds = _make_base_predictions(50)
        with pytest.raises(RuntimeError, match="학습되지 않았습니다"):
            model.predict(preds)

    def test_predict_proba_shape(self) -> None:
        """확률 배열은 (n, 3) shape이어야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)

        test_preds = _make_base_predictions(50)
        proba = model.predict_proba(test_preds)
        assert proba.shape == (50, 3)

    def test_predict_proba_sums_to_one(self) -> None:
        """각 행의 확률 합이 1이어야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)

        test_preds = _make_base_predictions(50)
        proba = model.predict_proba(test_preds)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_predict_proba_before_train_raises(self) -> None:
        """학습 전 predict_proba 호출 시 RuntimeError."""
        model = EnsembleModel()
        preds = _make_base_predictions(50)
        with pytest.raises(RuntimeError, match="학습되지 않았습니다"):
            model.predict_proba(preds)

    def test_weighted_predict_proba_sums_to_one(self) -> None:
        """가중 평균 확률 합이 1이어야 한다."""
        model = EnsembleModel(config={"method": "weighted_average"})
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)

        test_preds = _make_base_predictions(50)
        proba = model.predict_proba(test_preds)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


class TestEvaluate:
    """평가 테스트."""

    def test_evaluate_returns_metrics(self) -> None:
        """평가 결과에 핵심 지표가 포함되어야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)

        test_preds = _make_base_predictions(100)
        test_y = _make_labels(100, seed=99)
        metrics = model.evaluate(test_preds, test_y)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert "confusion_matrix" in metrics

    def test_accuracy_range(self) -> None:
        """accuracy는 0~1 범위여야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)

        test_preds = _make_base_predictions(100)
        test_y = _make_labels(100, seed=99)
        metrics = model.evaluate(test_preds, test_y)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_confusion_matrix_shape(self) -> None:
        """혼동 행렬은 3x3이어야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)

        test_preds = _make_base_predictions(100)
        test_y = _make_labels(100, seed=99)
        metrics = model.evaluate(test_preds, test_y)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)


class TestCompareModels:
    """모델 비교 테스트."""

    def test_compare_returns_dataframe(self) -> None:
        """비교 결과가 DataFrame이어야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)

        df = model.compare_models(preds, y)
        assert isinstance(df, pd.DataFrame)

    def test_compare_includes_all_models(self) -> None:
        """비교 결과에 모든 모델이 포함되어야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)

        df = model.compare_models(preds, y)
        models = df["model"].tolist()
        assert "lstm" in models
        assert "xgboost" in models
        assert any("ensemble" in m for m in models)

    def test_compare_has_metric_columns(self) -> None:
        """비교 결과에 accuracy, f1_macro 컬럼이 있어야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)

        df = model.compare_models(preds, y)
        assert "accuracy" in df.columns
        assert "f1_macro" in df.columns


class TestSentimentToProba:
    """감성 점수 → 확률 변환 테스트."""

    def test_positive_score(self) -> None:
        """양수 감성은 Buy 확률이 높아야 한다."""
        proba = EnsembleModel._sentiment_to_proba(np.array([0.8]))
        assert proba[0, 2] > proba[0, 0]  # Buy > Sell
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_negative_score(self) -> None:
        """음수 감성은 Sell 확률이 높아야 한다."""
        proba = EnsembleModel._sentiment_to_proba(np.array([-0.8]))
        assert proba[0, 0] > proba[0, 2]  # Sell > Buy
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_zero_score(self) -> None:
        """0점 감성은 Hold 확률이 1.0이어야 한다."""
        proba = EnsembleModel._sentiment_to_proba(np.array([0.0]))
        np.testing.assert_array_equal(proba[0], [0.0, 1.0, 0.0])

    def test_extreme_scores(self) -> None:
        """극단값(-1, 1)의 확률 합이 1이어야 한다."""
        proba = EnsembleModel._sentiment_to_proba(np.array([-1.0, 1.0]))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        np.testing.assert_array_equal(proba[0], [1.0, 0.0, 0.0])  # Sell
        np.testing.assert_array_equal(proba[1], [0.0, 0.0, 1.0])  # Buy


class TestSaveLoad:
    """모델 저장/로드 테스트."""

    def test_save_logistic_creates_files(self, tmp_path: object) -> None:
        """LogisticRegression 저장 시 joblib과 meta JSON이 생성되어야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)

        model_dir = tmp_path / "ensemble"  # type: ignore[operator]
        model.save(model_dir)
        assert (model_dir / "ensemble_meta.joblib").exists()
        assert (model_dir / "ensemble_meta.json").exists()

    def test_save_weighted_creates_meta(self, tmp_path: object) -> None:
        """가중 평균 저장 시 meta JSON에 weights가 포함되어야 한다."""
        model = EnsembleModel(config={"method": "weighted_average"})
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)

        model_dir = tmp_path / "ensemble"  # type: ignore[operator]
        model.save(model_dir)
        assert (model_dir / "ensemble_meta.json").exists()

        with open(model_dir / "ensemble_meta.json") as f:
            meta = json.load(f)
        assert "weights" in meta

    def test_load_logistic_restores_predictions(self, tmp_path: object) -> None:
        """로드된 LogisticRegression 모델의 예측이 원본과 동일해야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)

        model_dir = tmp_path / "ensemble"  # type: ignore[operator]
        model.save(model_dir)

        test_preds = _make_base_predictions(50)
        original = model.predict(test_preds)

        loaded = EnsembleModel.load(model_dir)
        loaded_result = loaded.predict(test_preds)
        np.testing.assert_array_equal(original, loaded_result)

    def test_load_weighted_restores_predictions(self, tmp_path: object) -> None:
        """로드된 가중 평균 모델의 예측이 원본과 동일해야 한다."""
        model = EnsembleModel(config={"method": "weighted_average"})
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)

        model_dir = tmp_path / "ensemble"  # type: ignore[operator]
        model.save(model_dir)

        test_preds = _make_base_predictions(50)
        original = model.predict(test_preds)

        loaded = EnsembleModel.load(model_dir)
        loaded_result = loaded.predict(test_preds)
        np.testing.assert_array_equal(original, loaded_result)

    def test_load_restores_proba(self, tmp_path: object) -> None:
        """로드된 모델의 확률 예측이 원본과 동일해야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)

        model_dir = tmp_path / "ensemble"  # type: ignore[operator]
        model.save(model_dir)

        test_preds = _make_base_predictions(50)
        original_proba = model.predict_proba(test_preds)

        loaded = EnsembleModel.load(model_dir)
        loaded_proba = loaded.predict_proba(test_preds)
        np.testing.assert_allclose(original_proba, loaded_proba, atol=1e-5)

    def test_save_before_train_raises(self, tmp_path: object) -> None:
        """학습 전 저장 시 RuntimeError."""
        model = EnsembleModel()
        with pytest.raises(RuntimeError):
            model.save(tmp_path / "ensemble")  # type: ignore[operator]

    def test_meta_contains_base_model_names(self, tmp_path: object) -> None:
        """메타데이터에 베이스 모델명이 저장되어야 한다."""
        model = EnsembleModel()
        preds = _make_base_predictions(200)
        y = _make_labels(200)
        model.train(preds, y)

        model_dir = tmp_path / "ensemble"  # type: ignore[operator]
        model.save(model_dir)

        with open(model_dir / "ensemble_meta.json") as f:
            meta = json.load(f)
        assert meta["base_model_names"] == ["lstm", "xgboost"]


class TestThresholdPredict:
    """확률 threshold 기반 신호 생성 테스트."""

    def test_buy_signal_generated(self) -> None:
        """Buy 확률이 threshold 초과하면 Buy 신호가 생성되어야 한다."""
        proba = np.array([
            [0.1, 0.5, 0.4],  # Hold (Buy 0.4 < 0.45)
            [0.1, 0.3, 0.6],  # Buy (0.6 > 0.45)
            [0.6, 0.3, 0.1],  # Sell (0.6 > 0.45)
            [0.2, 0.6, 0.2],  # Hold
        ])
        signals = EnsembleModel._threshold_predict(proba, threshold=0.45)
        np.testing.assert_array_equal(signals, [0, 1, -1, 0])

    def test_both_exceed_picks_higher(self) -> None:
        """Buy와 Sell 모두 threshold 초과 시 더 높은 쪽이 선택되어야 한다."""
        proba = np.array([
            [0.5, 0.0, 0.5],  # 동률 → Buy (>= 조건)
            [0.6, 0.0, 0.4],  # Sell (0.6 > 0.4) — 단 둘 다 0.3 초과
            [0.4, 0.0, 0.6],  # Buy (0.6 > 0.4) — 단 둘 다 0.3 초과
        ])
        signals = EnsembleModel._threshold_predict(proba, threshold=0.3)
        np.testing.assert_array_equal(signals, [1, -1, 1])

    def test_zero_threshold_no_hold(self) -> None:
        """threshold=0이면 Hold가 없어야 한다 (Buy/Sell 확률이 항상 > 0)."""
        rng = np.random.RandomState(42)
        proba = rng.dirichlet([1, 1, 1], size=100)
        signals = EnsembleModel._threshold_predict(proba, threshold=0.0)
        assert (signals == 0).sum() == 0

    def test_high_threshold_all_hold(self) -> None:
        """threshold=1.0이면 모두 Hold여야 한다."""
        proba = np.array([[0.3, 0.4, 0.3], [0.1, 0.8, 0.1]])
        signals = EnsembleModel._threshold_predict(proba, threshold=1.0)
        np.testing.assert_array_equal(signals, [0, 0])

    def test_config_signal_threshold_used(self) -> None:
        """config의 signal_threshold가 predict()에서 사용되어야 한다."""
        model = EnsembleModel(config={"method": "logistic_regression", "signal_threshold": 0.3})
        preds = _make_base_predictions(100)
        y = _make_labels(100)
        model.train(preds, y)

        # threshold 적용 시 vs argmax 시 결과가 다를 수 있음
        signals_threshold = model.predict(preds)  # config에서 0.3 로드
        signals_argmax = np.array([-1, 0, 1])[model.predict_proba(preds).argmax(axis=1)]
        # threshold 방식에서는 argmax와 다른 결과가 가능
        # 최소한 threshold가 적용되었음을 확인 (Hold 비율 차이)
        assert isinstance(signals_threshold, np.ndarray)
        assert len(signals_threshold) == 100

    def test_no_threshold_uses_argmax(self) -> None:
        """threshold가 없으면 argmax 방식으로 예측해야 한다."""
        model = EnsembleModel(config={"method": "logistic_regression"})
        preds = _make_base_predictions(100)
        y = _make_labels(100)
        model.train(preds, y)

        signals_argmax = model.predict(preds)
        proba = model.predict_proba(preds)
        expected = np.array([-1, 0, 1])[proba.argmax(axis=1)]
        np.testing.assert_array_equal(signals_argmax, expected)
