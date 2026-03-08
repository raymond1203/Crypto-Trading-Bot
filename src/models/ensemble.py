"""앙상블 메타 모델.

XGBoost + LSTM + 감성 분석의 예측을 결합하여
최종 매매 신호를 생성하는 스태킹 앙상블 모델.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold

_LABEL_NAMES = ["Sell (-1)", "Hold (0)", "Buy (1)"]


class EnsembleModel:
    """스태킹 앙상블 메타 모델.

    각 베이스 모델(XGBoost, LSTM)의 예측 확률과
    선택적으로 감성 점수를 입력으로 받아 최종 매매 신호를 생성한다.

    지원하는 앙상블 방법:
        - ``logistic_regression``: LogisticRegression 메타 러너 (기본).
        - ``weighted_average``: 그리드 서치 기반 가중 평균.

    Attributes:
        meta_model: 학습된 LogisticRegression 인스턴스.
        weights: 가중 평균 시 각 모델의 가중치 배열.
        method: 앙상블 방법.
        base_model_names: 사용된 베이스 모델명 리스트.
    """

    def __init__(
        self,
        config: dict | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """앙상블 모델을 초기화한다.

        Args:
            config: 앙상블 하이퍼파라미터 딕셔너리.
            config_path: YAML 설정 파일 경로 (config 미지정 시 사용).
        """
        if config is None and config_path is not None:
            full_config = self._load_yaml(config_path)
            config = full_config.get("ensemble", {})
            self._seed = full_config.get("general", {}).get("random_seed", 42)
        else:
            config = config or {}
            self._seed = 42

        self.config = config
        self.method: str = config.get("method", "logistic_regression")
        self.meta_model: LogisticRegression | None = None
        self.weights: np.ndarray | None = None
        self.base_model_names: list[str] = []
        self._n_meta_features: int = 0
        self._has_sentiment: bool = False

    @staticmethod
    def _load_yaml(path: str | Path) -> dict:
        """YAML 설정 파일을 로드한다."""
        with open(path) as f:
            return yaml.safe_load(f)

    def _build_meta_features(
        self,
        base_predictions: dict[str, np.ndarray],
        sentiment_scores: np.ndarray | None = None,
    ) -> np.ndarray:
        """베이스 모델 예측과 감성 점수를 메타 피처로 결합한다.

        LogisticRegression 메타 러너의 입력을 생성한다.
        모델별 (n, 3) 확률을 수평으로 쌓고, 선택적으로 감성 점수를 추가한다.

        Args:
            base_predictions: {모델명: (n, 3) 확률 배열} 딕셔너리.
            sentiment_scores: (n,) 감성 점수 배열 (선택).

        Returns:
            (n, meta_features) 메타 피처 배열.
        """
        arrays = [base_predictions[name] for name in self.base_model_names]
        if sentiment_scores is not None:
            arrays.append(sentiment_scores.reshape(-1, 1))
        return np.hstack(arrays)

    @staticmethod
    def _sentiment_to_proba(scores: np.ndarray) -> np.ndarray:
        """감성 점수(-1~1)를 3-class 확률 분포로 변환한다.

        Args:
            scores: (n,) 감성 점수 배열.

        Returns:
            (n, 3) 확률 배열 [Sell, Hold, Buy].
        """
        n = len(scores)
        proba = np.zeros((n, 3))
        for i, s in enumerate(scores):
            if s > 0:
                proba[i] = [0.0, 1.0 - s, s]
            elif s < 0:
                proba[i] = [-s, 1.0 + s, 0.0]
            else:
                proba[i] = [0.0, 1.0, 0.0]
        return proba

    def train(
        self,
        base_predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
        sentiment_scores: np.ndarray | None = None,
    ) -> dict:
        """메타 모델을 학습한다.

        Args:
            base_predictions: {모델명: (n, 3) 확률 배열}.
            y_true: (n,) 정답 레이블 (-1, 0, 1).
            sentiment_scores: (n,) 감성 점수 (선택).

        Returns:
            학습 결과 딕셔너리.

        Raises:
            ValueError: 지원하지 않는 method.
        """
        self.base_model_names = sorted(base_predictions.keys())
        self._has_sentiment = sentiment_scores is not None

        if self.method == "logistic_regression":
            return self._train_logistic(base_predictions, y_true, sentiment_scores)
        elif self.method == "weighted_average":
            return self._train_weighted(base_predictions, y_true, sentiment_scores)
        else:
            raise ValueError(f"지원하지 않는 method: {self.method}")

    def _train_logistic(
        self,
        base_predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
        sentiment_scores: np.ndarray | None,
    ) -> dict:
        """LogisticRegression 메타 러너를 out-of-fold 방식으로 학습한다.

        K-fold cross-validation으로 OOF 예측을 생성하여 과적합을 방지하고,
        최종 메타 모델은 전체 데이터로 학습한다.
        """
        x = self._build_meta_features(base_predictions, sentiment_scores)
        self._n_meta_features = x.shape[1]

        n_folds = self.config.get("n_folds", 5)

        # Out-of-fold cross-validation
        oof_accuracy = None
        if n_folds >= 2 and len(y_true) >= n_folds:
            kf = KFold(n_splits=n_folds, shuffle=False)
            oof_preds = np.zeros(len(y_true), dtype=int)

            for train_idx, val_idx in kf.split(x):
                fold_model = LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=self._seed,
                    C=1.0,
                )
                fold_model.fit(x[train_idx], y_true[train_idx])
                oof_preds[val_idx] = fold_model.predict(x[val_idx])

            oof_accuracy = accuracy_score(y_true, oof_preds)
            logger.info(f"OOF accuracy ({n_folds}-fold): {oof_accuracy:.4f}")

        # 최종 메타 모델: 전체 데이터로 학습
        self.meta_model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=self._seed,
            C=1.0,
        )
        self.meta_model.fit(x, y_true)

        train_preds = self.meta_model.predict(x)
        in_sample_accuracy = accuracy_score(y_true, train_preds)
        contributions = self._compute_contributions()

        logger.info(f"메타 모델 학습 완료 (LogisticRegression): train_accuracy={in_sample_accuracy:.4f}")
        logger.info(f"모델별 기여도: {contributions}")

        result: dict = {
            "method": self.method,
            "train_accuracy": in_sample_accuracy,
            "n_meta_features": self._n_meta_features,
            "contributions": contributions,
            "n_folds": n_folds,
        }
        if oof_accuracy is not None:
            result["oof_accuracy"] = oof_accuracy

        return result

    def _train_weighted(
        self,
        base_predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
        sentiment_scores: np.ndarray | None,
    ) -> dict:
        """그리드 서치로 최적 가중치를 탐색하여 가중 평균 모델을 학습한다."""
        n_models = len(base_predictions)
        n_weights = n_models + (1 if sentiment_scores is not None else 0)

        best_accuracy = 0.0
        best_weights = np.ones(n_weights) / n_weights

        steps = np.arange(0.0, 1.05, 0.1)

        if n_weights == 2:
            for w0 in steps:
                w1 = round(1.0 - w0, 2)
                if w1 < 0:
                    continue
                weights = np.array([w0, w1])
                acc = self._eval_weights(weights, base_predictions, y_true, sentiment_scores)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_weights = weights.copy()

        elif n_weights == 3:
            for w0 in steps:
                for w1 in steps:
                    w2 = round(1.0 - w0 - w1, 2)
                    if w2 < 0:
                        continue
                    weights = np.array([w0, w1, w2])
                    acc = self._eval_weights(weights, base_predictions, y_true, sentiment_scores)
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_weights = weights.copy()
        else:
            best_weights = np.ones(n_weights) / n_weights

        self.weights = best_weights

        weight_dict: dict[str, float] = {}
        idx = 0
        for name in self.base_model_names:
            weight_dict[name] = round(float(self.weights[idx]), 2)
            idx += 1
        if sentiment_scores is not None:
            weight_dict["sentiment"] = round(float(self.weights[idx]), 2)

        logger.info(f"가중 평균 학습 완료: accuracy={best_accuracy:.4f}, weights={weight_dict}")

        return {
            "method": self.method,
            "train_accuracy": best_accuracy,
            "weights": weight_dict,
        }

    def _eval_weights(
        self,
        weights: np.ndarray,
        base_predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
        sentiment_scores: np.ndarray | None,
    ) -> float:
        """주어진 가중치로 가중 평균 정확도를 계산한다."""
        proba = self._compute_weighted_proba(weights, base_predictions, sentiment_scores)
        preds = np.array([-1, 0, 1])[proba.argmax(axis=1)]
        return accuracy_score(y_true, preds)

    def _compute_weighted_proba(
        self,
        weights: np.ndarray,
        base_predictions: dict[str, np.ndarray],
        sentiment_scores: np.ndarray | None,
    ) -> np.ndarray:
        """주어진 가중치로 가중 평균 확률을 계산한다."""
        idx = 0
        weighted_sum = np.zeros_like(next(iter(base_predictions.values())))

        for name in self.base_model_names:
            weighted_sum += weights[idx] * base_predictions[name]
            idx += 1

        if sentiment_scores is not None:
            sentiment_proba = self._sentiment_to_proba(sentiment_scores)
            weighted_sum += weights[idx] * sentiment_proba

        row_sums = weighted_sum.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return weighted_sum / row_sums

    def predict(
        self,
        base_predictions: dict[str, np.ndarray],
        sentiment_scores: np.ndarray | None = None,
        signal_threshold: float | None = None,
    ) -> np.ndarray:
        """최종 매매 신호를 예측한다.

        Args:
            base_predictions: {모델명: (n, 3) 확률 배열}.
            sentiment_scores: (n,) 감성 점수 (선택).
            signal_threshold: 신호 생성 확률 임계값 (None이면 config에서 로드,
                config에도 없으면 argmax 사용). Buy/Sell 확률이 threshold를
                초과하면 해당 신호를 생성하고, 둘 다 초과하면 더 높은 쪽을 선택.

        Returns:
            예측 레이블 배열 (-1, 0, 1).

        Raises:
            RuntimeError: 모델이 학습되지 않았을 때.
            ValueError: 지원하지 않는 method.
        """
        if signal_threshold is None:
            signal_threshold = self.config.get("signal_threshold")

        if signal_threshold is not None:
            # threshold 방식: base model 가중 평균 확률에 직접 적용
            # (메타 모델은 Hold로 calibrated되어 threshold가 비효과적)
            contributions = self._compute_contributions()
            weighted_proba = np.zeros_like(next(iter(base_predictions.values())))
            for model_name, proba in base_predictions.items():
                weight = contributions.get(model_name, 1.0 / len(base_predictions))
                weighted_proba += weight * proba
            return self._threshold_predict(weighted_proba, signal_threshold)

        # argmax: 메타 모델 사용
        proba = self.predict_proba(base_predictions, sentiment_scores)
        return np.array([-1, 0, 1])[proba.argmax(axis=1)]

    @staticmethod
    def _threshold_predict(proba: np.ndarray, threshold: float) -> np.ndarray:
        """확률 threshold 기반으로 매매 신호를 생성한다.

        Buy/Sell 확률이 threshold를 초과하면 해당 신호를 생성한다.
        둘 다 초과하면 더 높은 쪽을 선택하고, 둘 다 미만이면 Hold.

        Args:
            proba: (n, 3) 확률 배열 [Sell, Hold, Buy].
            threshold: 신호 생성 확률 임계값.

        Returns:
            예측 레이블 배열 (-1, 0, 1).
        """
        sell_proba = proba[:, 0]
        buy_proba = proba[:, 2]

        signals = np.zeros(len(proba), dtype=int)  # 기본 Hold
        signals[buy_proba > threshold] = 1
        signals[sell_proba > threshold] = -1

        # 둘 다 threshold 초과 시 더 높은 쪽
        both_mask = (buy_proba > threshold) & (sell_proba > threshold)
        signals[both_mask & (buy_proba >= sell_proba)] = 1
        signals[both_mask & (sell_proba > buy_proba)] = -1

        return signals

    def predict_proba(
        self,
        base_predictions: dict[str, np.ndarray],
        sentiment_scores: np.ndarray | None = None,
    ) -> np.ndarray:
        """클래스별 확률을 예측한다.

        Args:
            base_predictions: {모델명: (n, 3) 확률 배열}.
            sentiment_scores: (n,) 감성 점수 (선택).

        Returns:
            (n_samples, 3) 확률 배열. 컬럼 순서: [Sell, Hold, Buy].

        Raises:
            RuntimeError: 모델이 학습되지 않았을 때.
            ValueError: 지원하지 않는 method.
        """
        if self.method == "logistic_regression":
            if self.meta_model is None:
                raise RuntimeError("메타 모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")
            x = self._build_meta_features(base_predictions, sentiment_scores)
            return self.meta_model.predict_proba(x)

        elif self.method == "weighted_average":
            if self.weights is None:
                raise RuntimeError("가중치가 학습되지 않았습니다. train()을 먼저 호출하세요.")
            return self._compute_weighted_proba(self.weights, base_predictions, sentiment_scores)

        else:
            raise ValueError(f"지원하지 않는 method: {self.method}")

    def evaluate(
        self,
        base_predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
        sentiment_scores: np.ndarray | None = None,
    ) -> dict:
        """앙상블 성능을 평가한다.

        Args:
            base_predictions: {모델명: (n, 3) 확률 배열}.
            y_true: (n,) 정답 레이블 (-1, 0, 1).
            sentiment_scores: (n,) 감성 점수 (선택).

        Returns:
            accuracy, f1_macro, f1_weighted, classification_report,
            confusion_matrix를 포함하는 딕셔너리.
        """
        y_pred = self.predict(base_predictions, sentiment_scores)

        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        report = classification_report(y_true, y_pred, target_names=_LABEL_NAMES, output_dict=True)
        cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])

        logger.info(f"앙상블 평가: accuracy={accuracy:.4f}, f1_macro={f1_macro:.4f}")
        logger.info(f"혼동 행렬:\n{cm}")

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

    def _compute_contributions(self) -> dict[str, float]:
        """LogisticRegression 계수에서 모델별 기여도를 계산한다.

        각 베이스 모델의 3개 확률 피처에 해당하는 계수 절대값 합을
        전체 계수 절대값 합으로 나누어 상대적 기여도를 산출한다.

        Returns:
            {모델명: 기여도(0~1)} 딕셔너리.
        """
        if self.meta_model is None:
            return {}

        coefs = np.abs(self.meta_model.coef_)
        total_abs = coefs.sum()
        if total_abs == 0:
            return {name: 0.0 for name in self.base_model_names}

        contributions: dict[str, float] = {}
        idx = 0
        for name in self.base_model_names:
            model_coefs = coefs[:, idx : idx + 3].sum()
            contributions[name] = round(float(model_coefs / total_abs), 4)
            idx += 3

        if self._has_sentiment and idx < coefs.shape[1]:
            sent_coefs = coefs[:, idx:].sum()
            contributions["sentiment"] = round(float(sent_coefs / total_abs), 4)

        return contributions

    def compare_models(
        self,
        base_predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
        sentiment_scores: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """개별 모델과 앙상블의 성능을 비교한다.

        Args:
            base_predictions: {모델명: (n, 3) 확률 배열}.
            y_true: (n,) 정답 레이블 (-1, 0, 1).
            sentiment_scores: (n,) 감성 점수 (선택).

        Returns:
            model, accuracy, f1_macro 컬럼을 가진 DataFrame.
        """
        results = []

        for name in sorted(base_predictions.keys()):
            proba = base_predictions[name]
            preds = np.array([-1, 0, 1])[proba.argmax(axis=1)]
            acc = accuracy_score(y_true, preds)
            f1 = f1_score(y_true, preds, average="macro")
            results.append({"model": name, "accuracy": round(acc, 4), "f1_macro": round(f1, 4)})

        ensemble_preds = self.predict(base_predictions, sentiment_scores)
        acc = accuracy_score(y_true, ensemble_preds)
        f1 = f1_score(y_true, ensemble_preds, average="macro")
        results.append(
            {
                "model": f"ensemble ({self.method})",
                "accuracy": round(acc, 4),
                "f1_macro": round(f1, 4),
            }
        )

        df = pd.DataFrame(results)
        logger.info(f"모델 비교:\n{df.to_string(index=False)}")
        return df

    def save(self, model_dir: str | Path) -> Path:
        """앙상블 메타 모델과 메타데이터를 저장한다.

        Args:
            model_dir: 저장 디렉토리.

        Returns:
            메타데이터 파일 경로.

        Raises:
            RuntimeError: 모델이 학습되지 않았을 때.
        """
        if self.method == "logistic_regression" and self.meta_model is None:
            raise RuntimeError("메타 모델이 학습되지 않았습니다.")
        if self.method == "weighted_average" and self.weights is None:
            raise RuntimeError("가중치가 학습되지 않았습니다.")

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        meta: dict = {
            "method": self.method,
            "base_model_names": self.base_model_names,
            "n_meta_features": self._n_meta_features,
            "has_sentiment": self._has_sentiment,
            "seed": self._seed,
            "config": self.config,
        }

        if self.method == "logistic_regression":
            joblib.dump(self.meta_model, model_dir / "ensemble_meta.joblib")
        elif self.method == "weighted_average":
            meta["weights"] = self.weights.tolist()

        meta_path = model_dir / "ensemble_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"앙상블 모델 저장 완료: {model_dir}")
        return meta_path

    @classmethod
    def load(cls, model_dir: str | Path) -> EnsembleModel:
        """저장된 앙상블 모델을 로드한다.

        Args:
            model_dir: 모델 디렉토리.

        Returns:
            로드된 EnsembleModel 인스턴스.
        """
        model_dir = Path(model_dir)

        meta_path = model_dir / "ensemble_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        instance = cls(config=meta.get("config", {}))
        instance.method = meta["method"]
        instance.base_model_names = meta["base_model_names"]
        instance._n_meta_features = meta.get("n_meta_features", 0)
        instance._has_sentiment = meta.get("has_sentiment", False)
        instance._seed = meta.get("seed", 42)

        if instance.method == "logistic_regression":
            instance.meta_model = joblib.load(model_dir / "ensemble_meta.joblib")
        elif instance.method == "weighted_average":
            instance.weights = np.array(meta["weights"])

        logger.info(f"앙상블 모델 로드 완료: {model_dir} ({instance.method})")
        return instance
