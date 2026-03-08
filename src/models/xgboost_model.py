"""XGBoost 기반 매매 신호 분류 모델.

3-class 분류(Buy=1, Hold=0, Sell=-1)를 수행하며,
학습/예측/저장/로드/평가/피처 중요도 분석 기능을 제공한다.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# 타겟 레이블 매핑: 원본(-1,0,1) ↔ XGBoost 내부(0,1,2)
_LABEL_TO_INTERNAL = {-1: 0, 0: 1, 1: 2}
_INTERNAL_TO_LABEL = {0: -1, 1: 0, 2: 1}
_LABEL_NAMES = ["Sell (-1)", "Hold (0)", "Buy (1)"]


def _encode_labels(y: np.ndarray) -> np.ndarray:
    """타겟 레이블을 XGBoost 내부 형식(0,1,2)으로 변환한다."""
    return np.vectorize(_LABEL_TO_INTERNAL.get)(y)


def _decode_labels(y: np.ndarray) -> np.ndarray:
    """XGBoost 내부 형식(0,1,2)을 원본 레이블(-1,0,1)로 변환한다."""
    return np.vectorize(_INTERNAL_TO_LABEL.get)(y)


class XGBoostSignalModel:
    """XGBoost 기반 매매 신호 모델.

    Attributes:
        model: 학습된 XGBClassifier 인스턴스.
        feature_names: 학습에 사용된 피처명 리스트.
        config: 모델 하이퍼파라미터 딕셔너리.
    """

    DEFAULT_PARAMS = {
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
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "random_state": 42,
        "verbosity": 0,
    }

    def __init__(self, config: dict | None = None, config_path: str | Path | None = None) -> None:
        """XGBoost 모델을 초기화한다.

        Args:
            config: 하이퍼파라미터 딕셔너리 (xgboost 섹션).
            config_path: YAML 설정 파일 경로 (config 미지정 시 사용).
        """
        if config is None and config_path is not None:
            full_config = self._load_yaml(config_path)
            config = full_config.get("xgboost", {})
            self._seed = full_config.get("general", {}).get("random_seed", 42)
        else:
            config = config or {}
            self._seed = config.get("random_state", 42)

        self._early_stopping_rounds = config.pop("early_stopping_rounds", 50)
        self.config = {**self.DEFAULT_PARAMS, **config, "random_state": self._seed}
        self.model: xgb.XGBClassifier | None = None
        self.feature_names: list[str] = []

    @staticmethod
    def _load_yaml(path: str | Path) -> dict:
        """YAML 설정 파일을 로드한다."""
        with open(path) as f:
            return yaml.safe_load(f)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str = "target",
    ) -> dict:
        """모델을 학습한다.

        Args:
            train_df: 학습 DataFrame (피처 + 타겟).
            val_df: 검증 DataFrame (early stopping용).
            target_col: 타겟 컬럼명.

        Returns:
            학습 이력 딕셔너리.
        """
        self.feature_names = [
            c for c in train_df.columns
            if c != target_col and not c.endswith("_raw")
        ]

        x_train = train_df[self.feature_names].values
        y_train = _encode_labels(train_df[target_col].values)
        x_val = val_df[self.feature_names].values
        y_val = _encode_labels(val_df[target_col].values)

        # 클래스 불균형 보정: inverse frequency weighting
        classes, counts = np.unique(y_train, return_counts=True)
        n_samples = len(y_train)
        n_classes = len(classes)
        class_weights = {c: n_samples / (n_classes * cnt) for c, cnt in zip(classes, counts, strict=True)}
        sample_weights = np.array([class_weights[y] for y in y_train])
        logger.info(f"클래스 가중치 적용: {class_weights}")

        self.model = xgb.XGBClassifier(**self.config)
        self.model.fit(
            x_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            verbose=False,
        )

        evals_result = self.model.evals_result()
        val_loss = evals_result["validation_1"]["mlogloss"]
        best_round = int(np.argmin(val_loss))

        # early stopping: best round 이후 patience만큼 개선 없으면 해당 라운드로 재학습
        if best_round < len(val_loss) - self._early_stopping_rounds:
            logger.info(f"Early stopping 적용: best_round={best_round}, val_mlogloss={val_loss[best_round]:.6f}")
            self.model = xgb.XGBClassifier(**{**self.config, "n_estimators": best_round + 1})
            self.model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
            evals_result = self.model.evals_result()
        else:
            logger.info(
                f"학습 완료: {len(val_loss)} rounds, best val_mlogloss={min(val_loss):.6f} (round {best_round})"
            )

        logger.info(f"피처 수: {len(self.feature_names)}, 학습: {len(x_train)}행, 검증: {len(x_val)}행")
        return evals_result

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """매매 신호를 예측한다.

        Args:
            df: 피처 DataFrame (타겟 컬럼 포함 가능, 무시됨).

        Returns:
            예측 레이블 배열 (-1, 0, 1).

        Raises:
            RuntimeError: 모델이 학습되지 않았을 때.
        """
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")

        x = df[self.feature_names].values
        internal_preds = self.model.predict(x)
        return _decode_labels(internal_preds)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """클래스별 확률을 예측한다.

        Args:
            df: 피처 DataFrame.

        Returns:
            (n_samples, 3) 확률 배열. 컬럼 순서: [Sell, Hold, Buy].

        Raises:
            RuntimeError: 모델이 학습되지 않았을 때.
        """
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")

        x = df[self.feature_names].values
        return self.model.predict_proba(x)

    def evaluate(self, df: pd.DataFrame, target_col: str = "target") -> dict:
        """모델 성능을 평가한다.

        Args:
            df: 평가 DataFrame (피처 + 타겟).
            target_col: 타겟 컬럼명.

        Returns:
            accuracy, f1_macro, f1_weighted, classification_report,
            confusion_matrix를 포함하는 딕셔너리.
        """
        y_true = df[target_col].values
        y_pred = self.predict(df)

        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        report = classification_report(y_true, y_pred, target_names=_LABEL_NAMES, output_dict=True)
        cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])

        logger.info(f"평가 결과: accuracy={accuracy:.4f}, f1_macro={f1_macro:.4f}, f1_weighted={f1_weighted:.4f}")
        logger.info(f"혼동 행렬:\n{cm}")

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

    def feature_importance(self, importance_type: str = "gain", top_n: int = 20) -> pd.DataFrame:
        """피처 중요도를 반환한다.

        Args:
            importance_type: 중요도 유형 ("gain", "weight", "cover").
            top_n: 상위 N개 피처.

        Returns:
            feature, importance 컬럼을 가진 DataFrame (내림차순).

        Raises:
            RuntimeError: 모델이 학습되지 않았을 때.
        """
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다.")

        booster = self.model.get_booster()
        scores = booster.get_score(importance_type=importance_type)

        # f0, f1, ... → 실제 피처명 매핑
        feature_map = {f"f{i}": name for i, name in enumerate(self.feature_names)}
        rows = [{"feature": feature_map.get(k, k), "importance": v} for k, v in scores.items()]

        importance_df = pd.DataFrame(rows)
        importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
        return importance_df.head(top_n)

    def save(self, model_dir: str | Path) -> Path:
        """모델과 메타데이터를 저장한다.

        Args:
            model_dir: 저장 디렉토리.

        Returns:
            모델 파일 경로.

        Raises:
            RuntimeError: 모델이 학습되지 않았을 때.
        """
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다.")

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "xgboost_model.json"
        self.model.save_model(str(model_path))

        meta = {
            "feature_names": self.feature_names,
            "config": self.config,
            "early_stopping_rounds": self._early_stopping_rounds,
        }
        meta_path = model_dir / "xgboost_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"모델 저장 완료: {model_path}")
        return model_path

    @classmethod
    def load(cls, model_dir: str | Path) -> XGBoostSignalModel:
        """저장된 모델을 로드한다.

        Args:
            model_dir: 모델 디렉토리.

        Returns:
            로드된 XGBoostSignalModel 인스턴스.
        """
        model_dir = Path(model_dir)

        meta_path = model_dir / "xgboost_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        instance = cls(config=meta["config"])
        instance.feature_names = meta["feature_names"]
        instance._early_stopping_rounds = meta.get("early_stopping_rounds", 50)

        model_path = model_dir / "xgboost_model.json"
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(str(model_path))

        logger.info(f"모델 로드 완료: {model_path} (피처 {len(instance.feature_names)}개)")
        return instance


def train_from_parquet(
    data_dir: str | Path = "data/processed",
    config_path: str | Path = "configs/model_config.yaml",
    output_dir: str | Path = "data/models",
) -> dict:
    """Parquet 파일에서 데이터를 로드하여 XGBoost 모델을 학습/평가/저장한다.

    Args:
        data_dir: train/val/test Parquet 파일 디렉토리.
        config_path: 모델 설정 YAML 경로.
        output_dir: 모델 저장 디렉토리.

    Returns:
        테스트 세트 평가 결과 딕셔너리.
    """
    from src.data.collector import load_from_parquet

    data_dir = Path(data_dir)
    train_df = load_from_parquet(data_dir / "train.parquet")
    val_df = load_from_parquet(data_dir / "val.parquet")
    test_df = load_from_parquet(data_dir / "test.parquet")

    model = XGBoostSignalModel(config_path=config_path)

    logger.info("=== XGBoost 학습 시작 ===")
    model.train(train_df, val_df)

    logger.info("=== Validation 세트 평가 ===")
    val_metrics = model.evaluate(val_df)

    logger.info("=== Test 세트 평가 ===")
    test_metrics = model.evaluate(test_df)

    logger.info("=== 피처 중요도 (Top 20) ===")
    importance = model.feature_importance(top_n=20)
    logger.info(f"\n{importance.to_string(index=False)}")

    # 과적합 확인: train vs val 성능 비교
    train_metrics = model.evaluate(train_df)
    overfit_gap = train_metrics["accuracy"] - val_metrics["accuracy"]
    logger.info(
        f"과적합 점검: train_acc={train_metrics['accuracy']:.4f}, "
        f"val_acc={val_metrics['accuracy']:.4f}, gap={overfit_gap:.4f}"
    )
    if overfit_gap > 0.10:
        logger.warning(f"과적합 의심: train-val accuracy gap이 {overfit_gap:.2%}입니다.")

    model.save(output_dir)

    # 평가 결과 JSON 저장
    results = {
        "train_accuracy": train_metrics["accuracy"],
        "val": {k: v for k, v in val_metrics.items() if k != "classification_report"},
        "test": {k: v for k, v in test_metrics.items() if k != "classification_report"},
        "overfit_gap": overfit_gap,
        "feature_importance": importance.to_dict(orient="records"),
    }
    results_path = Path(output_dir) / "xgboost_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"평가 결과 저장: {results_path}")

    return test_metrics


if __name__ == "__main__":
    train_from_parquet()
