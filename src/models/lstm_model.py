"""LSTM 기반 가격 방향 예측 모델.

시계열 시퀀스에서 3-class 분류(Buy=1, Hold=0, Sell=-1)를 수행한다.
Self-Attention + LSTM 아키텍처로 시간적 패턴을 학습한다.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

# 타겟 레이블 매핑: 원본(-1,0,1) ↔ 내부(0,1,2)
_LABEL_TO_INTERNAL = {-1: 0, 0: 1, 1: 2}
_INTERNAL_TO_LABEL = {0: -1, 1: 0, 2: 1}
_LABEL_NAMES = ["Sell (-1)", "Hold (0)", "Buy (1)"]


def _encode_labels(y: np.ndarray) -> np.ndarray:
    """타겟 레이블을 내부 형식(0,1,2)으로 변환한다."""
    return np.vectorize(_LABEL_TO_INTERNAL.get)(y)


def _decode_labels(y: np.ndarray) -> np.ndarray:
    """내부 형식(0,1,2)을 원본 레이블(-1,0,1)로 변환한다."""
    return np.vectorize(_INTERNAL_TO_LABEL.get)(y)


class TimeSeriesDataset(Dataset):
    """시계열 슬라이딩 윈도우 데이터셋.

    Args:
        features: (n_samples, n_features) 피처 배열.
        targets: (n_samples,) 타겟 배열 (내부 형식 0,1,2).
        seq_length: 시퀀스 길이.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_length: int = 60) -> None:
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.features) - self.seq_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.seq_length]
        y = self.targets[idx + self.seq_length - 1]
        return x, y


class LSTMPredictor(nn.Module):
    """LSTM + Self-Attention 기반 가격 방향 예측 모델.

    Args:
        input_size: 피처 수.
        hidden_size: LSTM hidden state 크기.
        num_layers: LSTM 레이어 수.
        dropout: 드롭아웃 비율.
        num_classes: 분류 클래스 수.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파.

        Args:
            x: (batch, seq_len, features) 입력 텐서.

        Returns:
            (batch, num_classes) 로짓 텐서.
        """
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = attn_out[:, -1, :]
        return self.fc(out)


class LSTMSignalModel:
    """LSTM 매매 신호 모델 래퍼.

    학습/예측/평가/저장/로드를 관리한다.

    Attributes:
        model: LSTMPredictor 인스턴스.
        feature_names: 학습에 사용된 피처명 리스트.
        config: 하이퍼파라미터 딕셔너리.
        device: 연산 디바이스 (cpu/cuda).
    """

    def __init__(self, config: dict | None = None, config_path: str | Path | None = None) -> None:
        """LSTM 모델을 초기화한다.

        Args:
            config: 하이퍼파라미터 딕셔너리 (lstm 섹션).
            config_path: YAML 설정 파일 경로.
        """
        if config is None and config_path is not None:
            full_config = self._load_yaml(config_path)
            config = full_config.get("lstm", {})
            self._seed = full_config.get("general", {}).get("random_seed", 42)
        else:
            config = config or {}
            self._seed = 42

        self.config = config
        self.model: LSTMPredictor | None = None
        self.feature_names: list[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed()

    @staticmethod
    def _load_yaml(path: str | Path) -> dict:
        """YAML 설정 파일을 로드한다."""
        with open(path) as f:
            return yaml.safe_load(f)

    def _set_seed(self) -> None:
        """재현성을 위해 시드를 고정한다."""
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)

    def _build_model(self, input_size: int) -> LSTMPredictor:
        """설정값으로 LSTMPredictor를 생성한다."""
        return LSTMPredictor(
            input_size=input_size,
            hidden_size=self.config.get("hidden_size", 128),
            num_layers=self.config.get("num_layers", 2),
            dropout=self.config.get("dropout", 0.3),
            num_classes=self.config.get("num_classes", 3),
        ).to(self.device)

    def _make_dataloader(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
        shuffle: bool = False,
    ) -> DataLoader:
        """DataFrame에서 DataLoader를 생성한다."""
        feature_cols = [c for c in df.columns if c != target_col]
        features = df[feature_cols].values
        targets = _encode_labels(df[target_col].values)
        seq_length = self.config.get("seq_length", 60)
        batch_size = self.config.get("batch_size", 64)

        dataset = TimeSeriesDataset(features, targets, seq_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str = "target",
    ) -> dict[str, list[float]]:
        """모델을 학습한다.

        Args:
            train_df: 학습 DataFrame (피처 + 타겟).
            val_df: 검증 DataFrame (early stopping용).
            target_col: 타겟 컬럼명.

        Returns:
            학습 이력 {"train_loss": [...], "val_loss": [...]}.
        """
        self.feature_names = [c for c in train_df.columns if c != target_col]
        input_size = len(self.feature_names)
        self.model = self._build_model(input_size)

        train_loader = self._make_dataloader(train_df, target_col, shuffle=False)
        val_loader = self._make_dataloader(val_df, target_col, shuffle=False)

        epochs = self.config.get("epochs", 100)
        lr = self.config.get("learning_rate", 0.001)
        weight_decay = self.config.get("weight_decay", 1e-5)
        grad_clip = self.config.get("grad_clip", 1.0)
        patience = self.config.get("patience", 15)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # 클래스 불균형 보정: inverse frequency weighting
        y_all = train_df[target_col].values.astype(int)
        # 타겟 레이블 (-1,0,1) → 내부 인덱스 (0,1,2)
        y_internal = y_all + 1
        classes, counts = np.unique(y_internal, return_counts=True)
        n_samples = len(y_internal)
        n_classes = len(classes)
        weight_arr = np.ones(3)
        for c, cnt in zip(classes, counts, strict=True):
            weight_arr[c] = n_samples / (n_classes * cnt)
        class_weights = torch.tensor(weight_arr, dtype=torch.float32, device=self.device)
        logger.info(f"LSTM 클래스 가중치: {dict(zip(['Sell', 'Hold', 'Buy'], weight_arr.tolist(), strict=True))}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        logger.info(
            f"LSTM 학습 시작: {input_size} features, seq_length={self.config.get('seq_length', 60)}, "
            f"device={self.device}"
        )

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                optimizer.step()
                train_loss += loss.item()
                train_batches += 1

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    output = self.model(x_batch)
                    val_loss += criterion(output, y_batch).item()
                    val_batches += 1

            scheduler.step()

            avg_train = train_loss / max(train_batches, 1)
            avg_val = val_loss / max(val_batches, 1)
            history["train_loss"].append(avg_train)
            history["val_loss"].append(avg_val)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")

            # Early stopping
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}, best val_loss={best_val_loss:.4f}")
                    break

        # best 모델 복원
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        logger.info(f"학습 완료: {len(history['train_loss'])} epochs, best val_loss={best_val_loss:.4f}")
        return history

    def predict(self, df: pd.DataFrame, target_col: str = "target") -> np.ndarray:
        """매매 신호를 예측한다.

        시퀀스 길이만큼 앞부분이 잘리므로, 반환 배열 길이는 len(df) - seq_length이다.

        Args:
            df: 피처 DataFrame.
            target_col: 제외할 타겟 컬럼명.

        Returns:
            예측 레이블 배열 (-1, 0, 1).

        Raises:
            RuntimeError: 모델이 학습되지 않았을 때.
        """
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")

        loader = self._make_dataloader(df, target_col, shuffle=False)
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(self.device)
                output = self.model(x_batch)
                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)

        internal_preds = np.concatenate(all_preds)
        return _decode_labels(internal_preds)

    def predict_proba(self, df: pd.DataFrame, target_col: str = "target") -> np.ndarray:
        """클래스별 확률을 예측한다.

        Args:
            df: 피처 DataFrame.
            target_col: 제외할 타겟 컬럼명.

        Returns:
            (n_samples, 3) 확률 배열. 컬럼 순서: [Sell, Hold, Buy].

        Raises:
            RuntimeError: 모델이 학습되지 않았을 때.
        """
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")

        loader = self._make_dataloader(df, target_col, shuffle=False)
        self.model.eval()
        all_proba = []
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(self.device)
                output = self.model(x_batch)
                proba = torch.softmax(output, dim=1).cpu().numpy()
                all_proba.append(proba)

        return np.concatenate(all_proba)

    def evaluate(self, df: pd.DataFrame, target_col: str = "target") -> dict:
        """모델 성능을 평가한다.

        Args:
            df: 평가 DataFrame (피처 + 타겟).
            target_col: 타겟 컬럼명.

        Returns:
            accuracy, f1_macro, f1_weighted, classification_report,
            confusion_matrix를 포함하는 딕셔너리.
        """
        seq_length = self.config.get("seq_length", 60)
        y_true = df[target_col].values[seq_length - 1 : -1]
        y_pred = self.predict(df, target_col)

        # DataLoader 배치 처리로 끝부분이 잘릴 수 있으므로 길이 맞춤
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

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

        model_path = model_dir / "lstm_model.pt"
        torch.save(self.model.state_dict(), model_path)

        meta = {
            "feature_names": self.feature_names,
            "config": self.config,
            "input_size": len(self.feature_names),
            "seed": self._seed,
        }
        meta_path = model_dir / "lstm_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"모델 저장 완료: {model_path}")
        return model_path

    @classmethod
    def load(cls, model_dir: str | Path) -> LSTMSignalModel:
        """저장된 모델을 로드한다.

        Args:
            model_dir: 모델 디렉토리.

        Returns:
            로드된 LSTMSignalModel 인스턴스.
        """
        model_dir = Path(model_dir)

        meta_path = model_dir / "lstm_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        instance = cls(config=meta["config"])
        instance.feature_names = meta["feature_names"]
        instance._seed = meta.get("seed", 42)

        instance.model = instance._build_model(meta["input_size"])
        model_path = model_dir / "lstm_model.pt"
        state_dict = torch.load(model_path, map_location=instance.device, weights_only=True)
        instance.model.load_state_dict(state_dict)
        instance.model.eval()

        logger.info(f"모델 로드 완료: {model_path} (피처 {len(instance.feature_names)}개)")
        return instance


def train_from_parquet(
    data_dir: str | Path = "data/processed",
    config_path: str | Path = "configs/model_config.yaml",
    output_dir: str | Path = "data/models",
) -> dict:
    """Parquet 파일에서 데이터를 로드하여 LSTM 모델을 학습/평가/저장한다.

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

    model = LSTMSignalModel(config_path=config_path)

    logger.info("=== LSTM 학습 시작 ===")
    history = model.train(train_df, val_df)

    logger.info("=== Validation 세트 평가 ===")
    val_metrics = model.evaluate(val_df)

    logger.info("=== Test 세트 평가 ===")
    test_metrics = model.evaluate(test_df)

    # 과적합 확인
    train_metrics = model.evaluate(train_df)
    overfit_gap = train_metrics["accuracy"] - val_metrics["accuracy"]
    logger.info(
        f"과적합 점검: train_acc={train_metrics['accuracy']:.4f}, "
        f"val_acc={val_metrics['accuracy']:.4f}, gap={overfit_gap:.4f}"
    )

    model.save(output_dir)

    # 평가 결과 JSON 저장
    results = {
        "train_accuracy": train_metrics["accuracy"],
        "val": {k: v for k, v in val_metrics.items() if k != "classification_report"},
        "test": {k: v for k, v in test_metrics.items() if k != "classification_report"},
        "overfit_gap": overfit_gap,
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": min(history["val_loss"]),
    }
    results_path = Path(output_dir) / "lstm_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"평가 결과 저장: {results_path}")

    return test_metrics


if __name__ == "__main__":
    train_from_parquet()
