"""뉴스/소셜 텍스트 감성 분석 모듈.

HuggingFace 로컬 모델 또는 AWS Bedrock을 사용하여
텍스트의 시장 감성을 -1.0(극도 부정) ~ 1.0(극도 긍정)으로 점수화한다.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml
from loguru import logger


def _load_config(config_path: str | Path) -> dict:
    """YAML 설정 파일을 로드한다."""
    with open(config_path) as f:
        return yaml.safe_load(f)


class SentimentCache:
    """감성 점수 캐시.

    동일 텍스트에 대한 중복 추론을 방지한다.

    Args:
        ttl: 캐시 유효 시간 (초).
    """

    def __init__(self, ttl: int = 3600) -> None:
        self._cache: dict[str, tuple[float, float]] = {}  # hash → (score, timestamp)
        self._ttl = ttl

    @staticmethod
    def _hash(text: str) -> str:
        """텍스트의 SHA256 해시를 반환한다."""
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> float | None:
        """캐시된 감성 점수를 반환한다.

        Args:
            text: 입력 텍스트.

        Returns:
            캐시된 점수. 캐시 미스 또는 만료 시 None.
        """
        key = self._hash(text)
        if key not in self._cache:
            return None
        score, ts = self._cache[key]
        if time.time() - ts > self._ttl:
            del self._cache[key]
            return None
        return score

    def set(self, text: str, score: float) -> None:
        """감성 점수를 캐시에 저장한다.

        Args:
            text: 입력 텍스트.
            score: 감성 점수.
        """
        key = self._hash(text)
        self._cache[key] = (score, time.time())

    def clear(self) -> None:
        """캐시를 초기화한다."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class LocalSentimentAnalyzer:
    """HuggingFace 로컬 모델 기반 감성 분석기.

    cardiffnlp/twitter-roberta-base-sentiment-latest 모델을 사용하여
    API 비용 없이 감성 분석을 수행한다.

    Args:
        model_name: HuggingFace 모델 이름.
        cache_ttl: 캐시 유효 시간 (초).
    """

    # 모델 라벨 → 점수 매핑
    _LABEL_SCORES = {
        "negative": -1.0,
        "neutral": 0.0,
        "positive": 1.0,
    }

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        cache_ttl: int = 3600,
    ) -> None:
        self._model_name = model_name
        self._pipeline = None  # lazy loading
        self._cache = SentimentCache(ttl=cache_ttl)

    def _load_pipeline(self) -> None:
        """감성 분석 파이프라인을 로드한다 (최초 호출 시 1회)."""
        if self._pipeline is not None:
            return

        from transformers import pipeline

        self._pipeline = pipeline(
            "sentiment-analysis",
            model=self._model_name,
            top_k=None,
        )
        logger.info(f"감성 분석 모델 로드 완료: {self._model_name}")

    def analyze(self, text: str) -> float:
        """텍스트의 감성 점수를 반환한다.

        Args:
            text: 입력 텍스트 (뉴스 제목, 트윗 등).

        Returns:
            -1.0(극도 부정) ~ 1.0(극도 긍정) 감성 점수.
        """
        if not text or not text.strip():
            return 0.0

        cached = self._cache.get(text)
        if cached is not None:
            return cached

        self._load_pipeline()
        results = self._pipeline(text[:512])  # 토큰 제한

        # 가중 평균 점수 계산: sum(label_score * confidence)
        score = 0.0
        for item in results[0]:
            label = item["label"].lower()
            confidence = item["score"]
            label_score = self._LABEL_SCORES.get(label, 0.0)
            score += label_score * confidence

        # -1 ~ 1 범위 클리핑
        score = max(-1.0, min(1.0, score))

        self._cache.set(text, score)
        return score

    def analyze_batch(self, texts: list[str]) -> list[float]:
        """여러 텍스트의 감성 점수를 일괄 반환한다.

        Args:
            texts: 입력 텍스트 리스트.

        Returns:
            감성 점수 리스트.
        """
        return [self.analyze(text) for text in texts]


class BedrockSentimentAnalyzer:
    """AWS Bedrock Claude 기반 감성 분석기.

    프로덕션 환경에서 더 정교한 감성 분석을 위해 사용한다.

    Args:
        model_id: Bedrock 모델 ID.
        region: AWS 리전.
        cache_ttl: 캐시 유효 시간 (초).
    """

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
        region: str = "us-east-1",
        cache_ttl: int = 3600,
    ) -> None:
        self._model_id = model_id
        self._region = region
        self._client = None  # lazy loading
        self._cache = SentimentCache(ttl=cache_ttl)

    def _load_client(self) -> None:
        """Bedrock 클라이언트를 로드한다."""
        if self._client is not None:
            return

        import boto3

        self._client = boto3.client("bedrock-runtime", region_name=self._region)
        logger.info(f"Bedrock 클라이언트 로드 완료: {self._model_id} ({self._region})")

    def analyze(self, text: str) -> float:
        """텍스트의 감성 점수를 반환한다.

        Args:
            text: 입력 텍스트.

        Returns:
            -1.0 ~ 1.0 감성 점수.
        """
        if not text or not text.strip():
            return 0.0

        cached = self._cache.get(text)
        if cached is not None:
            return cached

        self._load_client()

        prompt = (
            "Analyze the following crypto market news/text and rate the sentiment "
            "from -1.0 (extremely bearish) to 1.0 (extremely bullish). "
            'Respond with ONLY a JSON: {"score": <float>, "reason": "<brief>"}\n\n'
            f"Text: {text[:1000]}"
        )

        response = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": prompt}],
                }
            ),
        )
        result = json.loads(response["body"].read())
        content = result["content"][0]["text"]

        try:
            score = float(json.loads(content)["score"])
            score = max(-1.0, min(1.0, score))
        except (json.JSONDecodeError, KeyError, ValueError):  # fmt: skip
            logger.warning(f"Bedrock 응답 파싱 실패, 0.0 반환: {content}")
            score = 0.0

        self._cache.set(text, score)
        return score

    def analyze_batch(self, texts: list[str]) -> list[float]:
        """여러 텍스트의 감성 점수를 일괄 반환한다.

        Args:
            texts: 입력 텍스트 리스트.

        Returns:
            감성 점수 리스트.
        """
        return [self.analyze(text) for text in texts]


def create_analyzer(
    provider: str = "local",
    config_path: str | Path | None = None,
) -> LocalSentimentAnalyzer | BedrockSentimentAnalyzer:
    """설정에 따라 적절한 감성 분석기를 생성한다.

    Args:
        provider: 분석기 종류 ("local" 또는 "bedrock").
        config_path: YAML 설정 파일 경로.

    Returns:
        감성 분석기 인스턴스.

    Raises:
        ValueError: 지원하지 않는 provider.
    """
    if config_path is not None:
        full_config = _load_config(config_path)
        sentiment_config = full_config.get("sentiment", {})
        provider = sentiment_config.get("provider", provider)
        cache_ttl = sentiment_config.get("cache_ttl", 3600)
    else:
        sentiment_config = {}
        cache_ttl = 3600

    if provider == "local":
        model_name = sentiment_config.get("local_model", "cardiffnlp/twitter-roberta-base-sentiment-latest")
        return LocalSentimentAnalyzer(model_name=model_name, cache_ttl=cache_ttl)
    elif provider == "bedrock":
        model_id = sentiment_config.get("bedrock_model", "anthropic.claude-3-haiku-20240307-v1:0")
        return BedrockSentimentAnalyzer(model_id=model_id, cache_ttl=cache_ttl)
    else:
        raise ValueError(f"지원하지 않는 provider: {provider} ('local' 또는 'bedrock')")


def compute_sentiment_features(
    texts: list[str],
    analyzer: LocalSentimentAnalyzer | BedrockSentimentAnalyzer | None = None,
) -> np.ndarray:
    """텍스트 리스트로부터 감성 피처를 생성한다.

    Args:
        texts: 뉴스/소셜 텍스트 리스트.
        analyzer: 감성 분석기 인스턴스 (None이면 로컬 분석기 생성).

    Returns:
        (n_samples,) 감성 점수 배열.
    """
    if analyzer is None:
        analyzer = LocalSentimentAnalyzer()

    scores = analyzer.analyze_batch(texts)
    return np.array(scores, dtype=np.float64)
