"""뉴스/소셜 텍스트 감성 분석 모듈.

CryptoBERT (크립토 소셜) + FinBERT (금융 뉴스) 듀얼 모델 또는
AWS Bedrock을 사용하여 텍스트의 시장 감성을 -1.0(극도 부정) ~ 1.0(극도 긍정)으로 점수화한다.
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


class CryptoBertSentimentAnalyzer:
    """CryptoBERT 기반 크립토 소셜 미디어 감성 분석기.

    ElKulako/cryptobert 모델을 사용하여 Reddit, Twitter 등
    크립토 소셜 텍스트의 감성을 분석한다.
    라벨: Bearish(0), Neutral(1), Bullish(2)

    Args:
        model_name: HuggingFace 모델 이름.
        cache_ttl: 캐시 유효 시간 (초).
    """

    # CryptoBERT 라벨 → 점수 매핑
    _LABEL_SCORES = {
        "bearish": -1.0,
        "neutral": 0.0,
        "bullish": 1.0,
    }

    def __init__(
        self,
        model_name: str = "ElKulako/cryptobert",
        cache_ttl: int = 3600,
    ) -> None:
        self._model_name = model_name
        self._tokenizer = None
        self._model = None
        self._cache = SentimentCache(ttl=cache_ttl)

    def _load_model(self) -> None:
        """모델과 토크나이저를 로드한다 (최초 호출 시 1회)."""
        if self._model is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        self._model.eval()
        logger.info(f"CryptoBERT 모델 로드 완료: {self._model_name}")

    def analyze(self, text: str) -> float:
        """크립토 소셜 텍스트의 감성 점수를 반환한다.

        Args:
            text: 입력 텍스트 (트윗, 레딧 포스트 등).

        Returns:
            -1.0(극도 부정) ~ 1.0(극도 긍정) 감성 점수.
        """
        if not text or not text.strip():
            return 0.0

        cached = self._cache.get(text)
        if cached is not None:
            return cached

        self._load_model()

        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze()

        probs_np = probs.cpu().numpy()
        # Bearish(-1) * prob[0] + Neutral(0) * prob[1] + Bullish(+1) * prob[2]
        score = float(-1.0 * probs_np[0] + 0.0 * probs_np[1] + 1.0 * probs_np[2])
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


class FinBertSentimentAnalyzer:
    """FinBERT 기반 금융 뉴스 감성 분석기.

    ProsusAI/finbert 모델을 사용하여 Bloomberg, Reuters 등
    금융 뉴스 텍스트의 감성을 분석한다.
    라벨: positive(0), negative(1), neutral(2)

    Args:
        model_name: HuggingFace 모델 이름.
        cache_ttl: 캐시 유효 시간 (초).
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        cache_ttl: int = 3600,
    ) -> None:
        self._model_name = model_name
        self._tokenizer = None
        self._model = None
        self._cache = SentimentCache(ttl=cache_ttl)

    def _load_model(self) -> None:
        """모델과 토크나이저를 로드한다 (최초 호출 시 1회)."""
        if self._model is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        self._model.eval()
        logger.info(f"FinBERT 모델 로드 완료: {self._model_name}")

    def analyze(self, text: str) -> float:
        """금융 뉴스 텍스트의 감성 점수를 반환한다.

        Args:
            text: 입력 텍스트 (뉴스 기사, 헤드라인 등).

        Returns:
            -1.0(극도 부정) ~ 1.0(극도 긍정) 감성 점수.
        """
        if not text or not text.strip():
            return 0.0

        cached = self._cache.get(text)
        if cached is not None:
            return cached

        self._load_model()

        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze()

        probs_np = probs.cpu().numpy()
        # positive(+1) * prob[0] + negative(-1) * prob[1] + neutral(0) * prob[2]
        score = float(1.0 * probs_np[0] + (-1.0) * probs_np[1] + 0.0 * probs_np[2])
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


class DualSentimentAnalyzer:
    """CryptoBERT + FinBERT 듀얼 감성 분석기.

    소스(social/news)에 따라 적절한 도메인 특화 모델로 분석하고,
    배치 분석 시 가중 평균으로 통합 점수를 반환한다.

    Args:
        crypto_model: CryptoBERT 모델 이름.
        fin_model: FinBERT 모델 이름.
        cache_ttl: 캐시 유효 시간 (초).
    """

    def __init__(
        self,
        crypto_model: str = "ElKulako/cryptobert",
        fin_model: str = "ProsusAI/finbert",
        cache_ttl: int = 3600,
    ) -> None:
        self._crypto = CryptoBertSentimentAnalyzer(model_name=crypto_model, cache_ttl=cache_ttl)
        self._fin = FinBertSentimentAnalyzer(model_name=fin_model, cache_ttl=cache_ttl)

    def analyze(self, text: str, source: str = "social") -> float:
        """텍스트 소스에 따라 적절한 모델로 감성을 분석한다.

        Args:
            text: 분석할 텍스트.
            source: "social" (Reddit, Twitter 등) 또는 "news" (금융 뉴스).

        Returns:
            -1.0(극도 부정) ~ 1.0(극도 긍정) 감성 점수.

        Raises:
            ValueError: 지원하지 않는 source.
        """
        if source == "social":
            return self._crypto.analyze(text)
        elif source == "news":
            return self._fin.analyze(text)
        else:
            raise ValueError(f"지원하지 않는 source: {source} ('social' 또는 'news')")

    def analyze_batch(
        self,
        social_texts: list[str] | None = None,
        news_texts: list[str] | None = None,
        weights: tuple[float, float] = (0.4, 0.6),
    ) -> float:
        """소셜 + 뉴스 텍스트를 배치로 분석하여 가중 평균 감성 점수를 반환한다.

        Args:
            social_texts: 크립토 소셜 텍스트 리스트.
            news_texts: 금융 뉴스 텍스트 리스트.
            weights: (소셜 가중치, 뉴스 가중치).

        Returns:
            가중 평균 감성 점수 (-1.0 ~ 1.0).
        """
        scores: list[tuple[float, float]] = []  # (avg_score, weight)
        w_social, w_news = weights

        if social_texts:
            social_scores = self._crypto.analyze_batch(social_texts)
            social_avg = float(np.mean(social_scores))
            scores.append((social_avg, w_social))

        if news_texts:
            news_scores = self._fin.analyze_batch(news_texts)
            news_avg = float(np.mean(news_scores))
            scores.append((news_avg, w_news))

        if not scores:
            return 0.0

        total_weight = sum(s[1] for s in scores)
        weighted_score = sum(s[0] * s[1] for s in scores) / total_weight
        return float(weighted_score)


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


SentimentAnalyzerType = (
    CryptoBertSentimentAnalyzer | FinBertSentimentAnalyzer | DualSentimentAnalyzer | BedrockSentimentAnalyzer
)


def create_analyzer(
    provider: str = "dual",
    config_path: str | Path | None = None,
) -> SentimentAnalyzerType:
    """설정에 따라 적절한 감성 분석기를 생성한다.

    Args:
        provider: 분석기 종류 ("dual", "crypto", "finbert", "bedrock").
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

    if provider == "dual":
        crypto_model = sentiment_config.get("crypto_model", "ElKulako/cryptobert")
        fin_model = sentiment_config.get("fin_model", "ProsusAI/finbert")
        return DualSentimentAnalyzer(crypto_model=crypto_model, fin_model=fin_model, cache_ttl=cache_ttl)
    elif provider == "crypto":
        crypto_model = sentiment_config.get("crypto_model", "ElKulako/cryptobert")
        return CryptoBertSentimentAnalyzer(model_name=crypto_model, cache_ttl=cache_ttl)
    elif provider == "finbert":
        fin_model = sentiment_config.get("fin_model", "ProsusAI/finbert")
        return FinBertSentimentAnalyzer(model_name=fin_model, cache_ttl=cache_ttl)
    elif provider == "bedrock":
        model_id = sentiment_config.get("bedrock_model", "anthropic.claude-3-haiku-20240307-v1:0")
        return BedrockSentimentAnalyzer(model_id=model_id, cache_ttl=cache_ttl)
    else:
        raise ValueError(f"지원하지 않는 provider: {provider} ('dual', 'crypto', 'finbert', 'bedrock')")


def compute_sentiment_features(
    texts: list[str],
    analyzer: SentimentAnalyzerType | None = None,
    source: str = "social",
) -> np.ndarray:
    """텍스트 리스트로부터 감성 피처를 생성한다.

    Args:
        texts: 뉴스/소셜 텍스트 리스트.
        analyzer: 감성 분석기 인스턴스 (None이면 CryptoBERT 분석기 생성).
        source: 텍스트 소스 ("social" 또는 "news"). DualSentimentAnalyzer 사용 시 적용.

    Returns:
        (n_samples,) 감성 점수 배열.
    """
    if analyzer is None:
        analyzer = CryptoBertSentimentAnalyzer()

    if isinstance(analyzer, DualSentimentAnalyzer):
        scores = [analyzer.analyze(text, source=source) for text in texts]
    else:
        scores = analyzer.analyze_batch(texts)
    return np.array(scores, dtype=np.float64)
