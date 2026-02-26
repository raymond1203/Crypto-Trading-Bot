"""감성 분석 모듈 단위 테스트.

HuggingFace 모델 호출을 mock하여 빠르게 검증한다.
"""

import json
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from src.data.sentiment import (
    BedrockSentimentAnalyzer,
    LocalSentimentAnalyzer,
    SentimentCache,
    compute_sentiment_features,
    create_analyzer,
)


class TestSentimentCache:
    """SentimentCache 테스트."""

    def test_set_and_get(self) -> None:
        """저장 후 조회가 가능해야 한다."""
        cache = SentimentCache(ttl=60)
        cache.set("hello", 0.5)
        assert cache.get("hello") == 0.5

    def test_cache_miss(self) -> None:
        """존재하지 않는 키는 None을 반환해야 한다."""
        cache = SentimentCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self) -> None:
        """TTL이 지나면 None을 반환해야 한다."""
        cache = SentimentCache(ttl=0)  # 즉시 만료
        cache.set("test", 0.3)
        time.sleep(0.01)
        assert cache.get("test") is None

    def test_clear(self) -> None:
        """clear 후 캐시가 비어야 한다."""
        cache = SentimentCache()
        cache.set("a", 0.1)
        cache.set("b", 0.2)
        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0

    def test_same_text_same_hash(self) -> None:
        """동일 텍스트는 동일 해시를 생성해야 한다."""
        cache = SentimentCache()
        cache.set("bitcoin rally", 0.8)
        assert cache.get("bitcoin rally") == 0.8

    def test_different_text_different_hash(self) -> None:
        """다른 텍스트는 서로 다른 캐시 엔트리여야 한다."""
        cache = SentimentCache()
        cache.set("bullish", 0.9)
        cache.set("bearish", -0.9)
        assert cache.get("bullish") == 0.9
        assert cache.get("bearish") == -0.9


class TestLocalSentimentAnalyzer:
    """LocalSentimentAnalyzer 테스트 (HuggingFace mock)."""

    def _mock_pipeline_result(self, label: str, score: float) -> list[list[dict]]:
        """mock pipeline 결과를 생성한다."""
        return [
            [
                {"label": label, "score": score},
                {"label": "neutral", "score": 1.0 - score},
            ]
        ]

    @patch("src.data.sentiment.LocalSentimentAnalyzer._load_pipeline")
    def test_analyze_positive(self, mock_load: MagicMock) -> None:
        """긍정 텍스트는 양수 점수를 반환해야 한다."""
        analyzer = LocalSentimentAnalyzer()
        analyzer._pipeline = MagicMock()
        analyzer._pipeline.return_value = [
            [
                {"label": "positive", "score": 0.85},
                {"label": "neutral", "score": 0.10},
                {"label": "negative", "score": 0.05},
            ]
        ]

        score = analyzer.analyze("Bitcoin hits new all-time high!")
        assert score > 0
        assert -1.0 <= score <= 1.0

    @patch("src.data.sentiment.LocalSentimentAnalyzer._load_pipeline")
    def test_analyze_negative(self, mock_load: MagicMock) -> None:
        """부정 텍스트는 음수 점수를 반환해야 한다."""
        analyzer = LocalSentimentAnalyzer()
        analyzer._pipeline = MagicMock()
        analyzer._pipeline.return_value = [
            [
                {"label": "negative", "score": 0.90},
                {"label": "neutral", "score": 0.07},
                {"label": "positive", "score": 0.03},
            ]
        ]

        score = analyzer.analyze("Crypto market crashes 50%")
        assert score < 0
        assert -1.0 <= score <= 1.0

    @patch("src.data.sentiment.LocalSentimentAnalyzer._load_pipeline")
    def test_analyze_neutral(self, mock_load: MagicMock) -> None:
        """중립 텍스트는 0 근처 점수를 반환해야 한다."""
        analyzer = LocalSentimentAnalyzer()
        analyzer._pipeline = MagicMock()
        analyzer._pipeline.return_value = [
            [
                {"label": "neutral", "score": 0.95},
                {"label": "positive", "score": 0.03},
                {"label": "negative", "score": 0.02},
            ]
        ]

        score = analyzer.analyze("Bitcoin price unchanged today")
        assert abs(score) < 0.1

    def test_empty_text_returns_zero(self) -> None:
        """빈 텍스트는 0.0을 반환해야 한다."""
        analyzer = LocalSentimentAnalyzer()
        assert analyzer.analyze("") == 0.0
        assert analyzer.analyze("   ") == 0.0

    @patch("src.data.sentiment.LocalSentimentAnalyzer._load_pipeline")
    def test_cache_prevents_duplicate_calls(self, mock_load: MagicMock) -> None:
        """캐시가 동작하여 같은 텍스트에 대해 파이프라인을 재호출하지 않아야 한다."""
        analyzer = LocalSentimentAnalyzer()
        analyzer._pipeline = MagicMock()
        analyzer._pipeline.return_value = [
            [
                {"label": "positive", "score": 0.8},
                {"label": "neutral", "score": 0.15},
                {"label": "negative", "score": 0.05},
            ]
        ]

        text = "BTC to the moon"
        score1 = analyzer.analyze(text)
        score2 = analyzer.analyze(text)
        assert score1 == score2
        # 파이프라인은 1번만 호출되어야 함
        assert analyzer._pipeline.call_count == 1

    @patch("src.data.sentiment.LocalSentimentAnalyzer._load_pipeline")
    def test_analyze_batch(self, mock_load: MagicMock) -> None:
        """배치 분석이 올바른 길이의 결과를 반환해야 한다."""
        analyzer = LocalSentimentAnalyzer()
        analyzer._pipeline = MagicMock()
        analyzer._pipeline.return_value = [
            [
                {"label": "positive", "score": 0.7},
                {"label": "neutral", "score": 0.2},
                {"label": "negative", "score": 0.1},
            ]
        ]

        texts = ["good news", "bad news", "neutral news"]
        scores = analyzer.analyze_batch(texts)
        assert len(scores) == 3
        assert all(-1.0 <= s <= 1.0 for s in scores)

    @patch("src.data.sentiment.LocalSentimentAnalyzer._load_pipeline")
    def test_score_range_clipping(self, mock_load: MagicMock) -> None:
        """점수는 항상 -1~1 범위로 클리핑되어야 한다."""
        analyzer = LocalSentimentAnalyzer()
        analyzer._pipeline = MagicMock()
        # 극단적 결과
        analyzer._pipeline.return_value = [
            [
                {"label": "positive", "score": 1.0},
                {"label": "neutral", "score": 0.0},
                {"label": "negative", "score": 0.0},
            ]
        ]

        score = analyzer.analyze("extreme positive")
        assert -1.0 <= score <= 1.0


class TestBedrockSentimentAnalyzer:
    """BedrockSentimentAnalyzer 테스트 (boto3 mock)."""

    def test_empty_text_returns_zero(self) -> None:
        """빈 텍스트는 0.0을 반환해야 한다."""
        analyzer = BedrockSentimentAnalyzer()
        assert analyzer.analyze("") == 0.0

    @patch("src.data.sentiment.BedrockSentimentAnalyzer._load_client")
    def test_analyze_with_mock(self, mock_load: MagicMock) -> None:
        """Bedrock 응답을 정상 파싱해야 한다."""
        analyzer = BedrockSentimentAnalyzer()

        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(
            {
                "content": [{"text": '{"score": 0.75, "reason": "bullish news"}'}],
            }
        ).encode()

        analyzer._client = MagicMock()
        analyzer._client.invoke_model.return_value = {"body": mock_body}

        score = analyzer.analyze("Bitcoin ETF approved")
        assert score == 0.75

    @patch("src.data.sentiment.BedrockSentimentAnalyzer._load_client")
    def test_invalid_response_returns_zero(self, mock_load: MagicMock) -> None:
        """파싱 실패 시 0.0을 반환해야 한다."""
        analyzer = BedrockSentimentAnalyzer()

        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(
            {
                "content": [{"text": "invalid json response"}],
            }
        ).encode()

        analyzer._client = MagicMock()
        analyzer._client.invoke_model.return_value = {"body": mock_body}

        score = analyzer.analyze("Some news")
        assert score == 0.0


class TestCreateAnalyzer:
    """create_analyzer 팩토리 함수 테스트."""

    def test_local_default(self) -> None:
        """기본값은 LocalSentimentAnalyzer여야 한다."""
        analyzer = create_analyzer(provider="local")
        assert isinstance(analyzer, LocalSentimentAnalyzer)

    def test_bedrock(self) -> None:
        """bedrock provider는 BedrockSentimentAnalyzer여야 한다."""
        analyzer = create_analyzer(provider="bedrock")
        assert isinstance(analyzer, BedrockSentimentAnalyzer)

    def test_invalid_provider(self) -> None:
        """잘못된 provider는 ValueError."""
        with pytest.raises(ValueError, match="지원하지 않는 provider"):
            create_analyzer(provider="openai")

    def test_from_config_file(self, tmp_path: object) -> None:
        """YAML 설정 파일에서 로드되어야 한다."""
        config = {
            "sentiment": {
                "provider": "local",
                "local_model": "custom/model-name",
                "cache_ttl": 7200,
            },
        }
        config_path = tmp_path / "config.yaml"  # type: ignore[operator]
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        analyzer = create_analyzer(config_path=config_path)
        assert isinstance(analyzer, LocalSentimentAnalyzer)
        assert analyzer._model_name == "custom/model-name"
        assert analyzer._cache._ttl == 7200


class TestComputeSentimentFeatures:
    """compute_sentiment_features 테스트."""

    @patch("src.data.sentiment.LocalSentimentAnalyzer._load_pipeline")
    def test_returns_numpy_array(self, mock_load: MagicMock) -> None:
        """numpy 배열을 반환해야 한다."""
        analyzer = LocalSentimentAnalyzer()
        analyzer._pipeline = MagicMock()
        analyzer._pipeline.return_value = [
            [
                {"label": "positive", "score": 0.6},
                {"label": "neutral", "score": 0.3},
                {"label": "negative", "score": 0.1},
            ]
        ]

        result = compute_sentiment_features(["text1", "text2"], analyzer=analyzer)
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    @patch("src.data.sentiment.LocalSentimentAnalyzer._load_pipeline")
    def test_score_dtype(self, mock_load: MagicMock) -> None:
        """반환 배열의 dtype이 float64여야 한다."""
        analyzer = LocalSentimentAnalyzer()
        analyzer._pipeline = MagicMock()
        analyzer._pipeline.return_value = [
            [
                {"label": "neutral", "score": 1.0},
                {"label": "positive", "score": 0.0},
                {"label": "negative", "score": 0.0},
            ]
        ]

        result = compute_sentiment_features(["test"], analyzer=analyzer)
        assert result.dtype == np.float64
