"""감성 분석 모듈 단위 테스트.

CryptoBERT, FinBERT, DualSentimentAnalyzer 및 Bedrock 분석기를 mock하여 검증한다.
"""

import json
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml

from src.data.sentiment import (
    BedrockSentimentAnalyzer,
    CryptoBertSentimentAnalyzer,
    DualSentimentAnalyzer,
    FinBertSentimentAnalyzer,
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


def _mock_model_output(probs: list[float]) -> MagicMock:
    """모델 출력을 mock한다. probs: [p0, p1, p2] softmax 확률."""
    logits = torch.tensor([probs])  # 이미 softmax 된 값을 logits로 사용
    output = MagicMock()
    output.logits = logits
    return output


class TestCryptoBertSentimentAnalyzer:
    """CryptoBertSentimentAnalyzer 테스트."""

    @patch("src.data.sentiment.CryptoBertSentimentAnalyzer._load_model")
    def test_analyze_bullish(self, mock_load: MagicMock) -> None:
        """Bullish 텍스트는 양수 점수를 반환해야 한다."""
        analyzer = CryptoBertSentimentAnalyzer()
        analyzer._tokenizer = MagicMock()
        analyzer._tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        analyzer._model = MagicMock()
        # Bearish=0.05, Neutral=0.10, Bullish=0.85
        analyzer._model.return_value = _mock_model_output([0.05, 0.10, 0.85])

        score = analyzer.analyze("BTC to the moon! 🚀")
        assert score > 0
        assert -1.0 <= score <= 1.0

    @patch("src.data.sentiment.CryptoBertSentimentAnalyzer._load_model")
    def test_analyze_bearish(self, mock_load: MagicMock) -> None:
        """Bearish 텍스트는 음수 점수를 반환해야 한다."""
        analyzer = CryptoBertSentimentAnalyzer()
        analyzer._tokenizer = MagicMock()
        analyzer._tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        analyzer._model = MagicMock()
        # Bearish=0.90, Neutral=0.07, Bullish=0.03
        analyzer._model.return_value = _mock_model_output([0.90, 0.07, 0.03])

        score = analyzer.analyze("Crypto crash incoming, sell everything")
        assert score < 0
        assert -1.0 <= score <= 1.0

    @patch("src.data.sentiment.CryptoBertSentimentAnalyzer._load_model")
    def test_analyze_neutral(self, mock_load: MagicMock) -> None:
        """중립 텍스트는 0 근처 점수를 반환해야 한다."""
        analyzer = CryptoBertSentimentAnalyzer()
        analyzer._tokenizer = MagicMock()
        analyzer._tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        analyzer._model = MagicMock()
        # Bearish=0.10, Neutral=0.80, Bullish=0.10
        analyzer._model.return_value = _mock_model_output([0.10, 0.80, 0.10])

        score = analyzer.analyze("Bitcoin price unchanged today")
        assert abs(score) < 0.1

    def test_empty_text_returns_zero(self) -> None:
        """빈 텍스트는 0.0을 반환해야 한다."""
        analyzer = CryptoBertSentimentAnalyzer()
        assert analyzer.analyze("") == 0.0
        assert analyzer.analyze("   ") == 0.0

    @patch("src.data.sentiment.CryptoBertSentimentAnalyzer._load_model")
    def test_cache_prevents_duplicate_calls(self, mock_load: MagicMock) -> None:
        """캐시가 동작하여 같은 텍스트에 대해 모델을 재호출하지 않아야 한다."""
        analyzer = CryptoBertSentimentAnalyzer()
        analyzer._tokenizer = MagicMock()
        analyzer._tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        analyzer._model = MagicMock()
        analyzer._model.return_value = _mock_model_output([0.05, 0.10, 0.85])

        text = "BTC to the moon"
        score1 = analyzer.analyze(text)
        score2 = analyzer.analyze(text)
        assert score1 == score2
        # 모델은 1번만 호출되어야 함
        assert analyzer._model.call_count == 1


class TestFinBertSentimentAnalyzer:
    """FinBertSentimentAnalyzer 테스트."""

    @patch("src.data.sentiment.FinBertSentimentAnalyzer._load_model")
    def test_analyze_positive(self, mock_load: MagicMock) -> None:
        """긍정 뉴스는 양수 점수를 반환해야 한다."""
        analyzer = FinBertSentimentAnalyzer()
        analyzer._tokenizer = MagicMock()
        analyzer._tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        analyzer._model = MagicMock()
        # positive=0.85, negative=0.05, neutral=0.10
        analyzer._model.return_value = _mock_model_output([0.85, 0.05, 0.10])

        score = analyzer.analyze("Bitcoin ETF approved by SEC")
        assert score > 0
        assert -1.0 <= score <= 1.0

    @patch("src.data.sentiment.FinBertSentimentAnalyzer._load_model")
    def test_analyze_negative(self, mock_load: MagicMock) -> None:
        """부정 뉴스는 음수 점수를 반환해야 한다."""
        analyzer = FinBertSentimentAnalyzer()
        analyzer._tokenizer = MagicMock()
        analyzer._tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        analyzer._model = MagicMock()
        # positive=0.03, negative=0.90, neutral=0.07
        analyzer._model.return_value = _mock_model_output([0.03, 0.90, 0.07])

        score = analyzer.analyze("Major exchange files for bankruptcy")
        assert score < 0
        assert -1.0 <= score <= 1.0

    @patch("src.data.sentiment.FinBertSentimentAnalyzer._load_model")
    def test_analyze_neutral(self, mock_load: MagicMock) -> None:
        """중립 뉴스는 0 근처 점수를 반환해야 한다."""
        analyzer = FinBertSentimentAnalyzer()
        analyzer._tokenizer = MagicMock()
        analyzer._tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        analyzer._model = MagicMock()
        # positive=0.10, negative=0.10, neutral=0.80
        analyzer._model.return_value = _mock_model_output([0.10, 0.10, 0.80])

        score = analyzer.analyze("Fed to release minutes on Wednesday")
        assert abs(score) < 0.1

    def test_empty_text_returns_zero(self) -> None:
        """빈 텍스트는 0.0을 반환해야 한다."""
        analyzer = FinBertSentimentAnalyzer()
        assert analyzer.analyze("") == 0.0
        assert analyzer.analyze("   ") == 0.0

    @patch("src.data.sentiment.FinBertSentimentAnalyzer._load_model")
    def test_cache_prevents_duplicate_calls(self, mock_load: MagicMock) -> None:
        """캐시가 동작하여 같은 텍스트에 대해 모델을 재호출하지 않아야 한다."""
        analyzer = FinBertSentimentAnalyzer()
        analyzer._tokenizer = MagicMock()
        analyzer._tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        analyzer._model = MagicMock()
        analyzer._model.return_value = _mock_model_output([0.85, 0.05, 0.10])

        text = "Bitcoin ETF approved"
        score1 = analyzer.analyze(text)
        score2 = analyzer.analyze(text)
        assert score1 == score2
        assert analyzer._model.call_count == 1


class TestDualSentimentAnalyzer:
    """DualSentimentAnalyzer 테스트."""

    def _make_analyzer(self) -> DualSentimentAnalyzer:
        """mock된 DualSentimentAnalyzer를 생성한다."""
        analyzer = DualSentimentAnalyzer()

        # CryptoBERT mock
        analyzer._crypto._tokenizer = MagicMock()
        analyzer._crypto._tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        analyzer._crypto._model = MagicMock()
        # Bearish=0.05, Neutral=0.10, Bullish=0.85 → score ≈ 0.8
        analyzer._crypto._model.return_value = _mock_model_output([0.05, 0.10, 0.85])

        # FinBERT mock
        analyzer._fin._tokenizer = MagicMock()
        analyzer._fin._tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        analyzer._fin._model = MagicMock()
        # positive=0.80, negative=0.10, neutral=0.10 → score ≈ 0.7
        analyzer._fin._model.return_value = _mock_model_output([0.80, 0.10, 0.10])

        return analyzer

    @patch("src.data.sentiment.CryptoBertSentimentAnalyzer._load_model")
    @patch("src.data.sentiment.FinBertSentimentAnalyzer._load_model")
    def test_analyze_social(self, mock_fin: MagicMock, mock_crypto: MagicMock) -> None:
        """social 소스는 CryptoBERT를 사용해야 한다."""
        analyzer = self._make_analyzer()
        score = analyzer.analyze("BTC moon! 🚀", source="social")
        assert score > 0
        assert analyzer._crypto._model.call_count == 1
        assert analyzer._fin._model.call_count == 0

    @patch("src.data.sentiment.CryptoBertSentimentAnalyzer._load_model")
    @patch("src.data.sentiment.FinBertSentimentAnalyzer._load_model")
    def test_analyze_news(self, mock_fin: MagicMock, mock_crypto: MagicMock) -> None:
        """news 소스는 FinBERT를 사용해야 한다."""
        analyzer = self._make_analyzer()
        score = analyzer.analyze("SEC approves Bitcoin ETF", source="news")
        assert score > 0
        assert analyzer._crypto._model.call_count == 0
        assert analyzer._fin._model.call_count == 1

    def test_analyze_invalid_source(self) -> None:
        """잘못된 source는 ValueError."""
        analyzer = DualSentimentAnalyzer()
        with pytest.raises(ValueError, match="지원하지 않는 source"):
            analyzer.analyze("some text", source="invalid")

    @patch("src.data.sentiment.CryptoBertSentimentAnalyzer._load_model")
    @patch("src.data.sentiment.FinBertSentimentAnalyzer._load_model")
    def test_analyze_batch_weighted(self, mock_fin: MagicMock, mock_crypto: MagicMock) -> None:
        """배치 분석이 가중 평균 점수를 반환해야 한다."""
        analyzer = self._make_analyzer()
        score = analyzer.analyze_batch(
            social_texts=["bullish tweet"],
            news_texts=["positive news"],
            weights=(0.4, 0.6),
        )
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0
        # 두 모델 모두 호출되어야 함
        assert analyzer._crypto._model.call_count == 1
        assert analyzer._fin._model.call_count == 1

    @patch("src.data.sentiment.CryptoBertSentimentAnalyzer._load_model")
    @patch("src.data.sentiment.FinBertSentimentAnalyzer._load_model")
    def test_analyze_batch_social_only(self, mock_fin: MagicMock, mock_crypto: MagicMock) -> None:
        """소셜만 있어도 점수를 반환해야 한다."""
        analyzer = self._make_analyzer()
        score = analyzer.analyze_batch(social_texts=["tweet1", "tweet2"])
        assert isinstance(score, float)
        assert analyzer._crypto._model.call_count == 2
        assert analyzer._fin._model.call_count == 0

    def test_analyze_batch_empty(self) -> None:
        """빈 입력은 0.0을 반환해야 한다."""
        analyzer = DualSentimentAnalyzer()
        assert analyzer.analyze_batch() == 0.0


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

    def test_dual_default(self) -> None:
        """기본값은 DualSentimentAnalyzer여야 한다."""
        analyzer = create_analyzer()
        assert isinstance(analyzer, DualSentimentAnalyzer)

    def test_crypto_provider(self) -> None:
        """crypto provider는 CryptoBertSentimentAnalyzer여야 한다."""
        analyzer = create_analyzer(provider="crypto")
        assert isinstance(analyzer, CryptoBertSentimentAnalyzer)

    def test_finbert_provider(self) -> None:
        """finbert provider는 FinBertSentimentAnalyzer여야 한다."""
        analyzer = create_analyzer(provider="finbert")
        assert isinstance(analyzer, FinBertSentimentAnalyzer)

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
                "provider": "dual",
                "crypto_model": "ElKulako/cryptobert",
                "fin_model": "ProsusAI/finbert",
                "cache_ttl": 7200,
            },
        }
        config_path = tmp_path / "config.yaml"  # type: ignore[operator]
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        analyzer = create_analyzer(config_path=config_path)
        assert isinstance(analyzer, DualSentimentAnalyzer)
        assert analyzer._crypto._model_name == "ElKulako/cryptobert"
        assert analyzer._fin._model_name == "ProsusAI/finbert"
        assert analyzer._crypto._cache._ttl == 7200

    def test_from_config_file_crypto_only(self, tmp_path: object) -> None:
        """crypto provider를 config에서 로드할 수 있어야 한다."""
        config = {
            "sentiment": {
                "provider": "crypto",
                "crypto_model": "custom/cryptobert",
                "cache_ttl": 1800,
            },
        }
        config_path = tmp_path / "config.yaml"  # type: ignore[operator]
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        analyzer = create_analyzer(config_path=config_path)
        assert isinstance(analyzer, CryptoBertSentimentAnalyzer)
        assert analyzer._model_name == "custom/cryptobert"


class TestComputeSentimentFeatures:
    """compute_sentiment_features 테스트."""

    @patch("src.data.sentiment.CryptoBertSentimentAnalyzer._load_model")
    def test_returns_numpy_array(self, mock_load: MagicMock) -> None:
        """numpy 배열을 반환해야 한다."""
        analyzer = CryptoBertSentimentAnalyzer()
        analyzer._tokenizer = MagicMock()
        analyzer._tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        analyzer._model = MagicMock()
        analyzer._model.return_value = _mock_model_output([0.10, 0.30, 0.60])

        result = compute_sentiment_features(["text1", "text2"], analyzer=analyzer)
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    @patch("src.data.sentiment.CryptoBertSentimentAnalyzer._load_model")
    def test_score_dtype(self, mock_load: MagicMock) -> None:
        """반환 배열의 dtype이 float64여야 한다."""
        analyzer = CryptoBertSentimentAnalyzer()
        analyzer._tokenizer = MagicMock()
        analyzer._tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        analyzer._model = MagicMock()
        analyzer._model.return_value = _mock_model_output([0.10, 0.80, 0.10])

        result = compute_sentiment_features(["test"], analyzer=analyzer)
        assert result.dtype == np.float64

    @patch("src.data.sentiment.CryptoBertSentimentAnalyzer._load_model")
    @patch("src.data.sentiment.FinBertSentimentAnalyzer._load_model")
    def test_with_dual_analyzer(self, mock_fin: MagicMock, mock_crypto: MagicMock) -> None:
        """DualSentimentAnalyzer로도 피처 생성이 가능해야 한다."""
        analyzer = DualSentimentAnalyzer()
        analyzer._crypto._tokenizer = MagicMock()
        analyzer._crypto._tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        analyzer._crypto._model = MagicMock()
        analyzer._crypto._model.return_value = _mock_model_output([0.10, 0.10, 0.80])

        result = compute_sentiment_features(["tweet1", "tweet2"], analyzer=analyzer, source="social")
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
