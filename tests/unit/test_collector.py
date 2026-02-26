"""collector 모듈 단위 테스트.

ccxt API 호출을 mock하여 네트워크 없이 테스트한다.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.collector import (
    _candles_to_dataframe,
    fetch_ohlcv,
    load_from_parquet,
    save_to_parquet,
    validate_data,
)

# 테스트용 캔들 데이터 (timestamp_ms, open, high, low, close, volume)
SAMPLE_CANDLES = [
    [1704067200000, 42000.0, 42500.0, 41800.0, 42300.0, 100.5],
    [1704070800000, 42300.0, 42700.0, 42100.0, 42600.0, 95.3],
    [1704074400000, 42600.0, 42900.0, 42400.0, 42800.0, 110.2],
    [1704078000000, 42800.0, 43000.0, 42500.0, 42700.0, 88.7],
    [1704081600000, 42700.0, 42800.0, 42200.0, 42400.0, 102.1],
]


class TestCandlesToDataframe:
    """_candles_to_dataframe 함수 테스트."""

    def test_basic_conversion(self) -> None:
        """캔들 리스트를 DataFrame으로 변환한다."""
        df = _candles_to_dataframe(SAMPLE_CANDLES)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "timestamp"

    def test_utc_timezone(self) -> None:
        """타임스탬프가 UTC로 설정되어야 한다."""
        df = _candles_to_dataframe(SAMPLE_CANDLES)
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_sorted_index(self) -> None:
        """인덱스가 오름차순 정렬되어야 한다."""
        reversed_candles = SAMPLE_CANDLES[::-1]
        df = _candles_to_dataframe(reversed_candles)
        assert df.index.is_monotonic_increasing

    def test_duplicate_removal(self) -> None:
        """중복 타임스탬프가 제거되어야 한다."""
        duped = SAMPLE_CANDLES + [SAMPLE_CANDLES[0]]
        df = _candles_to_dataframe(duped)
        assert len(df) == 5

    def test_empty_candles(self) -> None:
        """빈 리스트는 빈 DataFrame을 반환한다."""
        df = _candles_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestFetchOhlcv:
    """fetch_ohlcv 함수 테스트 (API mock)."""

    @patch("src.data.collector._create_exchange")
    def test_single_batch(self, mock_create: MagicMock) -> None:
        """캔들 수가 limit 미만이면 한 번만 요청한다."""
        mock_exchange = MagicMock()
        mock_exchange.parse8601.return_value = 1704067200000
        mock_exchange.fetch_ohlcv.return_value = SAMPLE_CANDLES
        mock_create.return_value = mock_exchange

        df = fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", since="2024-01-01", limit=1000)

        assert len(df) == 5
        mock_exchange.fetch_ohlcv.assert_called_once()

    @patch("src.data.collector._create_exchange")
    def test_multi_batch(self, mock_create: MagicMock) -> None:
        """캔들 수가 limit 이상이면 여러 번 요청한다."""
        batch1 = SAMPLE_CANDLES[:3]
        batch2 = SAMPLE_CANDLES[3:]

        mock_exchange = MagicMock()
        mock_exchange.parse8601.return_value = 1704067200000
        mock_exchange.fetch_ohlcv.side_effect = [batch1, batch2]
        mock_create.return_value = mock_exchange

        df = fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", since="2024-01-01", limit=3)

        assert len(df) == 5
        assert mock_exchange.fetch_ohlcv.call_count == 2

    @patch("src.data.collector._create_exchange")
    def test_empty_response(self, mock_create: MagicMock) -> None:
        """API가 빈 응답을 반환하면 빈 DataFrame을 반환한다."""
        mock_exchange = MagicMock()
        mock_exchange.parse8601.return_value = 1704067200000
        mock_exchange.fetch_ohlcv.return_value = []
        mock_create.return_value = mock_exchange

        df = fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", since="2024-01-01")

        assert len(df) == 0


class TestParquetIO:
    """Parquet 저장/로드 테스트."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """저장 후 로드하면 동일한 데이터를 얻는다."""
        df = _candles_to_dataframe(SAMPLE_CANDLES)
        filepath = tmp_path / "test.parquet"

        save_to_parquet(df, filepath)
        loaded = load_from_parquet(filepath)

        pd.testing.assert_frame_equal(df, loaded)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """부모 디렉토리가 없으면 자동 생성한다."""
        df = _candles_to_dataframe(SAMPLE_CANDLES)
        filepath = tmp_path / "nested" / "dir" / "test.parquet"

        save_to_parquet(df, filepath)
        assert filepath.exists()

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """존재하지 않는 파일 로드 시 FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_from_parquet(tmp_path / "nonexistent.parquet")


class TestValidateData:
    """validate_data 함수 테스트."""

    def test_clean_data(self) -> None:
        """정상 데이터는 이슈 없음."""
        df = _candles_to_dataframe(SAMPLE_CANDLES)
        issues = validate_data(df, timeframe="1h")
        assert issues == {}

    def test_detect_zero_price(self) -> None:
        """가격이 0인 캔들을 탐지한다."""
        candles = [*SAMPLE_CANDLES, [1704085200000, 0.0, 42500.0, 41800.0, 42300.0, 100.0]]
        df = _candles_to_dataframe(candles)
        issues = validate_data(df, timeframe="1h")
        assert "zero_or_negative_prices" in issues

    def test_detect_zero_volume(self) -> None:
        """volume이 0인 캔들을 탐지한다."""
        candles = [*SAMPLE_CANDLES, [1704085200000, 42000.0, 42500.0, 41800.0, 42300.0, 0.0]]
        df = _candles_to_dataframe(candles)
        issues = validate_data(df, timeframe="1h")
        assert "zero_volume" in issues

    def test_detect_missing_candles(self) -> None:
        """빠진 캔들(gap)을 탐지한다."""
        # 중간 캔들 하나 제거하여 gap 생성
        gapped = [SAMPLE_CANDLES[0], SAMPLE_CANDLES[2], SAMPLE_CANDLES[4]]
        df = _candles_to_dataframe(gapped)
        issues = validate_data(df, timeframe="1h")
        assert "missing_candles" in issues
