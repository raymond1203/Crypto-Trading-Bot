"""BTC/USDT OHLCV 데이터 다운로드 스크립트.

Usage:
    python -m scripts.download_data --symbol BTC/USDT --timeframes 1h 4h 1d --since 2024-01-01 --output data/raw/
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가하여 직접 실행도 지원
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collector import fetch_ohlcv, save_to_parquet, validate_data


def main() -> None:
    """커맨드라인 인자를 파싱하고 데이터를 다운로드한다."""
    parser = argparse.ArgumentParser(description="Binance OHLCV 데이터 다운로드")
    parser.add_argument("--symbol", default="BTC/USDT", help="거래쌍 (default: BTC/USDT)")
    parser.add_argument("--timeframes", nargs="+", default=["1h", "4h", "1d"], help="타임프레임 목록")
    parser.add_argument("--since", default="2024-01-01", help="시작 날짜 (default: 2024-01-01)")
    parser.add_argument("--output", default="data/raw/", help="저장 디렉토리 (default: data/raw/)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    today = datetime.now().strftime("%Y%m%d")
    since_str = args.since.replace("-", "")
    symbol_slug = args.symbol.replace("/", "_").lower()

    for tf in args.timeframes:
        df = fetch_ohlcv(symbol=args.symbol, timeframe=tf, since=args.since)
        issues = validate_data(df, timeframe=tf)

        filename = f"{symbol_slug}_{tf}_{since_str}_{today}.parquet"
        save_to_parquet(df, output_dir / filename)

        if issues:
            print(f"  ⚠ {tf}: {issues}")


if __name__ == "__main__":
    main()
