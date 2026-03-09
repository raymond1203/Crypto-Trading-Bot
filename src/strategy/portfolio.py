"""포지션 및 거래 관리 모듈.

로컬 dict 모드와 DynamoDB 모드를 지원하여,
백테스트/테스트 시에는 인메모리로, Lambda 배포 시에는 DynamoDB로 동작한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

import boto3
from boto3.dynamodb.conditions import Key
from loguru import logger


@dataclass
class Position:
    """오픈 포지션 정보.

    Attributes:
        symbol: 거래 심볼 (e.g., "BTC/USDT").
        side: 포지션 방향 ("long" | "short").
        entry_price: 진입 가격.
        size: 포지션 크기 (자본 비율, 0~1).
        entry_time: 진입 시각 (ISO 8601).
    """

    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    size: float
    entry_time: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class Trade:
    """완료된 거래 정보.

    Attributes:
        symbol: 거래 심볼.
        side: 포지션 방향.
        entry_price: 진입 가격.
        exit_price: 청산 가격.
        size: 포지션 크기.
        entry_time: 진입 시각.
        exit_time: 청산 시각.
        pnl_pct: 손익률 (%).
    """

    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    exit_price: float
    size: float
    entry_time: str
    exit_time: str
    pnl_pct: float


class PortfolioManager:
    """포지션 관리자.

    로컬 모드(dict)와 DynamoDB 모드를 지원한다.

    Attributes:
        mode: 저장 모드 ("local" | "dynamodb").
    """

    def __init__(
        self,
        mode: Literal["local", "dynamodb"] = "local",
        table_prefix: str = "cryptosentinel",
        region: str = "ap-northeast-2",
    ) -> None:
        """포지션 관리자를 초기화한다.

        Args:
            mode: 저장 모드.
            table_prefix: DynamoDB 테이블 이름 접두사.
            region: AWS 리전 (DynamoDB 모드 시).
        """
        self.mode = mode
        self._table_prefix = table_prefix

        if mode == "local":
            self._positions: dict[str, Position] = {}
            self._trades: dict[str, list[Trade]] = {}
            self._bot_state: dict[str, str] = {}
        else:
            dynamodb = boto3.resource("dynamodb", region_name=region)
            self._positions_table = dynamodb.Table(f"{table_prefix}-positions")
            self._trades_table = dynamodb.Table(f"{table_prefix}-trades")
            self._bot_state_table = dynamodb.Table(f"{table_prefix}-bot-state")

        logger.info(f"PortfolioManager 초기화: mode={mode}, prefix={table_prefix}")

    def open_position(
        self,
        symbol: str,
        side: Literal["long", "short"],
        entry_price: float,
        size: float,
    ) -> Position:
        """새 포지션을 연다.

        Args:
            symbol: 거래 심볼.
            side: 포지션 방향.
            entry_price: 진입 가격.
            size: 포지션 크기 (자본 비율).

        Returns:
            생성된 Position.

        Raises:
            ValueError: 해당 심볼에 이미 오픈 포지션이 있을 때.
        """
        existing = self.get_current_position(symbol)
        if existing is not None:
            raise ValueError(f"{symbol} 포지션이 이미 존재합니다: {existing}")

        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=size,
        )

        if self.mode == "local":
            self._positions[symbol] = position
        else:
            self._positions_table.put_item(Item=_to_dynamo_item(asdict(position)))

        logger.info(f"포지션 오픈: {symbol} {side} @ {entry_price}, size={size}")
        return position

    def close_position(self, symbol: str, exit_price: float) -> Trade:
        """포지션을 청산한다.

        Args:
            symbol: 거래 심볼.
            exit_price: 청산 가격.

        Returns:
            완료된 Trade.

        Raises:
            ValueError: 해당 심볼에 오픈 포지션이 없을 때.
        """
        position = self.get_current_position(symbol)
        if position is None:
            raise ValueError(f"{symbol} 오픈 포지션이 없습니다.")

        if position.side == "long":
            pnl_pct = (exit_price - position.entry_price) / position.entry_price * 100
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price * 100

        exit_time = datetime.now(UTC).isoformat()

        trade = Trade(
            symbol=symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            entry_time=position.entry_time,
            exit_time=exit_time,
            pnl_pct=round(pnl_pct, 4),
        )

        if self.mode == "local":
            del self._positions[symbol]
            self._trades.setdefault(symbol, []).append(trade)
        else:
            self._positions_table.delete_item(Key={"symbol": symbol})
            self._trades_table.put_item(Item=_to_dynamo_item(asdict(trade)))

        logger.info(f"포지션 청산: {symbol} {position.side} @ {exit_price}, PnL={pnl_pct:+.2f}%")
        return trade

    def get_current_position(self, symbol: str) -> Position | None:
        """현재 오픈 포지션을 조회한다.

        Args:
            symbol: 거래 심볼.

        Returns:
            Position 또는 없으면 None.
        """
        if self.mode == "local":
            return self._positions.get(symbol)

        response = self._positions_table.get_item(Key={"symbol": symbol})
        item = response.get("Item")
        if item is None:
            return None
        return Position(
            symbol=item["symbol"],
            side=item["side"],
            entry_price=float(item["entry_price"]),
            size=float(item["size"]),
            entry_time=item["entry_time"],
        )

    def get_trade_history(self, symbol: str, limit: int = 50) -> list[Trade]:
        """거래 이력을 조회한다.

        Args:
            symbol: 거래 심볼.
            limit: 최대 조회 건수.

        Returns:
            Trade 목록 (최신순).
        """
        if self.mode == "local":
            trades = self._trades.get(symbol, [])
            return list(reversed(trades[-limit:]))

        response = self._trades_table.query(
            KeyConditionExpression=Key("symbol").eq(symbol),
            ScanIndexForward=False,
            Limit=limit,
        )
        return [
            Trade(
                symbol=item["symbol"],
                side=item["side"],
                entry_price=float(item["entry_price"]),
                exit_price=float(item["exit_price"]),
                size=float(item["size"]),
                entry_time=item["entry_time"],
                exit_time=item["exit_time"],
                pnl_pct=float(item["pnl_pct"]),
            )
            for item in response.get("Items", [])
        ]

    def get_bot_state(self, key: str) -> str | None:
        """봇 상태 값을 조회한다.

        Args:
            key: 상태 키 (e.g., "running", "last_signal").

        Returns:
            값 문자열 또는 없으면 None.
        """
        if self.mode == "local":
            return self._bot_state.get(key)

        response = self._bot_state_table.get_item(Key={"key": key})
        item = response.get("Item")
        return item["value"] if item else None

    def set_bot_state(self, key: str, value: str) -> None:
        """봇 상태 값을 설정한다.

        Args:
            key: 상태 키.
            value: 값 문자열.
        """
        if self.mode == "local":
            self._bot_state[key] = value
        else:
            self._bot_state_table.put_item(Item={"key": key, "value": value})

        logger.debug(f"봇 상태 설정: {key}={value}")

    def delete_bot_state(self, key: str) -> None:
        """봇 상태 값을 삭제한다.

        Args:
            key: 상태 키.
        """
        if self.mode == "local":
            self._bot_state.pop(key, None)
        else:
            self._bot_state_table.delete_item(Key={"key": key})


def _to_dynamo_item(d: dict) -> dict:
    """Python dict를 DynamoDB 호환 아이템으로 변환한다.

    float → Decimal 변환 (DynamoDB는 float 미지원).

    Args:
        d: 원본 딕셔너리.

    Returns:
        Decimal 변환된 딕셔너리.
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, float):
            result[k] = Decimal(str(v))
        else:
            result[k] = v
    return result
