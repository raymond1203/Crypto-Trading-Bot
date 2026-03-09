"""포지션 관리 모듈 단위 테스트."""

import pytest

from src.strategy.portfolio import PortfolioManager, Position, Trade


class TestOpenPosition:
    """open_position 테스트."""

    def test_open_long_position(self) -> None:
        """롱 포지션을 정상적으로 연다."""
        pm = PortfolioManager(mode="local")
        pos = pm.open_position("BTC/USDT", "long", 50000.0, 0.5)

        assert isinstance(pos, Position)
        assert pos.symbol == "BTC/USDT"
        assert pos.side == "long"
        assert pos.entry_price == 50000.0
        assert pos.size == 0.5

    def test_open_short_position(self) -> None:
        """숏 포지션을 정상적으로 연다."""
        pm = PortfolioManager(mode="local")
        pos = pm.open_position("BTC/USDT", "short", 50000.0, 0.3)

        assert pos.side == "short"
        assert pos.size == 0.3

    def test_duplicate_open_raises(self) -> None:
        """같은 심볼에 중복 오픈 시 ValueError."""
        pm = PortfolioManager(mode="local")
        pm.open_position("BTC/USDT", "long", 50000.0, 0.5)

        with pytest.raises(ValueError, match="이미 존재"):
            pm.open_position("BTC/USDT", "long", 51000.0, 0.3)

    def test_open_different_symbols(self) -> None:
        """다른 심볼은 각각 오픈 가능."""
        pm = PortfolioManager(mode="local")
        pm.open_position("BTC/USDT", "long", 50000.0, 0.5)
        pm.open_position("ETH/USDT", "long", 3000.0, 0.3)

        assert pm.get_current_position("BTC/USDT") is not None
        assert pm.get_current_position("ETH/USDT") is not None


class TestClosePosition:
    """close_position 테스트."""

    def test_close_long_profit(self) -> None:
        """롱 포지션 수익 청산."""
        pm = PortfolioManager(mode="local")
        pm.open_position("BTC/USDT", "long", 50000.0, 0.5)
        trade = pm.close_position("BTC/USDT", 55000.0)

        assert isinstance(trade, Trade)
        assert trade.pnl_pct == pytest.approx(10.0, abs=0.01)
        assert pm.get_current_position("BTC/USDT") is None

    def test_close_long_loss(self) -> None:
        """롱 포지션 손실 청산."""
        pm = PortfolioManager(mode="local")
        pm.open_position("BTC/USDT", "long", 50000.0, 0.5)
        trade = pm.close_position("BTC/USDT", 48000.0)

        assert trade.pnl_pct == pytest.approx(-4.0, abs=0.01)

    def test_close_short_profit(self) -> None:
        """숏 포지션 수익 청산."""
        pm = PortfolioManager(mode="local")
        pm.open_position("BTC/USDT", "short", 50000.0, 0.5)
        trade = pm.close_position("BTC/USDT", 45000.0)

        assert trade.pnl_pct == pytest.approx(10.0, abs=0.01)

    def test_close_no_position_raises(self) -> None:
        """오픈 포지션 없이 청산 시 ValueError."""
        pm = PortfolioManager(mode="local")

        with pytest.raises(ValueError, match="오픈 포지션이 없습니다"):
            pm.close_position("BTC/USDT", 50000.0)


class TestGetPosition:
    """get_current_position 테스트."""

    def test_no_position_returns_none(self) -> None:
        """포지션 없으면 None."""
        pm = PortfolioManager(mode="local")
        assert pm.get_current_position("BTC/USDT") is None

    def test_get_open_position(self) -> None:
        """오픈 포지션 조회."""
        pm = PortfolioManager(mode="local")
        pm.open_position("BTC/USDT", "long", 50000.0, 0.5)

        pos = pm.get_current_position("BTC/USDT")
        assert pos is not None
        assert pos.entry_price == 50000.0


class TestTradeHistory:
    """get_trade_history 테스트."""

    def test_empty_history(self) -> None:
        """거래 이력 없으면 빈 리스트."""
        pm = PortfolioManager(mode="local")
        assert pm.get_trade_history("BTC/USDT") == []

    def test_history_after_trades(self) -> None:
        """거래 후 이력 조회."""
        pm = PortfolioManager(mode="local")
        pm.open_position("BTC/USDT", "long", 50000.0, 0.5)
        pm.close_position("BTC/USDT", 52000.0)
        pm.open_position("BTC/USDT", "long", 53000.0, 0.5)
        pm.close_position("BTC/USDT", 51000.0)

        history = pm.get_trade_history("BTC/USDT")
        assert len(history) == 2
        # 최신순
        assert history[0].pnl_pct < 0  # 두 번째 거래: 손실
        assert history[1].pnl_pct > 0  # 첫 번째 거래: 수익

    def test_history_limit(self) -> None:
        """이력 조회 limit 적용."""
        pm = PortfolioManager(mode="local")
        for i in range(5):
            pm.open_position("BTC/USDT", "long", 50000.0 + i * 100, 0.5)
            pm.close_position("BTC/USDT", 50100.0 + i * 100)

        history = pm.get_trade_history("BTC/USDT", limit=3)
        assert len(history) == 3


class TestBotState:
    """봇 상태 관리 테스트."""

    def test_set_and_get(self) -> None:
        """상태 설정 후 조회."""
        pm = PortfolioManager(mode="local")
        pm.set_bot_state("running", "true")

        assert pm.get_bot_state("running") == "true"

    def test_get_nonexistent(self) -> None:
        """존재하지 않는 키 조회 시 None."""
        pm = PortfolioManager(mode="local")
        assert pm.get_bot_state("nonexistent") is None

    def test_delete_state(self) -> None:
        """상태 삭제."""
        pm = PortfolioManager(mode="local")
        pm.set_bot_state("running", "true")
        pm.delete_bot_state("running")

        assert pm.get_bot_state("running") is None
