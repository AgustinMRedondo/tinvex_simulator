"""
Continuous Auto-Trading Engine for Tinvex AMM Simulator
Simple continuous trading - one transaction at a time
"""

import asyncio
import random
import time
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TradeConfig:
    """Configuration for trading simulation"""
    transaction_interval_seconds: float = 0.5  # Time between transactions
    buy_probability: float = 0.55  # 55% buy, 45% sell
    panic_sell_probability: float = 0.05  # 5% chance of selling 100% of holdings
    min_buy_tokens: int = 1
    max_buy_tokens: int = 5000  # REDUCED from 15000 to 5000 - smaller buy volumes
    min_sell_tokens: int = 1
    max_sell_tokens: int = 20000  # INCREASED to 20000 - larger sell volumes for more secondary liquidity
    max_consecutive_failures: int = 20  # Stop after this many consecutive failed trades


class TradingEngine:
    """
    Continuous trading engine - one transaction at a time
    """
    
    def __init__(self, simulation_engine, config: Optional[TradeConfig] = None, initial_fees: float = 0.0, initial_volume: float = 0.0):
        """
        Initialize trading engine

        Args:
            simulation_engine: Instance of SimulationEngine
            config: Trading configuration
            initial_fees: Initial fees from setup (liquidity pool creation)
            initial_volume: Initial volume from liquidity pool creation
        """
        self.engine = simulation_engine
        self.config = config or TradeConfig()

        self.is_running = False
        self.trade_count = 0
        self.price_history: List[Dict] = []
        self.total_fees_generated = initial_fees  # Start with initial fees from liquidity pool
        self.total_volume_eur = initial_volume  # Track total volume (sum of all transaction values in EUR)
    
    def get_random_user(self) -> Optional[int]:
        """Get a random user ID (excluding LP)"""
        eligible_users = [uid for uid in self.engine.user_balance.keys() if uid != 0]
        return random.choice(eligible_users) if eligible_users else None
    
    def should_buy(self) -> bool:
        """Determine if next action should be buy (55%) or sell (45%)"""
        return random.random() < self.config.buy_probability
    
    def get_trade_amount(self, user_id: int, is_buy: bool, force_all: bool = False) -> float:
        """
        Get random trade amount

        Args:
            user_id: User ID
            is_buy: True for buy, False for sell
            force_all: If True and selling, sell all tokens (panic sell)

        Returns:
            Amount of tokens to trade
        """
        if is_buy:
            # REDUCED buy amounts: 1 to 5,000 tokens for lower buy pressure
            return random.randint(self.config.min_buy_tokens, self.config.max_buy_tokens)
        else:
            # INCREASED sell amounts for more secondary market liquidity
            user_balance = self.engine.user_balance.get(user_id, {}).get("tokens", 0)

            if user_balance <= 0:
                return 0

            # Panic sell: sell 100% of holdings
            if force_all:
                return user_balance

            # IMPROVED: Holders sell LARGER quantities (40-95% of balance)
            # This ensures MUCH more tokens go to secondary market
            if user_balance < 100:
                # Small holders: sell 1 to all their tokens
                max_sell = min(user_balance, self.config.max_sell_tokens)
                return random.uniform(1, max_sell) if max_sell >= 1 else 0
            else:
                # Larger holders: sell 40% to 95% of their balance (INCREASED)
                min_sell_pct = 0.40  # 40% (was 20%)
                max_sell_pct = 0.95  # 95% (was 80%)
                sell_percentage = random.uniform(min_sell_pct, max_sell_pct)
                sell_amount = user_balance * sell_percentage
                # Cap to max_sell_tokens if needed
                return min(sell_amount, self.config.max_sell_tokens)
    
    def is_primary_phase(self) -> bool:
        """Check if still in primary market phase"""
        return self.engine.tokens_available_primary > 0
    
    def execute_single_trade(self) -> Dict:
        """
        Execute a single trade (buy or sell)

        Returns:
            Trade result dictionary
        """
        user_id = self.get_random_user()
        if not user_id:
            return {"success": False, "reason": "No eligible users"}

        should_buy = self.should_buy()

        # ✅ ALLOW SELLS DURING PRIMARY PHASE
        # Users can sell tokens they received from liquidity_distribution
        # while primary market is still available

        if should_buy:
            # Try to buy from primary first, then secondary
            if self.is_primary_phase():
                # BUY FROM PRIMARY
                amount = self.get_trade_amount(user_id, is_buy=True)
                amount = min(amount, self.engine.tokens_available_primary)

                if amount <= 0:
                    return {"success": False, "reason": "No primary tokens available"}

                result = self.engine.purchase_primary([{
                    "id": user_id,
                    "name": self.engine.user_balance[user_id]["name"],
                    "desired_tokens": amount
                }])

                if result.get("success"):
                    self.trade_count += 1

                    # Calculate fees: 1% of purchase value (does NOT affect AMM price/formulas)
                    last_tx = self.engine.transaction_history[-1] if self.engine.transaction_history else {}
                    amount_eur = last_tx.get("amount_eur", 0)
                    fee = amount_eur * 0.01
                    self.total_fees_generated += fee

                    # Track volume: add transaction value to total volume
                    self.total_volume_eur += amount_eur

                    self.price_history.append({
                        "trade_number": self.trade_count,
                        "price": self.engine.current_price,
                        "timestamp": time.time(),
                        "action": "buy_primary",
                        "fee": fee
                    })
                    return {
                        "success": True,
                        "trade_number": self.trade_count,
                        "user_id": user_id,
                        "action": "buy_primary",
                        "amount": amount,
                        "price": self.engine.current_price,
                        "timestamp": time.time(),
                        "fee": fee
                    }
                return {"success": False, "reason": result.get("message")}

            else:
                # BUY FROM SECONDARY
                if self.engine.tokens_available_secondary <= 0:
                    return {"success": False, "reason": "No secondary supply"}

                amount = self.get_trade_amount(user_id, is_buy=True)
                amount = min(amount, self.engine.tokens_available_secondary)

                result = self.engine.purchase_secondary(user_id, amount)

                if result.get("success"):
                    self.trade_count += 1

                    # Calculate fees: 1% of purchase value (does NOT affect AMM price/formulas)
                    last_tx = self.engine.transaction_history[-1] if self.engine.transaction_history else {}
                    amount_eur = last_tx.get("amount_eur", 0)
                    fee = amount_eur * 0.01
                    self.total_fees_generated += fee

                    # Track volume: add transaction value to total volume
                    self.total_volume_eur += amount_eur

                    self.price_history.append({
                        "trade_number": self.trade_count,
                        "price": self.engine.current_price,
                        "timestamp": time.time(),
                        "action": "buy_secondary",
                        "fee": fee
                    })
                    return {
                        "success": True,
                        "trade_number": self.trade_count,
                        "user_id": user_id,
                        "action": "buy_secondary",
                        "amount": amount,
                        "price": self.engine.current_price,
                        "timestamp": time.time(),
                        "fee": fee
                    }
                return {"success": False, "reason": result.get("message")}

        else:
            # SELL - allowed in BOTH primary and secondary phases
            is_panic_sell = random.random() < self.config.panic_sell_probability
            amount = self.get_trade_amount(user_id, is_buy=False, force_all=is_panic_sell)

            if amount <= 0:
                return {"success": False, "reason": "User has no tokens to sell"}

            result = self.engine.sell(user_id, amount)

            if result.get("success"):
                self.trade_count += 1

                # Calculate fees: 1% of sell payout (does NOT affect AMM price/formulas)
                last_tx = self.engine.transaction_history[-1] if self.engine.transaction_history else {}
                payout_eur = last_tx.get("payout_eur", 0)
                fee = payout_eur * 0.01
                self.total_fees_generated += fee

                # Track volume: add sell payout to total volume
                self.total_volume_eur += payout_eur

                self.price_history.append({
                    "trade_number": self.trade_count,
                    "price": self.engine.current_price,
                    "timestamp": time.time(),
                    "action": "sell",
                    "fee": fee
                })
                return {
                    "success": True,
                    "trade_number": self.trade_count,
                    "user_id": user_id,
                    "action": "sell",
                    "amount": amount,
                    "price": self.engine.current_price,
                    "timestamp": time.time(),
                    "fee": fee
                }
            return {"success": False, "reason": result.get("message")}
    
    async def run_continuous(self):
        """
        Run continuous trading simulation - one transaction at a time
        """
        print("\n🚀 Starting continuous trading simulation...")
        print(f"   Transaction interval: {self.config.transaction_interval_seconds}s")
        print(f"   Buy probability: {self.config.buy_probability * 100}%")

        self.is_running = True
        consecutive_failures = 0

        while self.is_running:
            # Execute single trade
            result = self.execute_single_trade()

            # Log if successful
            if result.get("success"):
                consecutive_failures = 0  # Reset failure counter
                action_emoji = "📈" if "buy" in result["action"] else "📉"
                fee_display = f" | Fee: €{result.get('fee', 0):.2f}" if result.get('fee') else ""
                print(f"{action_emoji} Trade #{self.trade_count}: {result['action']} | "
                      f"Amount: {result['amount']:.0f} | Price: €{result['price']:.4f}{fee_display}")
            else:
                consecutive_failures += 1
                if consecutive_failures >= self.config.max_consecutive_failures:
                    print(f"\n⚠️  {consecutive_failures} consecutive trade failures - stopping to prevent infinite loop")
                    print(f"   Last failure reason: {result.get('reason', 'unknown')}")
                    self.is_running = False
                    break

            # Wait before next transaction
            await asyncio.sleep(self.config.transaction_interval_seconds)
    
    def stop(self):
        """Stop the continuous trading simulation"""
        self.is_running = False
        print(f"\n⏹️  Trading stopped - Total trades: {self.trade_count}")
    
    def get_candlestick_data(self, interval_seconds: int = 60) -> List[Dict]:
        """
        Generate candlestick (OHLC) data from price history

        Args:
            interval_seconds: Time interval for each candle (default 60 = 1 minute)

        Returns:
            List of candlestick data [{time, open, high, low, close}, ...]
        """
        if not self.price_history:
            return []

        candles = []
        current_candle = None
        candle_start_time = None

        for trade in self.price_history:
            trade_time = trade["timestamp"]
            price = trade["price"]

            # Determine which candle this trade belongs to
            if candle_start_time is None:
                candle_start_time = trade_time

            # Check if we need to start a new candle
            if trade_time >= candle_start_time + interval_seconds:
                # Close current candle
                if current_candle:
                    candles.append(current_candle)

                # Start new candle
                candle_start_time = trade_time
                current_candle = {
                    "time": candle_start_time,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price
                }
            else:
                # Update current candle
                if current_candle is None:
                    current_candle = {
                        "time": candle_start_time,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price
                    }
                else:
                    current_candle["high"] = max(current_candle["high"], price)
                    current_candle["low"] = min(current_candle["low"], price)
                    current_candle["close"] = price

        # Add last candle
        if current_candle:
            candles.append(current_candle)

        return candles

    def get_stats(self) -> Dict:
        """Get trading statistics"""
        # Calculate market value and liquidity ratio
        current_price = self.engine.current_price
        tokens_in_circulation = self.engine.tokens_in_circulation
        current_liquidity = self.engine.current_liquidity

        # Market Value = tokens in circulation × current price
        market_value = tokens_in_circulation * current_price

        # Liquidity vs Market Value ratio (%)
        liquidity_ratio = (current_liquidity / market_value * 100) if market_value > 0 else 0

        if not self.price_history:
            return {
                "total_trades": 0,
                "price_change": 0,
                "price_min": 0,
                "price_max": 0,
                "total_fees_generated": 0.0,
                "total_volume_eur": self.total_volume_eur,
                "market_value": market_value,
                "liquidity_ratio_percent": liquidity_ratio
            }

        prices = [p["price"] for p in self.price_history]
        initial_price = prices[0]
        final_price = prices[-1]

        return {
            "total_trades": self.trade_count,
            "initial_price": initial_price,
            "final_price": final_price,
            "price_change_percent": ((final_price - initial_price) / initial_price * 100) if initial_price > 0 else 0,
            "price_min": min(prices),
            "price_max": max(prices),
            "price_history": self.price_history[-100:],  # Last 100 for chart
            "candlestick_data": self.get_candlestick_data(60),  # 1 minute candles
            "total_fees_generated": self.total_fees_generated,
            "total_volume_eur": self.total_volume_eur,  # NEW: Total trading volume
            "market_value": market_value,  # NEW: Market capitalization
            "liquidity_ratio_percent": liquidity_ratio  # NEW: Liquidity/Market Value %
        }