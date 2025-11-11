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
    """Configuration for trading simulation - ALL parameters configurable"""
    transaction_interval_seconds: float = 0.5  # Time between transactions
    buy_probability: float = 0.50  # 50% buy, 50% sell (BALANCED)
    panic_sell_probability: float = 0.02  # 2% chance (REDUCED from 5%)

    # Buy volumes (INCREASED)
    min_buy_tokens: int = 1
    max_buy_tokens: int = 10000  # INCREASED from 3000 - more buy pressure

    # Sell volumes (REDUCED to prevent secondary depletion)
    min_sell_tokens: int = 1
    max_sell_tokens: int = 30000  # REDUCED from 50000

    # Sell percentage ranges (SIGNIFICANTLY REDUCED)
    min_sell_percentage: float = 0.20  # Small holders sell 20-50% (was 60-100%)
    max_sell_percentage: float = 0.50   # Much more conservative

    # Large holder threshold (CONFIGURABLE)
    large_holder_threshold: int = 1000  # INCREASED from 100 - fewer "large" holders

    # Large holder sell percentages (REDUCED)
    large_holder_min_sell_pct: float = 0.30  # Large holders sell 30-60% (was 70-100%)
    large_holder_max_sell_pct: float = 0.60   # Much more conservative

    # Initial dump configuration (REDUCED and LIMITED)
    initial_dump_probability: float = 0.15  # REDUCED from 45% to 15%
    max_initial_dumps: int = 1000  # LIMIT to prevent blocking with large user counts

    max_consecutive_failures: int = 50  # INCREASED from 20 for large scenarios


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
        """
        Determine if next action should be buy or sell
        DYNAMICALLY ADJUSTED based on secondary market health
        """
        # Get secondary market ratio (0-1)
        total_supply = self.engine.total_supply
        secondary_tokens = self.engine.tokens_available_secondary
        secondary_ratio = secondary_tokens / total_supply if total_supply > 0 else 0

        # Base buy probability (50/50)
        buy_prob = self.config.buy_probability

        # MODERATE DYNAMIC ADJUSTMENT: Only adjust when critically low or very high
        if secondary_ratio < 0.01:  # Less than 1% of supply in secondary
            # Increase buys when critically low
            buy_prob = min(0.80, buy_prob + 0.30)  # Max 80% buy
        elif secondary_ratio < 0.03:  # Less than 3% of supply
            buy_prob = min(0.65, buy_prob + 0.15)  # Moderate increase
        elif secondary_ratio > 0.50:  # More than 50% of supply in secondary
            # Encourage sells when secondary is too high
            buy_prob = max(0.35, buy_prob - 0.15)  # Decrease to 35% buy

        return random.random() < buy_prob
    
    def get_trade_amount(self, user_id: int, is_buy: bool, force_all: bool = False) -> int:
        """
        Get random trade amount - FULLY CONFIGURABLE
        Returns INTEGER tokens only (no fractions)

        Args:
            user_id: User ID
            is_buy: True for buy, False for sell
            force_all: If True and selling, sell all tokens (panic sell)

        Returns:
            Amount of tokens to trade (INTEGER)
        """
        if is_buy:
            # Buy amounts: configurable min/max (already integers)
            return random.randint(self.config.min_buy_tokens, self.config.max_buy_tokens)
        else:
            # SELL amounts - MAXIMIZED for secondary market liquidity
            user_balance = int(self.engine.user_balance.get(user_id, {}).get("tokens", 0))

            if user_balance <= 0:
                return 0

            # Panic sell: sell 100% of holdings (integer)
            if force_all:
                return user_balance

            # Small holders vs Large holders (threshold configurable)
            if user_balance < self.config.large_holder_threshold:
                # Small holders: sell percentage (configurable 20-50%)
                sell_percentage = random.uniform(
                    self.config.min_sell_percentage,
                    self.config.max_sell_percentage
                )
                sell_amount = int(user_balance * sell_percentage)  # Convert to INTEGER
                return min(sell_amount, self.config.max_sell_tokens)
            else:
                # Large holders: sell percentage (configurable 30-60%)
                sell_percentage = random.uniform(
                    self.config.large_holder_min_sell_pct,
                    self.config.large_holder_max_sell_pct
                )
                sell_amount = int(user_balance * sell_percentage)  # Convert to INTEGER
                # Cap to max_sell_tokens if needed
                return min(sell_amount, self.config.max_sell_tokens)
    
    def is_primary_phase(self) -> bool:
        """Check if still in primary market phase"""
        return self.engine.tokens_available_primary > 0

    def execute_initial_dumps(self) -> Dict:
        """
        Execute initial dumps after liquidity_distribution
        Limited number to prevent blocking with large user counts

        Returns:
            Dict with dump statistics
        """
        eligible_users = [uid for uid in self.engine.user_balance.keys() if uid != 0]
        if not eligible_users:
            return {"success": False, "message": "No eligible users", "dumps_executed": 0}

        # Calculate how many users will dump
        target_dumps = int(len(eligible_users) * self.config.initial_dump_probability)
        # LIMIT to prevent blocking with huge user counts
        target_dumps = min(target_dumps, self.config.max_initial_dumps)

        # Calculate actual percentage after applying limit
        actual_percentage = (target_dumps / len(eligible_users) * 100) if eligible_users else 0

        print(f"\n💥 Executing initial dumps...")
        print(f"   Target: {target_dumps:,} users ({actual_percentage:.3f}% of {len(eligible_users):,} total)")
        print(f"   Config: {self.config.initial_dump_probability * 100:.0f}% probability, capped at {self.config.max_initial_dumps:,} max")

        # Randomly select users to dump
        users_to_dump = random.sample(eligible_users, min(target_dumps, len(eligible_users)))

        dumps_executed = 0
        total_tokens_dumped = 0.0
        total_eur_from_dumps = 0.0
        failed_dumps = 0

        for user_id in users_to_dump:
            user_tokens = self.engine.user_balance.get(user_id, {}).get("tokens", 0)

            if user_tokens > 0:
                # DUMP 100% of tokens
                result = self.engine.sell(user_id, user_tokens)

                if result.get("success"):
                    dumps_executed += 1
                    total_tokens_dumped += user_tokens

                    # Track fees and volume
                    last_tx = self.engine.transaction_history[-1] if self.engine.transaction_history else {}
                    payout_eur = last_tx.get("payout_eur", 0)
                    fee = payout_eur * 0.01

                    self.total_fees_generated += fee
                    self.total_volume_eur += payout_eur
                    total_eur_from_dumps += payout_eur

                    # Only log first 10 dumps to avoid spam
                    if dumps_executed <= 10:
                        print(f"   💸 User {user_id} dumped {user_tokens:.0f} tokens → €{payout_eur:.2f}")
                else:
                    failed_dumps += 1
                    if failed_dumps == 1:
                        print(f"   ⚠️ Some dumps failed (e.g., insufficient liquidity)")

        print(f"\n✅ Initial dumps complete:")
        print(f"   Users dumped: {dumps_executed}/{len(eligible_users)} ({dumps_executed/len(eligible_users)*100:.1f}%)")
        print(f"   Tokens dumped: {total_tokens_dumped:,.0f}")
        print(f"   EUR from dumps: €{total_eur_from_dumps:,.2f}")
        print(f"   Secondary market: {self.engine.tokens_available_secondary:,.0f} tokens")
        print(f"   Liquidity remaining: €{self.engine.current_liquidity:,.2f}\n")

        return {
            "success": True,
            "dumps_executed": dumps_executed,
            "total_users": len(eligible_users),
            "dump_percentage": dumps_executed / len(eligible_users) * 100 if eligible_users else 0,
            "total_tokens_dumped": total_tokens_dumped,
            "total_eur_from_dumps": total_eur_from_dumps,
            "secondary_market_after": self.engine.tokens_available_secondary,
            "liquidity_after_dumps": self.engine.current_liquidity
        }
    
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

                # Ensure minimum tokens in secondary before allowing purchase
                if self.engine.tokens_available_secondary < 10:
                    return {"success": False, "reason": "Secondary supply too low (< 10 tokens)"}

                amount = self.get_trade_amount(user_id, is_buy=True)
                # Don't buy more than 90% of secondary supply to maintain liquidity
                max_buyable = self.engine.tokens_available_secondary * 0.90
                amount = min(amount, max_buyable)

                if amount < 1:
                    return {"success": False, "reason": "Calculated amount too small"}

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

        # Market Value = tokens held by USERS (excluding LP) × current price
        # LP tokens should not count towards market cap (they're treasury/reserve)
        lp_tokens = self.engine.user_balance.get(0, {}).get("tokens", 0)
        user_tokens = tokens_in_circulation - lp_tokens
        market_value = user_tokens * current_price

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