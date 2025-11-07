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
    min_tokens: int = 1
    max_tokens: int = 10000


class TradingEngine:
    """
    Continuous trading engine - one transaction at a time
    """
    
    def __init__(self, simulation_engine, config: Optional[TradeConfig] = None):
        """
        Initialize trading engine
        
        Args:
            simulation_engine: Instance of SimulationEngine
            config: Trading configuration
        """
        self.engine = simulation_engine
        self.config = config or TradeConfig()
        
        self.is_running = False
        self.trade_count = 0
        self.price_history: List[Dict] = []
    
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
            # Random buy amount
            return random.randint(self.config.min_tokens, self.config.max_tokens)
        else:
            # Random sell amount (capped to user balance)
            user_balance = self.engine.user_balance.get(user_id, {}).get("tokens", 0)

            if user_balance <= 0:
                return 0

            # Panic sell: sell 100% of holdings
            if force_all:
                return user_balance

            max_sell = min(user_balance, self.config.max_tokens)
            return random.uniform(1, max_sell) if max_sell >= 1 else 0
    
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
                    self.price_history.append({
                        "trade_number": self.trade_count,
                        "price": self.engine.current_price,
                        "timestamp": time.time(),
                        "action": "buy_primary"
                    })
                    return {
                        "success": True,
                        "trade_number": self.trade_count,
                        "user_id": user_id,
                        "action": "buy_primary",
                        "amount": amount,
                        "price": self.engine.current_price,
                        "timestamp": time.time()
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
                    self.price_history.append({
                        "trade_number": self.trade_count,
                        "price": self.engine.current_price,
                        "timestamp": time.time(),
                        "action": "buy_secondary"
                    })
                    return {
                        "success": True,
                        "trade_number": self.trade_count,
                        "user_id": user_id,
                        "action": "buy_secondary",
                        "amount": amount,
                        "price": self.engine.current_price,
                        "timestamp": time.time()
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
                self.price_history.append({
                    "trade_number": self.trade_count,
                    "price": self.engine.current_price,
                    "timestamp": time.time(),
                    "action": "sell"
                })
                return {
                    "success": True,
                    "trade_number": self.trade_count,
                    "user_id": user_id,
                    "action": "sell",
                    "amount": amount,
                    "price": self.engine.current_price,
                    "timestamp": time.time()
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
        
        while self.is_running:
            # Execute single trade
            result = self.execute_single_trade()
            
            # Log if successful
            if result.get("success"):
                action_emoji = "📈" if "buy" in result["action"] else "📉"
                print(f"{action_emoji} Trade #{self.trade_count}: {result['action']} | "
                      f"Amount: {result['amount']:.0f} | Price: €{result['price']:.4f}")
            
            # Wait before next transaction
            await asyncio.sleep(self.config.transaction_interval_seconds)
    
    def stop(self):
        """Stop the continuous trading simulation"""
        self.is_running = False
        print(f"\n⏹️  Trading stopped - Total trades: {self.trade_count}")
    
    def get_stats(self) -> Dict:
        """Get trading statistics"""
        if not self.price_history:
            return {
                "total_trades": 0,
                "price_change": 0,
                "price_min": 0,
                "price_max": 0
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
            "price_history": self.price_history[-100:]  # Last 100 for chart
        }