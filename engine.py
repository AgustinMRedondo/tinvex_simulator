"""
Tinvex AMM Simulation Engine
Implements the core tokenomics formulas for primary and secondary markets
BASED ON funciones.py - THE CORRECT IMPLEMENTATION
"""

import random
from typing import Dict, List, Optional
from math import sqrt


class SimulationEngine:
    """
    Core simulation engine for Tinvex AMM mechanics

    Key formulas:
    - Primary market: P = Y/X (anchor price)
    - Sell: P = Y/X (no slippage)
    - Secondary buy: P = [(Y*X)*(1 + s*(q/S))] / [X - q^2 * (s/S)]

    IMPORTANT: ALL PURCHASES ADD CAPITAL TO THE LIQUIDITY POOL
    """

    def __init__(
        self,
        total_supply: int = 100,
        initial_price: float = 1.0,
        min_tokens: int = 5,
        max_tokens: int = 30,
        max_slippage: float = 0.05
    ):
        # Configuration parameters
        self.total_supply = total_supply
        self.initial_price = initial_price
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.max_slippage = max_slippage

        # Market state variables
        self.tokens_in_circulation = 0.0
        self.tokens_available_primary = float(total_supply)
        self.tokens_available_secondary = 0.0
        self.current_liquidity = 0.0
        self.current_price = 0.0

        # User and transaction tracking
        self.user_balance: Dict[int, Dict] = {}
        self.users_list: List[Dict] = []
        self.transaction_history: List[Dict] = []

    def reset(self):
        """Reset simulation to initial state"""
        self.tokens_in_circulation = 0.0
        self.tokens_available_primary = float(self.total_supply)
        self.tokens_available_secondary = 0.0
        self.current_liquidity = 0.0
        self.current_price = 0.0
        self.user_balance = {}
        self.users_list = []
        self.transaction_history = []

    def current_info(self) -> Dict:
        """Get current market state"""
        return {
            "total_supply": self.total_supply,
            "tokens_in_circulation": self.tokens_in_circulation,
            "tokens_available_primary": self.tokens_available_primary,
            "tokens_available_secondary": self.tokens_available_secondary,
            "current_liquidity": self.current_liquidity,
            "current_price": self.current_price,
            "user_balance": self.user_balance,
            "transaction_count": len(self.transaction_history)
        }

    def initial_liquidity(self, first_purchase_percentage: float) -> Dict:
        """
        PASO 1: Initial liquidity provision by LP (user_id = 0)
        This is the initial purchase by the issuer - puts capital and receives initial tokens

        Args:
            first_purchase_percentage: Percentage of total supply (0-100)

        Returns:
            Transaction details
        """
        first_purchase_liquidity = (first_purchase_percentage / 100) * self.total_supply

        price = self.initial_price
        tokens_to_purchase = first_purchase_liquidity
        amount_eur = tokens_to_purchase * price

        # Update circulation and primary availability
        self.tokens_in_circulation += tokens_to_purchase
        self.tokens_available_primary -= tokens_to_purchase

        user_id = 0
        user_name = "Liquidity_Provider"

        if user_id not in self.user_balance:
            self.user_balance[user_id] = {"name": user_name, "tokens": 0.0}

        self.user_balance[user_id]["tokens"] += tokens_to_purchase

        # Register transaction
        transaction = {
            'user_id': user_id,
            'user_name': user_name,
            'action': 'liquidity',
            'tokens_bought': tokens_to_purchase,
            'amount_eur': amount_eur,
            'price': price
        }
        self.transaction_history.append(transaction)

        # Update liquidity - ALL CAPITAL GOES TO LIQUIDITY POOL
        self.current_liquidity += amount_eur

        return {
            "success": True,
            "message": f"Initial purchase of {tokens_to_purchase} tokens for {amount_eur} EUR at {price} EUR/token",
            "tokens_in_circulation": self.tokens_in_circulation,
            "tokens_available_primary": self.tokens_available_primary,
            "lp_balance": self.user_balance[user_id]['tokens']
        }

    def create_users(self, total_users: int) -> Dict:
        """
        PASO 2: Create users with random desired token amounts

        Args:
            total_users: Number of users to create

        Returns:
            List of created users
        """
        self.users_list = []

        for i in range(1, total_users + 1):
            name = f"User_{i}"
            desired = random.randint(self.min_tokens, self.max_tokens)

            # List for primary purchases
            self.users_list.append({
                "id": i,
                "name": name,
                "desired_tokens": desired
            })

            # Register in balances with 0 tokens
            self.user_balance[i] = {"name": name, "tokens": 0.0}

        return {
            "success": True,
            "message": f"Created {len(self.users_list)} users",
            "users": self.users_list
        }

    def liquidity_distribution(self, to_distribute: Optional[int] = None) -> Dict:
        """
        PASO 3: Distribute INTEGER tokens from LP (user 0) to other users
        The LP distributes tokens randomly among other users

        Args:
            to_distribute: Number of tokens to distribute (if None, distributes all LP balance)

        Returns:
            Distribution details
        """
        # Validations
        if 0 not in self.user_balance or self.user_balance[0].get("tokens", 0) <= 0:
            return {"success": False, "message": "No tokens in Liquidity Provider (user 0)"}

        lp_tokens = int(self.user_balance[0]["tokens"])
        if lp_tokens <= 0:
            return {"success": False, "message": "LP has no integer tokens to distribute"}

        # Get amount to distribute
        if to_distribute is None:
            to_distribute = lp_tokens
        else:
            to_distribute = int(to_distribute)

        if to_distribute <= 0:
            return {"success": False, "message": "Amount to distribute must be > 0"}

        if to_distribute > lp_tokens:
            to_distribute = lp_tokens

        # Eligible users (all except LP)
        eligible_users = [uid for uid in self.user_balance.keys() if uid != 0]
        if not eligible_users:
            return {"success": False, "message": "No eligible users for distribution"}

        n = len(eligible_users)

        # Random weighted distribution
        weights = [random.random() for _ in eligible_users]
        total_w = sum(weights)
        if total_w == 0:
            proportions = [1 / n] * n
        else:
            proportions = [w / total_w for w in weights]

        # Integer allocation (round down)
        allocations = [int(to_distribute * p) for p in proportions]
        assigned = sum(allocations)

        # Distribute remainder one by one to random users
        remainder = to_distribute - assigned
        if remainder > 0:
            picks = random.sample(eligible_users, k=min(remainder, n))
            idx_map = {uid: i for i, uid in enumerate(eligible_users)}
            for uid in picks:
                allocations[idx_map[uid]] += 1

            # If still remaining (to_distribute > n), distribute in cycles
            remainder_left = to_distribute - sum(allocations)
            i = 0
            while remainder_left > 0:
                allocations[i % n] += 1
                remainder_left -= 1
                i += 1

        # Apply allocations to balances (only integers)
        allocations_out = []
        for uid, alloc in zip(eligible_users, allocations):
            if alloc <= 0:
                continue

            if uid not in self.user_balance:
                self.user_balance[uid] = {"name": f"User_{uid}", "tokens": 0}

            # Add INTEGER
            self.user_balance[uid]["tokens"] = int(self.user_balance[uid].get("tokens", 0)) + int(alloc)
            allocations_out.append({
                "user_id": uid,
                "user_name": self.user_balance[uid]["name"],
                "tokens_received": int(alloc)
            })

        # Subtract from LP exactly what was distributed
        actually_distributed = sum(a["tokens_received"] for a in allocations_out)
        self.user_balance[0]["tokens"] = int(self.user_balance[0]["tokens"]) - actually_distributed

        # Log
        self.transaction_history.append({
            "action": "liquidity_distribution",
            "from_user": 0,
            "from_user_name": self.user_balance[0]["name"],
            "total_distributed": int(actually_distributed),
            "allocations": allocations_out
        })

        return {
            "success": True,
            "message": f"Distributed {actually_distributed} tokens to {len(allocations_out)} users",
            "allocations": allocations_out,
            "lp_balance": self.user_balance[0]["tokens"]
        }

    def purchase_primary(self, users_list: Optional[List[Dict]] = None) -> Dict:
        """
        PASO 4: PRIMARY MARKET PURCHASES
        This marks the beginning of purchases. Only executes if there are tokens in primary market.
        ALL CAPITAL GOES TO LIQUIDITY POOL.

        Formula: P = Y/X (updated after EACH purchase)

        Args:
            users_list: List of users with desired_tokens, if None uses self.users_list

        Returns:
            Purchase execution summary
        """
        if users_list is None:
            users_list = self.users_list

        # Validations
        if self.tokens_in_circulation == 0:
            return {"success": False, "message": "No tokens in circulation. Run initial_liquidity first"}

        if self.tokens_available_primary <= 0:
            return {"success": False, "message": "Primary market closed: no tokens available"}

        purchases = []

        # PRIMARY PRICE: P = Y / X (calculated BEFORE loop, updated AFTER each purchase)
        self.current_price = self.current_liquidity / self.tokens_in_circulation

        for user in users_list:
            user_id = user["id"]
            desired_tokens = user["desired_tokens"]

            # Calculate amount at CURRENT price (price from previous iteration or initial)
            amount_eur = desired_tokens * self.current_price

            if self.tokens_available_primary <= 0:
                break

            # Cap to availability
            if desired_tokens > self.tokens_available_primary:
                desired_tokens = self.tokens_available_primary
                amount_eur = desired_tokens * self.current_price

            if desired_tokens <= 0:
                continue

            # Update global state
            self.tokens_in_circulation += desired_tokens
            self.tokens_available_primary -= desired_tokens
            self.current_liquidity += amount_eur  # ALL CAPITAL TO POOL

            # ✅ UPDATE PRICE AFTER EACH PURCHASE
            self.current_price = self.current_liquidity / self.tokens_in_circulation

            # Update user balance
            if user_id not in self.user_balance:
                self.user_balance[user_id] = {"name": user["name"], "tokens": 0}

            self.user_balance[user_id]["tokens"] += desired_tokens

            # Register transaction
            transaction = {
                "user_id": user_id,
                "user_name": user["name"],
                "action": "purchase_primary",
                "tokens_bought": desired_tokens,
                "amount_eur": amount_eur,
                "price": self.current_price
            }
            self.transaction_history.append(transaction)
            purchases.append(transaction)

        return {
            "success": True,
            "message": f"Executed {len(purchases)} primary purchases",
            "purchases": purchases,
            "current_price": self.current_price,
            "tokens_available_primary": self.tokens_available_primary
        }

    def sell(self, user_id: int, tokens_to_sell: float) -> Dict:
        """
        PASO 5: SELL FUNCTION
        Sales are against the liquidity pool and these sold tokens go to secondary market.
        The sale price becomes the execution price.

        Formula: P = Y/X (anchor price, no slippage)

        Args:
            user_id: User selling tokens
            tokens_to_sell: Amount to sell

        Returns:
            Sale execution details
        """
        # Validations
        if tokens_to_sell <= 0:
            return {"success": False, "message": "tokens_to_sell must be > 0"}

        if user_id not in self.user_balance or self.user_balance[user_id].get("tokens", 0) <= 0:
            return {"success": False, "message": f"User {user_id} has no tokens to sell"}

        if self.tokens_in_circulation <= 0:
            return {"success": False, "message": "No tokens in circulation to sell"}

        # Execution price: anchor price Y/X
        exec_price = self.current_liquidity / self.tokens_in_circulation

        # Cap quantity to user balance and circulating supply
        user_tokens = float(self.user_balance[user_id]["tokens"])
        q = min(float(tokens_to_sell), user_tokens, float(self.tokens_in_circulation))

        if q <= 0:
            return {"success": False, "message": "Nothing to sell after limits"}

        # Payout and cap by available liquidity
        payout_eur = q * exec_price
        if payout_eur > float(self.current_liquidity):
            # Reduce to maximum payable with available liquidity
            q = float(self.current_liquidity) / exec_price
            payout_eur = q * exec_price

        if q <= 0 or payout_eur <= 0:
            return {"success": False, "message": "Cannot execute sale with available liquidity"}

        # Update state
        self.user_balance[user_id]["tokens"] -= q
        self.tokens_in_circulation -= q  # X decreases
        self.tokens_available_secondary += q  # S increases
        self.current_liquidity -= payout_eur  # Y decreases
        # Note: current_price NOT updated here (will be calculated in next operation)

        # Register transaction
        transaction = {
            "user_id": user_id,
            "user_name": self.user_balance[user_id].get("name", f"User_{user_id}"),
            "action": "sell_simple",
            "tokens_sold": q,
            "payout_eur": payout_eur,
            "exec_price": exec_price
        }
        self.transaction_history.append(transaction)

        return {
            "success": True,
            "message": f"User {user_id} sold {q:.6f} tokens at {exec_price:.6f} EUR/token",
            "transaction": transaction,
            "user_balance": self.user_balance[user_id]["tokens"],
            "tokens_available_secondary": self.tokens_available_secondary
        }

    def purchase_secondary(self, user_id: int, quantity_to_buy: float) -> Dict:
        """
        PASO 6: SECONDARY MARKET PURCHASES
        Purchases in secondary market with slippage formula.
        ALL CAPITAL GOES TO LIQUIDITY POOL.

        Formula: P = [(Y*X)*(1 + s*(q/S))] / [X - q^2 * (s/S)]
        Where:
            Y = current_price * tokens_in_circulation
            X = tokens_in_circulation
            S = tokens_available_secondary
            s = max_slippage
            q = quantity_to_buy

        Args:
            user_id: Buyer user ID
            quantity_to_buy: Amount to purchase

        Returns:
            Purchase execution details
        """
        # Validations
        if quantity_to_buy <= 0:
            return {"success": False, "message": "quantity_to_buy must be > 0"}

        if self.tokens_available_secondary <= 0:
            return {"success": False, "message": "No supply in secondary market"}

        if self.tokens_in_circulation <= 0 or self.current_liquidity < 0:
            return {"success": False, "message": "Invalid state: ensure X>0 and Y≥0"}

        # Ensure user in balances
        if user_id not in self.user_balance:
            self.user_balance[user_id] = {"name": f"User_{user_id}", "tokens": 0.0}

        # Calculate Y from current price and circulation
        Y = self.current_price * self.tokens_in_circulation
        X = float(self.tokens_in_circulation)
        S = float(self.tokens_available_secondary)
        s = float(self.max_slippage)
        q_req = float(quantity_to_buy)

        # Cap to secondary supply
        q = min(q_req, S)

        # Avoid denominator <= 0: den = X - q^2 * (s/S) > 0
        if s > 0:
            # q_max_den ~ sqrt(X * S / s)
            q_max_den = sqrt((X * S) / s) if X > 0 else 0.0
            if q >= q_max_den:
                q = max(0.0, min(q, q_max_den - 1e-9))  # small margin

        # If s == 0, force s>0 minimally
        if s <= 0:
            s = 1e-9

        if q <= 0:
            return {"success": False, "message": "Executable quantity is 0 after caps"}

        # Calculate execution price using THE FORMULA
        numerator = (Y) * (1.0 + s * (q / S))
        denominator = X - (q ** 2) * (s / S)

        if denominator <= 0:
            return {"success": False, "message": "Denominator not positive: reduce quantity"}

        exec_price = numerator / denominator

        if exec_price <= 0:
            return {"success": False, "message": "Calculated price not positive"}

        # Calculate cost and update state
        cost_eur = q * exec_price

        self.tokens_available_secondary -= q  # S decreases
        self.tokens_in_circulation += q  # X increases
        self.current_liquidity += cost_eur  # Y increases - ALL CAPITAL TO POOL
        self.current_price = exec_price  # Update price
        self.user_balance[user_id]["tokens"] += q

        # Log transaction
        transaction = {
            "user_id": user_id,
            "user_name": self.user_balance[user_id]["name"],
            "action": "buy_from_secondary",
            "tokens_bought": q,
            "amount_eur": cost_eur,
            "exec_price": exec_price,
            "Y_before": Y,
            "X_before": X,
            "S_before": S,
            "slippage": s
        }
        self.transaction_history.append(transaction)

        return {
            "success": True,
            "message": f"{self.user_balance[user_id]['name']} bought {q:.6f} tokens in secondary at {exec_price:.6f} EUR/token",
            "transaction": transaction,
            "current_state": {
                "tokens_available_secondary": self.tokens_available_secondary,
                "current_liquidity": self.current_liquidity,
                "current_price": self.current_price,
                "tokens_in_circulation": self.tokens_in_circulation
            }
        }

    def create_sell_orders(self) -> List[Dict]:
        """
        Generate random sell orders from users with tokens
        Excludes LP (user 0)

        Returns:
            List of sell orders
        """
        sell_orders = []

        for uid, info in self.user_balance.items():
            if uid == 0:  # Exclude LP
                continue

            tokens_owned = float(info.get("tokens", 0))
            if tokens_owned <= 0:
                continue

            tokens_to_sell = random.randint(1, int(tokens_owned))

            sell_orders.append({
                "id": uid,
                "tokens_to_sell": tokens_to_sell
            })

        return sell_orders

    def create_secondary_buy_orders(self, num_orders: Optional[int] = None) -> List[Dict]:
        """
        Generate simple buy orders for secondary market

        Args:
            num_orders: Number of orders to create (if None, all eligible users)

        Returns:
            List of buy orders
        """
        A = int(self.tokens_available_secondary)
        if A <= 0:
            return []

        candidates = [uid for uid in self.user_balance.keys() if uid != 0]
        if not candidates:
            return []

        if num_orders is None:
            selected = candidates[:]
        else:
            num_orders = max(1, int(num_orders))
            if len(candidates) >= num_orders:
                selected = random.sample(candidates, num_orders)
            else:
                selected = [random.choice(candidates) for _ in range(num_orders)]

        n = len(selected)
        per_user_cap = max(1, A // n)

        orders = []
        for uid in selected:
            name = self.user_balance.get(uid, {}).get("name", f"User_{uid}")
            qty = random.randint(1, per_user_cap)
            orders.append({"id": uid, "name": name, "quantity_to_buy": qty})

        return orders

    def execute_sell_orders(self, orders: List[Dict]) -> Dict:
        """Execute multiple sell orders"""
        results = []
        for order in orders:
            result = self.sell(order["id"], order["tokens_to_sell"])
            results.append(result)

        successful = sum(1 for r in results if r.get("success"))

        return {
            "success": True,
            "message": f"Executed {successful}/{len(orders)} sell orders",
            "results": results
        }

    def execute_secondary_buy_orders(self, orders: List[Dict]) -> Dict:
        """Execute multiple secondary buy orders"""
        results = []
        for order in orders:
            result = self.purchase_secondary(order["id"], order["quantity_to_buy"])
            results.append(result)

        successful = sum(1 for r in results if r.get("success"))

        return {
            "success": True,
            "message": f"Executed {successful}/{len(orders)} secondary buy orders",
            "results": results
        }
