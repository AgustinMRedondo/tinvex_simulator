"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# Request models
class InitRequest(BaseModel):
    """Initial liquidity setup"""
    first_purchase_percentage: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Percentage of total supply for initial liquidity (0-100)"
    )


class CreateUsersRequest(BaseModel):
    """Create users for simulation"""
    total_users: int = Field(..., ge=1, description="Number of users to create (no upper limit)")


class LiquidityDistributionRequest(BaseModel):
    """Distribute tokens from LP to users"""
    to_distribute: Optional[int] = Field(None, ge=1, description="Tokens to distribute (None = all LP balance)")


class BuyPrimaryRequest(BaseModel):
    """Purchase tokens in primary market"""
    users_list: Optional[List[Dict]] = Field(None, description="List of users with desired_tokens")


class SellRequest(BaseModel):
    """Sell tokens"""
    user_id: int = Field(..., ge=0, description="User ID selling tokens")
    tokens_to_sell: float = Field(..., gt=0, description="Amount of tokens to sell")


class BuySecondaryRequest(BaseModel):
    """Buy tokens in secondary market"""
    user_id: int = Field(..., ge=0, description="User ID buying tokens")
    quantity_to_buy: float = Field(..., gt=0, description="Amount of tokens to buy")


class SimulateRequest(BaseModel):
    """Run automated simulation"""
    num_transactions: int = Field(10, ge=1, le=100, description="Number of transactions to simulate")
    transaction_types: List[str] = Field(
        ["primary", "sell", "secondary"],
        description="Types of transactions to include"
    )


class AutoTradingSetupRequest(BaseModel):
    """Setup auto-trading simulation - ALL parameters required and configurable"""
    total_supply: int = Field(..., gt=0, description="Total token supply")
    num_users: int = Field(..., gt=0, description="Number of users to create")
    initial_liquidity_percentage: float = Field(..., gt=0, le=100, description="% of supply for initial liquidity")
    max_slippage_percentage: float = Field(..., gt=0, le=100, description="Maximum slippage % for secondary market")

    # Trading probabilities
    buy_probability_percentage: float = Field(..., ge=0, le=100, description="% chance of buy vs sell")
    panic_sell_probability_percentage: float = Field(..., ge=0, le=100, description="% chance of selling 100%")
    initial_dump_probability_percentage: float = Field(..., ge=0, le=100, description="% of users who dump 100% at start")

    # Volume controls
    min_buy_tokens: int = Field(..., gt=0, description="Minimum tokens per buy")
    max_buy_tokens: int = Field(..., gt=0, description="Maximum tokens per buy")
    min_sell_tokens: int = Field(..., gt=0, description="Minimum tokens per sell")
    max_sell_tokens: int = Field(..., gt=0, description="Maximum tokens per sell")

    # Sell percentage controls
    min_sell_percentage: float = Field(..., ge=0, le=100, description="Min % of balance to sell (small holders)")
    max_sell_percentage: float = Field(..., ge=0, le=100, description="Max % of balance to sell (small holders)")
    large_holder_threshold: int = Field(..., gt=0, description="Token threshold to be considered large holder")
    large_holder_min_sell_pct: float = Field(..., ge=0, le=100, description="Min % to sell (large holders)")
    large_holder_max_sell_pct: float = Field(..., ge=0, le=100, description="Max % to sell (large holders)")

    # Failure handling
    max_consecutive_failures: int = Field(..., gt=0, description="Max consecutive failures before stopping")

    # Fees
    fee_percentage: float = Field(..., gt=0, le=100, description="Fee % on transactions (e.g., 1.0 for 1%)")

    # Logging
    max_dump_logs: int = Field(..., gt=0, description="Maximum dump logs to display")

    # Dynamic adjustments
    dynamic_adjustment_enabled: bool = Field(..., description="Enable dynamic buy/sell probability adjustments")
    secondary_critical_ratio: float = Field(..., ge=0, le=1, description="Critical secondary ratio threshold (decimal)")
    secondary_low_ratio: float = Field(..., ge=0, le=1, description="Low secondary ratio threshold (decimal)")
    secondary_high_ratio: float = Field(..., ge=0, le=1, description="High secondary ratio threshold (decimal)")
    buy_boost_critical: float = Field(..., ge=0, le=1, description="Buy probability reduction when critical (decimal)")
    buy_boost_low: float = Field(..., ge=0, le=1, description="Buy probability reduction when low (decimal)")
    buy_reduce_high: float = Field(..., ge=0, le=1, description="Buy probability increase when high (decimal)")
    max_buy_probability: float = Field(..., ge=0, le=1, description="Maximum buy probability cap (decimal)")
    min_buy_probability: float = Field(..., ge=0, le=1, description="Minimum buy probability floor (decimal)")

    # Secondary protection
    min_secondary_tokens: int = Field(..., gt=0, description="Minimum tokens to keep in secondary market")
    max_buyable_ratio: float = Field(..., ge=0, le=1, description="Max % of secondary market buyable in one tx (decimal)")

    # Chart config
    candlestick_interval_seconds: int = Field(..., gt=0, description="Candlestick interval in seconds")
    max_price_history: int = Field(..., gt=0, description="Maximum price history points to keep")

    transaction_interval_seconds: float = Field(..., gt=0, description="Seconds between transactions")


# Response models
class TransactionResponse(BaseModel):
    """Single transaction details"""
    user_id: int
    user_name: str
    action: str
    tokens_bought: Optional[float] = None
    tokens_sold: Optional[float] = None
    amount_eur: Optional[float] = None
    payout_eur: Optional[float] = None
    price: Optional[float] = None
    exec_price: Optional[float] = None


class MarketStateResponse(BaseModel):
    """Current market state"""
    total_supply: int
    tokens_in_circulation: float
    tokens_available_primary: float
    tokens_available_secondary: float
    current_liquidity: float
    current_price: float
    user_balance: Dict[int, Dict[str, Any]]
    transaction_count: int


class ApiResponse(BaseModel):
    """Generic API response wrapper"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    details: Optional[str] = None