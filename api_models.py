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
    """SIMPLIFIED auto-trading setup - only essential parameters"""
    total_supply: int = Field(..., gt=0, description="Total token supply")
    num_users: int = Field(..., gt=0, description="Number of users to create")
    initial_liquidity_percentage: float = Field(..., gt=0, le=100, description="% of supply for initial liquidity")
    max_slippage_percentage: float = Field(..., gt=0, le=100, description="Maximum slippage % for secondary market")

    # Trading probabilities
    buy_probability_percentage: float = Field(..., ge=0, le=100, description="% chance of buy vs sell")
    panic_sell_probability_percentage: float = Field(..., ge=0, le=100, description="% chance of selling 100%")
    initial_dump_probability_percentage: float = Field(..., ge=0, le=100, description="% of users who dump 100% at start")

    # Volume ranges - SIMPLE: same logic for buys and sells
    min_buy_tokens: int = Field(..., gt=0, description="Minimum tokens per buy")
    max_buy_tokens: int = Field(..., gt=0, description="Maximum tokens per buy")
    min_sell_tokens: int = Field(..., gt=0, description="Minimum tokens per sell")
    max_sell_tokens: int = Field(..., gt=0, description="Maximum tokens per sell")

    # Fees
    fee_percentage: float = Field(..., gt=0, le=100, description="Fee % on transactions (e.g., 1.0 for 1%)")

    # Trading interval
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


class InjectLiquidityRequest(BaseModel):
    """Inject liquidity into pool during simulation"""
    amount_eur: float = Field(..., gt=0, description="Amount in EUR to inject into liquidity pool")