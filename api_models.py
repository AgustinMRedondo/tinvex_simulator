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
    total_users: int = Field(..., ge=1, le=100, description="Number of users to create")


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