"""
Tinvex AMM Simulator - FastAPI Application
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from engine import SimulationEngine
from trading_engine import TradingEngine, TradeConfig
from api_models import (
    InitRequest, CreateUsersRequest, LiquidityDistributionRequest,
    BuyPrimaryRequest, SellRequest, BuySecondaryRequest,
    MarketStateResponse, ApiResponse
)
import asyncio

# Initialize FastAPI app
app = FastAPI(
    title="Tinvex AMM Simulator",
    description="Simulation engine for Tinvex tokenomics with primary and secondary markets",
    version="1.0.0"
)

# CORS middleware (allow all origins for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Global simulation engine instance
engine = SimulationEngine()

# Global trading engine instance
trading_engine = None
trading_task = None


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)}
    )


# Web UI Route
@app.get("/")
async def home(request: Request):
    """Render main simulator UI"""
    return templates.TemplateResponse("simulator.html", {"request": request})


# API Routes

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Tinvex Simulator is running"}


@app.get("/api/state", response_model=MarketStateResponse)
async def get_state():
    """Get current market state"""
    return engine.current_info()


@app.post("/api/reset")
async def reset_simulation():
    """Reset simulation to initial state"""
    engine.reset()
    return {
        "success": True,
        "message": "Simulation reset successfully",
        "data": engine.current_info()
    }


@app.post("/api/init")
async def initialize(request: InitRequest):
    """
    Initialize simulation with liquidity provider
    
    Args:
        first_purchase_percentage: Percentage of total supply (0-100)
    """
    result = engine.initial_liquidity(request.first_purchase_percentage)
    return result


@app.post("/api/users/create")
async def create_users(request: CreateUsersRequest):
    """
    Create users for simulation
    
    Args:
        total_users: Number of users to create
    """
    result = engine.create_users(request.total_users)
    return result


@app.post("/api/liquidity/distribute")
async def distribute_liquidity(request: LiquidityDistributionRequest):
    """
    Distribute tokens from LP to users
    
    Args:
        to_distribute: Amount to distribute (None = all LP balance)
    """
    result = engine.liquidity_distribution(request.to_distribute)
    return result


@app.post("/api/primary/buy")
async def buy_primary(request: BuyPrimaryRequest):
    """
    Execute primary market purchases
    
    Args:
        users_list: Optional list of users (uses engine.users_list if None)
    """
    result = engine.purchase_primary(request.users_list)
    return result


@app.post("/api/sell")
async def sell_tokens(request: SellRequest):
    """
    Sell tokens at anchor price (no slippage)
    
    Args:
        user_id: User selling tokens
        tokens_to_sell: Amount to sell
    """
    result = engine.sell(request.user_id, request.tokens_to_sell)
    return result


@app.post("/api/secondary/buy")
async def buy_secondary(request: BuySecondaryRequest):
    """
    Buy tokens in secondary market (with slippage)
    
    Args:
        user_id: User buying tokens
        quantity_to_buy: Amount to buy
    """
    result = engine.purchase_secondary(request.user_id, request.quantity_to_buy)
    return result


@app.post("/api/orders/sell/create")
async def create_sell_orders():
    """Generate random sell orders from users with tokens"""
    orders = engine.create_sell_orders()
    return {
        "success": True,
        "message": f"Created {len(orders)} sell orders",
        "data": orders
    }


@app.post("/api/orders/sell/execute")
async def execute_sell_orders():
    """Create and execute sell orders"""
    orders = engine.create_sell_orders()
    if not orders:
        return {"success": False, "message": "No sell orders created"}
    
    result = engine.execute_sell_orders(orders)
    return result


@app.post("/api/orders/secondary/create")
async def create_secondary_buy_orders(num_orders: int = None):
    """
    Generate random secondary buy orders
    
    Args:
        num_orders: Number of orders (None = all users)
    """
    orders = engine.create_secondary_buy_orders(num_orders)
    return {
        "success": True,
        "message": f"Created {len(orders)} secondary buy orders",
        "data": orders
    }


@app.post("/api/orders/secondary/execute")
async def execute_secondary_buy_orders(num_orders: int = None):
    """Create and execute secondary buy orders"""
    orders = engine.create_secondary_buy_orders(num_orders)
    if not orders:
        return {"success": False, "message": "No secondary buy orders created"}
    
    result = engine.execute_secondary_buy_orders(orders)
    return result


@app.get("/api/transactions")
async def get_transactions(limit: int = 50):
    """
    Get transaction history
    
    Args:
        limit: Maximum number of transactions to return
    """
    transactions = engine.transaction_history[-limit:]
    return {
        "success": True,
        "message": f"Retrieved {len(transactions)} transactions",
        "data": transactions,
        "total": len(engine.transaction_history)
    }


@app.get("/api/users")
async def get_users():
    """Get all users and their balances"""
    return {
        "success": True,
        "message": f"Retrieved {len(engine.user_balance)} users",
        "data": engine.user_balance
    }


@app.post("/api/simulate/quick")
async def quick_simulation():
    """
    Run a quick full simulation:
    1. Initialize with 20% liquidity
    2. Create 3 users
    3. Distribute 10 tokens from LP
    4. Execute primary purchases
    5. Execute sell orders
    6. Execute secondary buy orders
    """
    steps = []
    
    # Step 1: Initialize
    result = engine.initial_liquidity(20)
    steps.append({"step": "initialize", "result": result})
    
    # Step 2: Create users
    result = engine.create_users(3)
    steps.append({"step": "create_users", "result": result})
    
    # Step 3: Distribute tokens
    result = engine.liquidity_distribution(10)
    steps.append({"step": "distribute_tokens", "result": result})
    
    # Step 4: Primary purchases
    result = engine.purchase_primary()
    steps.append({"step": "primary_purchases", "result": result})
    
    # Step 5: Sell orders
    sell_orders = engine.create_sell_orders()
    if sell_orders:
        result = engine.execute_sell_orders(sell_orders)
        steps.append({"step": "sell_orders", "result": result})
    
    # Step 6: Secondary buy orders
    buy_orders = engine.create_secondary_buy_orders()
    if buy_orders:
        result = engine.execute_secondary_buy_orders(buy_orders)
        steps.append({"step": "secondary_buy_orders", "result": result})
    
    return {
        "success": True,
        "message": "Quick simulation completed",
        "steps": steps,
        "final_state": engine.current_info()
    }


# ============================================
# AUTO-TRADING ENDPOINTS
# ============================================

@app.post("/api/trading/setup")
async def setup_auto_trading(
    total_supply: int = 100000000,
    num_users: int = 20,
    initial_liquidity_percentage: float = 5.0,
    max_slippage_percentage: float = 5.0,
    buy_probability_percentage: float = 55.0,
    panic_sell_probability_percentage: float = 5.0,
    transaction_interval_seconds: float = 0.5
):
    """
    Setup auto-trading simulation

    Args:
        total_supply: Total token supply
        num_users: Number of users to create
        initial_liquidity_percentage: % of supply for initial liquidity (at €1/token)
        max_slippage_percentage: Maximum slippage % for secondary market
        buy_probability_percentage: % chance of buy vs sell
        panic_sell_probability_percentage: % chance of selling 100% of holdings
        transaction_interval_seconds: Seconds between transactions (default 0.5)
    """
    global trading_engine

    # ✅ CRITICAL: Update parameters BEFORE reset
    engine.total_supply = total_supply
    engine.max_slippage = max_slippage_percentage / 100.0  # Convert % to decimal

    # Reset engine (will use the updated total_supply)
    engine.reset()

    # Step 1: Initialize with LP
    result = engine.initial_liquidity(initial_liquidity_percentage)
    if not result.get("success"):
        return {"success": False, "message": "Failed to initialize liquidity"}

    # Step 2: Create users
    result = engine.create_users(num_users)
    if not result.get("success"):
        return {"success": False, "message": "Failed to create users"}

    # Step 3: LP distributes 100% of tokens to users
    lp_balance = int(engine.user_balance[0]["tokens"])
    distribution_result = None
    if lp_balance > 0:
        distribution_result = engine.liquidity_distribution(lp_balance)
        if not distribution_result.get("success"):
            return {"success": False, "message": f"Failed to distribute tokens: {distribution_result.get('message')}"}

    # Step 4: Setup trading engine with custom probabilities
    config = TradeConfig(
        transaction_interval_seconds=transaction_interval_seconds,
        buy_probability=buy_probability_percentage / 100.0,  # Convert % to decimal
        panic_sell_probability=panic_sell_probability_percentage / 100.0  # Convert % to decimal
    )
    trading_engine = TradingEngine(engine, config)

    return {
        "success": True,
        "message": "Auto-trading setup completed",
        "config": {
            "total_supply": total_supply,
            "num_users": num_users,
            "initial_liquidity_percentage": initial_liquidity_percentage,
            "max_slippage_percentage": max_slippage_percentage,
            "buy_probability_percentage": buy_probability_percentage,
            "panic_sell_probability_percentage": panic_sell_probability_percentage,
            "transaction_interval_seconds": transaction_interval_seconds
        },
        "distribution": {
            "tokens_distributed": lp_balance if distribution_result else 0,
            "users_received": len(distribution_result.get("allocations", [])) if distribution_result else 0
        },
        "initial_state": engine.current_info()
    }


@app.post("/api/trading/start")
async def start_auto_trading():
    """Start continuous auto-trading simulation"""
    global trading_engine, trading_task
    
    if not trading_engine:
        return {
            "success": False,
            "message": "Trading engine not initialized. Call /api/trading/setup first"
        }
    
    if trading_engine.is_running:
        return {
            "success": False,
            "message": "Trading simulation already running"
        }
    
    # Start trading in background
    trading_task = asyncio.create_task(trading_engine.run_continuous())
    
    return {
        "success": True,
        "message": "Auto-trading simulation started",
        "config": {
            "transaction_interval_seconds": trading_engine.config.transaction_interval_seconds
        }
    }


@app.post("/api/trading/stop")
async def stop_auto_trading():
    """Stop continuous auto-trading simulation"""
    global trading_engine, trading_task
    
    if not trading_engine:
        return {"success": False, "message": "Trading engine not initialized"}
    
    if not trading_engine.is_running:
        return {"success": False, "message": "Trading simulation not running"}
    
    # Stop trading
    trading_engine.stop()
    
    # Cancel task if exists
    if trading_task and not trading_task.done():
        trading_task.cancel()
        try:
            await trading_task
        except asyncio.CancelledError:
            pass
    
    stats = trading_engine.get_stats()
    
    return {
        "success": True,
        "message": "Auto-trading simulation stopped",
        "stats": stats,
        "final_state": engine.current_info()
    }


@app.get("/api/trading/status")
async def get_trading_status():
    """Get current auto-trading status and statistics"""
    global trading_engine
    
    if not trading_engine:
        return {
            "success": False,
            "message": "Trading engine not initialized",
            "is_running": False
        }
    
    stats = trading_engine.get_stats()
    
    return {
        "success": True,
        "is_running": trading_engine.is_running,
        "trade_count": trading_engine.trade_count,
        "stats": stats,
        "current_state": engine.current_info()
    }


@app.get("/api/trading/price-history")
async def get_price_history(limit: int = 100):
    """
    Get price history for charting

    Args:
        limit: Maximum number of data points to return
    """
    global trading_engine

    if not trading_engine:
        return {
            "success": False,
            "message": "Trading engine not initialized",
            "data": []
        }

    history = trading_engine.price_history[-limit:] if trading_engine.price_history else []

    return {
        "success": True,
        "message": f"Retrieved {len(history)} price points",
        "data": history
    }


@app.get("/api/trading/top-traders")
async def get_top_traders(limit: int = 10):
    """
    Get top traders ranked by total value

    Args:
        limit: Maximum number of traders to return

    Returns:
        List of traders with investment and P&L metrics
    """
    traders = engine.get_top_traders(limit)

    return {
        "success": True,
        "message": f"Retrieved top {len(traders)} traders",
        "data": traders
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )