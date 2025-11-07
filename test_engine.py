"""
Quick test script to verify the engine works correctly
"""

from engine import SimulationEngine

# Create engine
engine = SimulationEngine()

print("=" * 50)
print("TINVEX AMM SIMULATOR - ENGINE TEST")
print("=" * 50)

# Step 1: Initialize
print("\n1️⃣  Initializing with 20% liquidity...")
result = engine.initial_liquidity(20)
print(f"✅ {result['message']}")
print(f"   LP Balance: {result['lp_balance']} tokens")

# Step 2: Create users
print("\n2️⃣  Creating 3 users...")
result = engine.create_users(3)
print(f"✅ {result['message']}")
for user in result['users']:
    print(f"   {user['name']}: wants {user['desired_tokens']} tokens")

# Step 3: Distribute tokens from LP
print("\n3️⃣  Distributing 10 tokens from LP...")
result = engine.liquidity_distribution(10)
print(f"✅ {result['message']}")
for alloc in result['allocations']:
    print(f"   {alloc['user_name']}: +{alloc['tokens_received']} tokens")

# Step 4: Primary purchases
print("\n4️⃣  Executing primary market purchases...")
result = engine.purchase_primary()
print(f"✅ {result['message']}")
print(f"   Current price: €{result['current_price']:.4f}")
print(f"   Remaining primary: {result['tokens_available_primary']} tokens")

# Step 5: Check state
print("\n5️⃣  Current market state:")
state = engine.current_info()
print(f"   💰 Liquidity: €{state['current_liquidity']:.2f}")
print(f"   📈 Price: €{state['current_price']:.4f}")
print(f"   📦 Circulation: {state['tokens_in_circulation']} tokens")
print(f"   🏪 Primary available: {state['tokens_available_primary']} tokens")
print(f"   🔄 Secondary available: {state['tokens_available_secondary']} tokens")

# Step 6: Create and execute sell orders
print("\n6️⃣  Creating and executing sell orders...")
sell_orders = engine.create_sell_orders()
print(f"   Created {len(sell_orders)} sell orders")
result = engine.execute_sell_orders(sell_orders)
print(f"✅ {result['message']}")

# Step 7: Check state after sells
print("\n7️⃣  State after sells:")
state = engine.current_info()
print(f"   💰 Liquidity: €{state['current_liquidity']:.2f}")
print(f"   📈 Price: €{state['current_price']:.4f}")
print(f"   🔄 Secondary available: {state['tokens_available_secondary']} tokens")

# Step 8: Create and execute secondary buy orders
print("\n8️⃣  Creating and executing secondary buy orders...")
buy_orders = engine.create_secondary_buy_orders()
print(f"   Created {len(buy_orders)} buy orders")
result = engine.execute_secondary_buy_orders(buy_orders)
print(f"✅ {result['message']}")

# Step 9: Final state
print("\n9️⃣  Final market state:")
state = engine.current_info()
print(f"   💰 Liquidity: €{state['current_liquidity']:.2f}")
print(f"   📈 Price: €{state['current_price']:.4f}")
print(f"   📦 Circulation: {state['tokens_in_circulation']} tokens")
print(f"   🔄 Secondary available: {state['tokens_available_secondary']} tokens")
print(f"   📊 Total transactions: {state['transaction_count']}")

# Step 10: User balances
print("\n🔟 User balances:")
for uid, info in engine.user_balance.items():
    print(f"   {info['name']}: {info['tokens']:.2f} tokens")

print("\n" + "=" * 50)
print("✅ ENGINE TEST COMPLETED SUCCESSFULLY!")
print("=" * 50)