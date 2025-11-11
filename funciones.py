#LAS FUNCIONES ORIGINALES EXPLICADAS QUE TIENEN QUE SEGUIRSE SIN FALLO.
# NOTA: SIN HARDCODING - Todos los valores deben pasarse como parámetros

# Variables globales de estado (se inicializan en runtime, NO hardcodeadas)
initial_price = None  # EL PRECIO DE SALIDA SIEMPRE ES IGUAL QUE 1
total_supply = None  # TAMAÑO TOTAL DE LA EMISION
transaction_history = []  # LISTA HISTORIAL DE TODOS LOS MOVIMIENTOS
user_balance = {}  # DICCIONARIO DE ESTADO DEL BALANCE DE CADA USUARIO
min_tokens = None  # TOKENS MINIMOS PARA RANDOMIZAR
max_tokens = None  # TOKENS MAXIMOS PARA RANDOMIZAR
max_slippage = None  # EL SLIPPAGE MAXIMO PARA PRECIO
first_purchase_liquidity = None  # %DEL TOTAL DE TOKENS QUE COMPRA EN LA PRIMERA LIQUIDEZ
tokens_in_circulation = 0  # TOKENS QUE HAY EN CIRCULACION
tokens_available_primary = None  # TOKENS QUE HAY EN EL MERCADO PRIMARIO (= total_supply al inicio)
tokens_available_secondary = 0  # TOKENS QUE HAN SIDO COMPRADOS Y VENDIDOS Y ESTÁN DISPONIBLES EN EL SECUNDARIO
current_liquidity = 0  # LIQUIDEZ ACTUALIZADA EN CADA TRANSACCIÓN
current_price = None  # PRECIO ACTUALIZADO EN CADA TRANSACCIÓN
total_minted = 0  # TOTAL MINTEADOS
users_list = []  # LISTA DE USUARIOS


#PASO 1. COMIENZA CON LA LIQUIDEZ INICIAL. ESTA ES UNA COMPRA INICIAL QUE HACE EL EMISOR, PONE EL CAPITAL Y RECIBE EL LOS TOKENS INICIALES, ESTOS TOKENS SE LOS DA A LOS U
def initial_liquity (first_purchase_liquidity):
    global tokens_in_circulation, tokens_available_primary, transaction_history, current_liquidity, user_balance

    price = initial_price
    tokens_to_purchase = first_purchase_liquidity
    amount_eur = tokens_to_purchase * price

    tokens_in_circulation += tokens_to_purchase
    tokens_available_primary -= tokens_to_purchase 

    user_id = 0
    user_name = "Liquidity_provider"
    if user_id not in user_balance:
       user_balance[user_id] = {"name": user_name, "tokens": 0}
    user_balance[user_id]["tokens"] += tokens_to_purchase

    # Registrar transacción
    transaction = {
        'user_id': user_id,
        'user_name': user_name,
        'action': 'liquidity',
        'tokens_bought': tokens_to_purchase,
        'amount_eur': amount_eur,
        'price': price
    }
    transaction_history.append(transaction)

    # Actualizar liquidez
    current_liquidity += amount_eur

#PASO 2. FUNCION PARA CREAR LOS USUARIOS CON SU ID Y CALCULAMOS ALEATORIAMENTE CUANTOS TOKENS QUIEREN COMPRAR AL PRINCIPIO
def create_users(total_users: int, min_tokens: int, max_tokens: int):
    """
    Crea usuarios (id 1..N) en:
      - users_list: con 'desired_tokens' para compras primarias
      - user_balance: con balance 0 tokens (no toca al LP user 0)
    """
    global users_list, user_balance

    users_list = []
    for i in range(1, total_users + 1):
        name = f"User_{i}"
        desired = random.randint(min_tokens, max_tokens)

        # Lista para compras primarias
        users_list.append({"id": i, "name": name, "desired_tokens": desired})

        # Registro en balances con 0 tokens
        user_balance[i] = {"name": name, "tokens": 0.0}


#PASO 3. LOS TOKENS QUE TIENE EL QUE PONE EL LIQUIDITY POOL QUE ES EL USER:0 LO REPARTE ALEATORIAMENTE ENTRE LOS DEMAS USUARIOS
def liquidity_distribution(to_distribute: int | None = None):
    """
    Reparte ENTEROS de tokens del LP (user 0) entre los demás usuarios.
    - Si `to_distribute` es None, pregunta por input.
    - Solo enteros (no fracciones). La suma de asignaciones = to_distribute.
    - Algunos usuarios pueden recibir 0.
    - No modifica X, Y ni el secundario.
    """
    global user_balance, transaction_history

    # Validaciones del LP
    if 0 not in user_balance or user_balance[0].get("tokens", 0) <= 0:
        print("❌ No hay tokens en el Liquidity Provider (user 0).")
        return

    lp_tokens = int(user_balance[0]["tokens"])  # fuerza a entero distribuible
    if lp_tokens <= 0:
        print("❌ El LP no tiene tokens enteros disponibles para repartir.")
        return

    # Obtener cantidad a repartir
    if to_distribute is None:
        to_distribute = int(input(
            f"💰 Hay {lp_tokens} tokens en el Liquidity Provider. "
            f"¿Cuántos tokens ENTEROS quieres repartir? "
        ))
    else:
        to_distribute = int(to_distribute)

    if to_distribute <= 0:
        print("❌ La cantidad a repartir debe ser > 0.")
        return

    if to_distribute > lp_tokens:
        print(f"⚠️ Solo se pueden repartir {lp_tokens} tokens (todo el balance entero del LP).")
        to_distribute = lp_tokens

    # Usuarios elegibles (todos menos LP)
    eligible_users = [uid for uid in user_balance.keys() if uid != 0]
    if not eligible_users:
        print("⚠️ No hay usuarios receptores (distintos de 0).")
        return

    # Si hay más a repartir que usuarios, perfecto; si no, también (algunos 0)
    n = len(eligible_users)

    # Pesos aleatorios y proporciones
    weights = [random.random() for _ in eligible_users]
    total_w = sum(weights)
    if total_w == 0:
        # Reparto uniforme si todos los pesos son 0 (muy improbable)
        proportions = [1 / n] * n
    else:
        proportions = [w / total_w for w in weights]

    # Asignación entera base (redondeo hacia abajo)
    allocations = [int(to_distribute * p) for p in proportions]
    assigned = sum(allocations)

    # Repartir el remanente de 1 en 1 a usuarios aleatorios
    remainder = to_distribute - assigned
    if remainder > 0:
        picks = random.sample(eligible_users, k=min(remainder, n))
        # Creamos mapa uid->idx para sumar fácilmente
        idx_map = {uid: i for i, uid in enumerate(eligible_users)}
        for uid in picks:
            allocations[idx_map[uid]] += 1
        # Si todavía sobrara (to_distribute > n), repartir en ciclos
        remainder_left = to_distribute - sum(allocations)
        i = 0
        while remainder_left > 0:
            allocations[i % n] += 1
            remainder_left -= 1
            i += 1

    # Aplicar asignaciones a balances (solo enteros)
    allocations_out = []
    for uid, alloc in zip(eligible_users, allocations):
        if alloc <= 0:
            continue
        # Asegurar entrada del user
        if uid not in user_balance:
            user_balance[uid] = {"name": f"User_{uid}", "tokens": 0}
        # Sumar ENTERO
        user_balance[uid]["tokens"] = int(user_balance[uid].get("tokens", 0)) + int(alloc)
        allocations_out.append({
            "user_id": uid,
            "user_name": user_balance[uid]["name"],
            "tokens_received": int(alloc)
        })

    # Restar del LP exactamente lo repartido
    actually_distributed = sum(a["tokens_received"] for a in allocations_out)
    user_balance[0]["tokens"] = int(user_balance[0]["tokens"]) - actually_distributed

    # Log
    transaction_history.append({
        "action": "liquidity_distribution",
        "from_user": 0,
        "from_user_name": user_balance[0]["name"],
        "total_distributed": int(actually_distributed),
        "allocations": allocations_out
    })

# PASO 4. COMPRAS EN EL MERCADO PRIMERIO. ESTA MARCA EL INICIO DE LAS COMPRAS. SOLO SE EJECUTA SI QUEDAN TOKENS EN EL MERCADO PRIMARIO, ES DECIR, SI HAY TOKENS DISPONIBLES PARA COMPRAR SI NO COMPRAN USANDO LA DEL SECUNDARIO
def purchase_primary(users_list):
    global tokens_in_circulation, tokens_available_primary, transaction_history
    global current_liquidity, user_balance, current_price

    # Asegurar que existe el diccionario de balances
    if 'user_balance' not in globals() or not isinstance(user_balance, dict):
        user_balance = {}

    # Evitar división por cero si aún no hay tokens en circulación
    if tokens_in_circulation == 0:
        print("❌ No hay tokens en circulación aún. Ejecuta primero initial_liquidity().")
        return

    if tokens_available_primary <= 0:
        print("❌ Mercado primario cerrado: no quedan tokens disponibles.")
        return

    # Precio primario estricto inicial: P = Y / X
    current_price = current_liquidity / tokens_in_circulation

    for user in users_list:
        user_id = user["id"]
        desired_tokens = user["desired_tokens"]
        amount_eur = desired_tokens * current_price

        if tokens_available_primary <= 0:
            print("❌ No quedan tokens disponibles en el mercado primario.")
            break

        # Recortar a disponibilidad
        if desired_tokens > tokens_available_primary:
            desired_tokens = tokens_available_primary
            amount_eur = desired_tokens * current_price

        if desired_tokens <= 0:
            continue

        # Actualizar estado global
        tokens_in_circulation += desired_tokens
        tokens_available_primary -= desired_tokens
        current_liquidity += amount_eur

        # ✅ Actualizar precio global después de cada compra
        current_price = current_liquidity / tokens_in_circulation

        # Actualizar balance del usuario
        if user_id not in user_balance:
            user_balance[user_id] = {"name": user["name"], "tokens": 0}
        user_balance[user_id]["tokens"] += desired_tokens

        # Registrar transacción
        transaction = {
            "user_id": user_id,
            "user_name": user["name"],
            "action": "purchase_primary",
            "tokens_bought": desired_tokens,
            "amount_eur": amount_eur,
            "price": current_price
        }
        transaction_history.append(transaction)

#PASO 5. FUNCIÓN PARA VENTA. LAS VENTAS SON CONTRA EL LIQUIDITY POOL Y ESTOS TOKENS VENDIDOS VAN AL SECUNDARIO. TANTO LAS COMPRAS DEL PRIMARIO COMO DEL SECUNDARIO USAN ESTA FUNCIÓN. EL PRECIO DE VENTA ES EL QUE SE CONVIERTE EN EL NUEVO PRECIO DE VENTA

def sell (user_id: int, tokens_to_sell: float):
    """
    Venta simple al contrato (contra la liquidez):
      - Disminuye tokens en circulación (X).
      - Disminuye el liquidity pool (Y) por el payout.
      - Aumenta tokens disponibles en secundario (S).
      - NO modifica current_price (el precio se gestiona aparte).
    Requiere globales:
      - current_price, current_liquidity, tokens_in_circulation, tokens_available_secondary
      - user_balance (dict {user_id: {"name": str, "tokens": float}})
      - transaction_history (list)
    """
    global current_price, current_liquidity
    global tokens_in_circulation, tokens_available_secondary
    global user_balance, transaction_history

    # Validaciones
    if tokens_to_sell <= 0:
        print("❌ tokens_to_sell debe ser > 0.")
        return
    if user_id not in user_balance or user_balance[user_id].get("tokens", 0) <= 0:
        print(f"❌ El usuario {user_id} no tiene tokens para vender.")
        return
    if tokens_in_circulation <= 0:
        print("❌ No hay tokens en circulación para vender.")
        return

    # Precio de ejecución: usamos el current_price vigente.
    # Si por alguna razón fuese 0/negativo, intentamos anclar a Y/X; si tampoco es viable, abortamos.
    exec_price = current_liquidity/tokens_in_circulation

    # Cantidad a vender (cap a balance del usuario y al circulante)
    user_tokens = float(user_balance[user_id]["tokens"])
    q = min(float(tokens_to_sell), user_tokens, float(tokens_in_circulation))
    if q <= 0:
        print("⚠️ Nada que vender tras límites.")
        return

    # Payout y cap por liquidez
    payout_eur = q * exec_price
    if payout_eur > float(current_liquidity):
        # recortar a lo máximo pagable con la liquidez disponible
        q = float(current_liquidity) / exec_price
        payout_eur = q * exec_price

    if q <= 0 or payout_eur <= 0:
        print("❌ No fue posible ejecutar la venta con la liquidez disponible.")
        return

    # Actualizaciones de estado
    user_balance[user_id]["tokens"] -= q
    tokens_in_circulation -= q
    tokens_available_secondary += q
    current_liquidity -= payout_eur
    # Nota: current_price NO se toca aquí.

    # Registro de transacción
    transaction = {
        "user_id": user_id,
        "user_name": user_balance[user_id].get("name", f"User_{user_id}"),
        "action": "sell_simple",
        "tokens_sold": q,
        "payout_eur": payout_eur,
        "exec_price": exec_price,
    }
    transaction_history.append(transaction)

# PASO 6. FUNCIÓN PARA COMPRAS EN EL MERCADO SECUNDARIO. ESTA FUNCIÓN USA LA FÓRMULA DADA PARA CALCULAR EL PRECIO DE EJECUCIÓN Y LUEGO ACTUALIZA TODOS LOS PARÁMETROS CORRESPONDIENTES. SOLO CUANDO YA NO QUEDA SUPPLY EN EL PRIMARIO

def purchase_secondary(user_id: int, quantity_to_buy: float):
    """
    Compra en mercado secundario usando la fórmula dada:
      current_price = [(Y*X)*(1 + s*(q/S))] / [X - q^2 * (s/S)]
    y luego:
      - X += q
      - S -= q
      - Y += q * current_price
      - current_price := precio de ejecución calculado
      - user_balance[user_id]['tokens'] += q
    """
    global current_liquidity, tokens_in_circulation, tokens_available_secondary
    global current_price, user_balance, transaction_history, max_slippage

    # Parámetros y validaciones básicas
    if quantity_to_buy <= 0:
        print("❌ quantity_to_buy debe ser > 0.")
        return
    if tokens_available_secondary <= 0:
        print("❌ No hay oferta en secundario.")
        return
    if tokens_in_circulation <= 0 or current_liquidity < 0:
        print("❌ Estado inválido: asegúrate de tener X>0 y Y≥0 (liquidez inicial y compras primarias).")
        return

    # Asegurar usuario en balances
    if user_id not in user_balance:
        user_balance[user_id] = {"name": f"User_{user_id}", "tokens": 0.0}

    Y = current_price*tokens_in_circulation
    X = float(tokens_in_circulation)
    S = float(tokens_available_secondary)
    s = float(max_slippage)
    q_req = float(quantity_to_buy)

    # Cap a la oferta del secundario
    q = min(q_req, S)

    # Evitar denominador ≤ 0: den = X - q^2 * (s/S) > 0
    # Si no, recortamos q al máximo permitido por estabilidad
    if s > 0:
        # q_max_den ~ sqrt(X * S / s)
        from math import sqrt
        q_max_den = sqrt((X * S) / s) if X > 0 else 0.0
        if q >= q_max_den:
            q = max(0.0, min(q, q_max_den - 1e-9))  # pequeño margen
    # Si s == 0, la fórmula se reduce a current_price = (Y*X)/X = Y (no tiene sentido económico);
    # asumimos que s>0 en secundario. Si es 0, forzamos s>0 mínimamente.
    if s <= 0:
        s = 1e-9

    if q <= 0:
        print("⚠️ La cantidad ejecutable es 0 tras caps (oferta/estabilidad).")
        return

    # Precio de ejecución según tu fórmula
    numerator = (Y) * (1.0 + s * (q / S))
    denominator = X - (q ** 2) * (s / S)
    if denominator <= 0:
        print("❌ Denominador no positivo: reduce cantidad o revisa estado.")
        return

    exec_price = numerator / denominator
    if exec_price <= 0:
        print("❌ Precio calculado no positivo. Revisa estado/params.")
        return

    # Coste y actualizaciones
    cost_eur = q * exec_price

    tokens_available_secondary -= q
    tokens_in_circulation += q
    current_liquidity += cost_eur
    current_price = exec_price
    user_balance[user_id]["tokens"] += q

    # Log
    transaction_history.append({
        "user_id": user_id,
        "user_name": user_balance[user_id]["name"],
        "action": "buy_from_secondary",
        "tokens_bought": q,
        "amount_eur": cost_eur,
        "exec_price": exec_price,
        "Y_before": Y,
        "X_before": X,
        "S_before": S,
        "slippage": s
    })
      