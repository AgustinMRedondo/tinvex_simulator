# 🚀 Tinvex AMM Simulator - Auto Trading Edition

Simulador continuo de trading con usuarios normales y rugpullers.

## ✨ Características

- **Trading continuo automático** - Simula mercado real 24/7
- **Rugpullers (5%)** - Usuarios que intentan vender todo su balance
- **Trading configurable** - Ajusta velocidad, tamaño de lotes, usuarios
- **Gráfica en tiempo real** - Visualiza precio mientras tradea
- **Dos fases**:
  - Fase 1: Compra primaria (hasta agotar)
  - Fase 2: Trading secundario + ventas continuas

## 🎮 Inicio Rápido

```bash
# 1. Instalar dependencias (si no lo has hecho)
pip3 install -r requirements.txt

# 2. Ejecutar servidor
python3 main.py

# 3. Abrir en navegador
http://localhost:8000
```

## 🎯 Cómo usar el simulador

### 1️⃣ Setup
En la UI, configura:
- **Total Supply**: Cantidad de tokens (ej: 100,000)
- **Number of Users**: Cuántos usuarios (ej: 20)
- **Initial Liquidity**: % del supply inicial (ej: 20%)
- **Transactions per Batch**: Transacciones por lote (ej: 10)
- **Batch Interval**: Segundos entre lotes (ej: 5)

Click "🚀 Setup Auto-Trading"

### 2️⃣ Start Trading
Click "▶️ START Trading"

El simulador comenzará a:
- Ejecutar 10 transacciones cada 5 segundos
- Mostrar precio en tiempo real
- Actualizar gráfica automáticamente
- Mostrar estadísticas del mercado

### 3️⃣ Stop Trading
Click "⏹️ STOP Trading"

Verás estadísticas finales:
- Total de trades ejecutados
- Cambio de precio (%)
- Estado final del mercado

## 📊 Lógica del Simulador

### Comportamiento de Usuarios

**Usuarios Normales (95%)**:
- 55% probabilidad de compra
- 45% probabilidad de venta
- Cantidad: 1-10,000 tokens aleatorio

**Rugpullers (5%)**:
- 30% probabilidad de compra
- 70% probabilidad de venta
- Cantidad al vender: **100% de su balance**

### Fases del Mercado

**Fase 1: Mercado Primario**
- Solo compras permitidas
- Precio fijo: P = Y/X (liquidez / circulación)
- Continúa hasta agotar `tokens_available_primary`

**Fase 2: Mercado Secundario**
- Compras desde secundario (con slippage 5%)
- Ventas al precio anchor (sin slippage)
- Las ventas liberan tokens al secundario

### Fórmulas

```
Primario:  P = Y / X

Venta:     P = Y / X  (precio anchor)

Secundario: P = [(Y*X) * (1 + s*(q/S))] / [X - q² * (s/S)]
donde:
  s = 0.05 (slippage 5%)
  q = cantidad a comprar
  S = tokens en secundario
```

## 🔌 API Endpoints

### Auto-Trading
```bash
# Setup
POST /api/trading/setup
{
  "total_supply": 100000,
  "num_users": 20,
  "initial_liquidity_percentage": 20,
  "transactions_per_batch": 10,
  "batch_interval_seconds": 5
}

# Start
POST /api/trading/start

# Stop
POST /api/trading/stop

# Status (polling para updates)
GET /api/trading/status

# Price history (para gráficas)
GET /api/trading/price-history?limit=100
```

### Estado del Mercado
```bash
GET /api/state              # Estado actual completo
GET /api/transactions       # Historial de transacciones
GET /api/users              # Lista de usuarios y balances
```

## 📈 Ejemplo de Uso con cURL

```bash
# 1. Setup
curl -X POST http://localhost:8000/api/trading/setup \
  -H "Content-Type: application/json" \
  -d '{
    "total_supply": 100000,
    "num_users": 20,
    "initial_liquidity_percentage": 20,
    "transactions_per_batch": 10,
    "batch_interval_seconds": 5
  }'

# 2. Start
curl -X POST http://localhost:8000/api/trading/start

# 3. Check status (en loop)
watch -n 2 'curl -s http://localhost:8000/api/trading/status | jq'

# 4. Stop
curl -X POST http://localhost:8000/api/trading/stop
```

## 🎨 Estructura del Proyecto

```
tinvex-simulator/
├── engine.py              # Core AMM engine (fórmulas)
├── trading_engine.py      # Auto-trading logic
├── main.py                # FastAPI app + endpoints
├── api_models.py          # Pydantic models
├── templates/
│   ├── simulator.html     # UI principal con auto-trading
│   └── index.html         # UI básica de prueba
└── requirements.txt
```

## 🧪 Testing

```bash
# Test básico del engine
python3 test_engine.py

# Test auto-trading (con curl)
./test_auto_trading.sh     # (si creamos script)
```

## 💡 Configuraciones Interesantes

### Mercado Lento y Estable
```json
{
  "total_supply": 10000,
  "num_users": 10,
  "transactions_per_batch": 5,
  "batch_interval_seconds": 10
}
```

### Mercado Rápido y Volátil
```json
{
  "total_supply": 100000,
  "num_users": 50,
  "transactions_per_batch": 20,
  "batch_interval_seconds": 2
}
```

### Test de Rugpull
```json
{
  "total_supply": 50000,
  "num_users": 20,
  "initial_liquidity_percentage": 30,
  "transactions_per_batch": 15,
  "batch_interval_seconds": 3
}
```

## 🔍 Observaciones Clave

**¿Qué pasa con rugpullers?**
- En fase primaria: Compran normal
- En fase secundaria: Venden TODO su balance (70% del tiempo)
- Liberan tokens masivos al secundario
- Puede causar caídas de precio... pero el sistema recupera con compras

**¿Por qué el precio se mantiene?**
- Las ventas son al precio anchor (Y/X)
- Mantiene proporción liquidez/circulación
- El slippage en compras secundarias ayuda a recuperar precio

**¿Qué observar en la gráfica?**
- Fase 1: Precio relativamente estable (compras primarias)
- Transición: Primera venta masiva de rugpuller
- Fase 2: Volatilidad por compras/ventas secundarias
- Tendencia: Debería mantenerse alrededor del precio anchor

## 📝 Notas Técnicas

- El simulador corre en **asyncio** (no bloquea el servidor)
- Los updates son por **polling** (la UI consulta cada 2 segundos)
- La gráfica guarda últimos 100 puntos de precio
- Todas las transacciones se guardan en memoria
- Cada usuario tiene tipo: "normal" o "rugpuller"

## 🐛 Troubleshooting

**"Trading engine not initialized"**
→ Ejecuta `/api/trading/setup` primero

**"Trading simulation already running"**
→ Detén con `/api/trading/stop` antes de reconfigurar

**La gráfica no se actualiza**
→ Verifica que el trading está activo (debe decir "Trading ACTIVE")

**Puerto ocupado**
→ Cambia el puerto en `main.py` (línea final)

## 🎯 Próximos Pasos

Para mejorar el simulador:
- [ ] WebSocket para updates en tiempo real (más fluido)
- [ ] Más tipos de usuarios (whales, bots, holders)
- [ ] Eventos especiales (airdrops, burns)
- [ ] Métricas avanzadas (volumen, volatilidad)
- [ ] Exportar datos a CSV/JSON
- [ ] Comparar múltiples simulaciones

---

¡Disfruta simulando! 🚀

Si algo no funciona, revisa:
1. Python 3.8+ instalado
2. Dependencias instaladas (`pip3 install -r requirements.txt`)
3. Puerto 8000 disponible
4. Navegador moderno (Chrome, Firefox, Safari)