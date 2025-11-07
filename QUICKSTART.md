# ⚡ QUICK START - TINVEX SIMULATOR

## 🚀 Ejecutar en 3 pasos:

### 1️⃣ Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2️⃣ Ejecutar servidor
```bash
python3 main.py
```

### 3️⃣ Abrir en navegador
```
http://localhost:8000        → UI de prueba
http://localhost:8000/docs   → API interactiva (Swagger)
```

---

## 🧪 Test rápido

```bash
# Testear que el engine funciona
python test_engine.py
```

Deberías ver:
```
✅ ENGINE TEST COMPLETED SUCCESSFULLY!
```

---

## 🎯 Prueba desde el navegador

1. Abre `http://localhost:8000`
2. Click en "Run Quick Simulation"
3. Ve los resultados en JSON
4. Click en "Get Current State" para ver el estado final

---

## 📱 Prueba desde cURL

```bash
# Reset
curl -X POST http://localhost:8000/api/reset

# Quick simulation
curl -X POST http://localhost:8000/api/simulate/quick

# Ver estado
curl http://localhost:8000/api/state
```

---

## 🎨 SIGUIENTE: Frontend con Tailwind

Lee `PASO_1_COMPLETADO.md` para ver qué construiremos en el PASO 2.

---

## ❓ Troubleshooting

**Puerto ocupado?**
```bash
# Cambiar puerto en main.py línea final:
uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
```

**Dependencias?**
```bash
# Si pip install falla, prueba:
pip install fastapi uvicorn pydantic jinja2 --break-system-packages
```

**Python version?**
```bash
# Requiere Python 3.8+
python --version
```

---

## 📖 Documentación completa

- `README.md` - Guía completa
- `PASO_1_COMPLETADO.md` - Resumen de lo construido
- `http://localhost:8000/docs` - API docs interactiva (cuando el servidor esté corriendo)

---

¡Listo para simular! 🎉