# ML Dashboard — Árbol de Decisión & Neurona

Dashboard interactivo con dos modelos de Machine Learning implementados en JS puro, sin dependencias externas ni backend.

## Modelos incluidos

| Modelo | Algoritmo | Parámetros configurables |
|--------|-----------|--------------------------|
| Árbol de Decisión | CART con impureza Gini | Profundidad máxima (1–6) |
| Neurona | Perceptrón con activación Sigmoid + SGD | Épocas (10–1000), Learning Rate (0.01–1.0) |

## Características

- **100% frontend** — sin servidor, sin instalación
- **Datos elásticos** — agrega puntos manualmente o usa presets (AND, OR, XOR, Iris)
- **Visualización en vivo** — árbol dibujado nodo por nodo, diagrama de neurona con pesos, frontera de decisión
- **Curva de pérdida** — histórico de entrenamiento del perceptrón
- **Predicción en tiempo real** — ingresa valores y ambos modelos responden al instante

## Despliegue

### GitHub Pages
1. Sube la carpeta a un repositorio en GitHub
2. Ve a Settings → Pages → Source: `main branch / root`
3. Tu app estará en `https://tu-usuario.github.io/tu-repo/`

### Render (Static Site)
1. Conecta tu repo de GitHub en [render.com](https://render.com)
2. Nuevo servicio → **Static Site**
3. Build command: *(vacío)*  
   Publish directory: `.`
4. Deploy — Render genera una URL pública automáticamente

## Estructura de archivos

```
ml-dashboard/
├── index.html   # Estructura del dashboard
├── style.css    # Diseño y tokens de color
├── models.js    # DecisionTree y Perceptron (CART + Sigmoid)
└── app.js       # Lógica de UI, visualizaciones, predicción
```

## Uso rápido

1. Abre `index.html` en el navegador (doble clic o servidor local)
2. Agrega datos manualmente o carga un **preset**
3. Ajusta los hiperparámetros con los sliders
4. Haz clic en **Entrenar Modelo**
5. Usa el panel **Predecir** para clasificar nuevos puntos
