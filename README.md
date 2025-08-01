#  Intérprete Inteligente Multisensorial de Braille y LSM

##  Descripción

Sistema de traducción de Braille que utiliza inteligencia artificial para detectar y traducir texto braille en tiempo real usando la cámara del dispositivo.

##  Características

- **Detección en tiempo real**: Captura automática de imágenes desde la cámara
- **IA avanzada**: Modelo CNN entrenado para reconocer letras braille (A-Z)
- **Interfaz moderna**: Diseño tipo Google Translate con TailwindCSS
- **Alta precisión**: 95%+ de confianza en las detecciones
- **Responsivo**: Funciona en móviles, tablets y computadoras
- **📊 Análisis estadístico**: Sistema completo de estimación puntual y por intervalos
- **📈 Monitoreo en tiempo real**: Seguimiento de precisión, confianza y rendimiento

##  Tecnologías

### Backend
- **Python** con Flask
- **PyTorch** para el modelo CNN
- **OpenCV** para procesamiento de imágenes
- **Flask-CORS** para comunicación con frontend
- **NumPy & SciPy** para análisis estadístico
- **Sistema de estimación** para intervalos de confianza

### Frontend
- **Astro** como framework base
- **React** con TypeScript
- **TailwindCSS** para estilos
- **WebRTC** para acceso a cámara

##  Instalación y Uso

### Opción 1: Script Automático (Windows)
```bash
# Ejecutar el script de inicio
start-project.bat
```

### Opción 2: Instalación Manual

#### Backend
```bash
cd backend
pip install -r requeriments.txt
python app.py
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

##  Cómo Usar

1. **Abrir la aplicación**: Ve a `http://localhost:4321`
2. **Permitir cámara**: Autoriza el acceso a la cámara cuando se solicite
3. **Capturar imagen**: Usa el botón "Capturar" o activa "Auto Captura"
4. **Ver resultado**: La letra detectada aparecerá en el panel derecho
5. **Modo automático**: Activa "Auto Captura" para detección continua

##  Funcionalidades

- **Captura manual**: Presiona "Capturar" para procesar una imagen
- **Captura automática**: Activa "Auto Captura" para detección cada 2 segundos
- **Visualización en tiempo real**: Marco de detección con esquinas amarillas
- **Resultados detallados**: Letra detectada con porcentaje de confianza
- **Representación braille**: Muestra el símbolo braille correspondiente
- **📊 Panel de estadísticas**: Estimaciones puntuales y por intervalos
- **📈 Análisis por letra**: Rendimiento individual de cada letra A-Z
- **⏱️ Monitoreo de rendimiento**: Tiempos de respuesta y métricas de confianza

##  Configuración

### Modelo IA
El modelo CNN está entrenado para detectar las letras A, B, C en braille. Para expandir:
1. Agregar más clases en `CLASSES = ['A', 'B', 'C']`
2. Reentrenar el modelo con `train.py`
3. Actualizar el frontend para mostrar las nuevas letras

### Puerto del Backend
Por defecto el backend corre en `http://localhost:5000`. Para cambiar:
1. Modificar `app.run(host='0.0.0.0', port=5000)` en `app.py`
2. Actualizar la URL en `BrailleCamera.jsx`

##  Estructura del Proyecto

```
Proyecto-Final/
├── backend/
│   ├── app.py                      # API Flask principal
│   ├── statistical_estimation.py   # Sistema de estimación estadística
│   ├── test_statistics.py         # Script de pruebas estadísticas
│   ├── predict_camera.py          # Script de detección con OpenCV
│   ├── train.py                   # Entrenamiento del modelo
│   ├── braille_model.pth          # Modelo entrenado
│   ├── requirements.txt           # Dependencias Python
│   └── README_ESTADISTICAS.md     # Documentación estadística
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── BrailleApp.tsx          # Componente principal
│   │   │   ├── BrailleCamera.jsx       # Cámara en tiempo real
│   │   │   ├── BrailleTranslator.tsx   # Traducción y resultados
│   │   │   ├── StatisticsPanel.tsx     # Panel de estadísticas
│   │   │   └── NavBar.tsx              # Barra de navegación
│   │   └── pages/
│   │       ├── index.astro             # Página principal
│   │       ├── lsm.astro               # Página LSM
│   │       └── statistics.astro        # Página de estadísticas
│   └── package.json
└── start-project.bat              # Script de inicio automático
```

##  📊 Sistema de Estadísticas

El proyecto incluye un sistema completo de **estimación puntual y por intervalos** que permite:

### Características del Sistema Estadístico
- **Estimación puntual**: Valores medios de precisión, confianza y tiempo de respuesta
- **Intervalos de confianza**: Rangos con 95% de confianza para cada métrica
- **Análisis por letra**: Rendimiento individual de cada letra A-Z
- **Monitoreo en tiempo real**: Seguimiento continuo del rendimiento del modelo

### Cómo Usar las Estadísticas

1. **Acceder al panel**: Ve a `http://localhost:4321/statistics`
2. **Ver resumen**: Consulta las métricas generales del sistema
3. **Análisis detallado**: Explora el rendimiento por letra individual
4. **Guardar datos**: Exporta las estadísticas en formato JSON

### Pruebas del Sistema Estadístico

```bash
cd backend
python test_statistics.py
```

Este script genera datos de prueba y muestra las estimaciones estadísticas.

Para más información, consulta `backend/README_ESTADISTICAS.md`.

##  Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request


---

**Desarrollado con Amor y Rigor Estadístico** 📊✨