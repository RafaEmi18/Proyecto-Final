# 🦯 Traductor de Braille en Tiempo Real

## 📋 Descripción

Sistema de traducción de Braille que utiliza inteligencia artificial para detectar y traducir texto braille en tiempo real usando la cámara del dispositivo.

## ✨ Características

- **Detección en tiempo real**: Captura automática de imágenes desde la cámara
- **IA avanzada**: Modelo CNN entrenado para reconocer letras braille (A, B, C)
- **Interfaz moderna**: Diseño tipo Google Translate con TailwindCSS
- **Alta precisión**: 95%+ de confianza en las detecciones
- **Responsivo**: Funciona en móviles, tablets y computadoras

## 🛠️ Tecnologías

### Backend
- **Python** con Flask
- **PyTorch** para el modelo CNN
- **OpenCV** para procesamiento de imágenes
- **Flask-CORS** para comunicación con frontend

### Frontend
- **Astro** como framework base
- **React** con TypeScript
- **TailwindCSS** para estilos
- **WebRTC** para acceso a cámara

## 🚀 Instalación y Uso

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

## 📱 Cómo Usar

1. **Abrir la aplicación**: Ve a `http://localhost:4321`
2. **Permitir cámara**: Autoriza el acceso a la cámara cuando se solicite
3. **Capturar imagen**: Usa el botón "Capturar" o activa "Auto Captura"
4. **Ver resultado**: La letra detectada aparecerá en el panel derecho
5. **Modo automático**: Activa "Auto Captura" para detección continua

## 🎯 Funcionalidades

- **Captura manual**: Presiona "Capturar" para procesar una imagen
- **Captura automática**: Activa "Auto Captura" para detección cada 2 segundos
- **Visualización en tiempo real**: Marco de detección con esquinas amarillas
- **Resultados detallados**: Letra detectada con porcentaje de confianza
- **Representación braille**: Muestra el símbolo braille correspondiente

## 🔧 Configuración

### Modelo IA
El modelo CNN está entrenado para detectar las letras A, B, C en braille. Para expandir:
1. Agregar más clases en `CLASSES = ['A', 'B', 'C']`
2. Reentrenar el modelo con `train.py`
3. Actualizar el frontend para mostrar las nuevas letras

### Puerto del Backend
Por defecto el backend corre en `http://localhost:5000`. Para cambiar:
1. Modificar `app.run(host='0.0.0.0', port=5000)` en `app.py`
2. Actualizar la URL en `BrailleCamera.jsx`

## 📊 Estructura del Proyecto

```
Proyecto-Final/
├── backend/
│   ├── app.py              # API Flask principal
│   ├── predict_camera.py   # Script de detección con OpenCV
│   ├── train.py           # Entrenamiento del modelo
│   ├── braille_model.pth  # Modelo entrenado
│   └── requeriments.txt   # Dependencias Python
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── BrailleApp.tsx      # Componente principal
│   │   │   ├── BrailleCamera.jsx   # Cámara en tiempo real
│   │   │   └── BrailleTranslator.tsx # Traducción y resultados
│   │   └── pages/
│   │       └── index.astro         # Página principal
│   └── package.json
└── start-project.bat      # Script de inicio automático
```

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es parte del Proyecto Final de Integrador.

---

**Desarrollado con ❤️ para hacer la tecnología más accesible**