@echo off
echo Iniciando Proyecto Braille Translator...
echo.

echo Instalando dependencias del backend...
cd backend
pip install -r requeriments.txt
echo.

echo Iniciando servidor backend en http://localhost:5000...
start "Backend Flask" cmd /k "python app.py"
cd ..

echo.
echo Instalando dependencias del frontend...
cd frontend
npm install
echo.

echo Iniciando servidor frontend en http://localhost:4321...
start "Frontend Astro" cmd /k "npm run dev"
cd ..

echo.
echo Proyecto iniciado correctamente!
echo Backend: http://localhost:5000
echo Frontend: http://localhost:4321
echo.
echo Presiona cualquier tecla para cerrar esta ventana...
pause > nul 