import React, { useRef, useState, useEffect, useCallback } from 'react';

const BrailleCamera = ({ onLetterDetected }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [autoCapture, setAutoCapture] = useState(false);
  const [captureInterval, setCaptureInterval] = useState(null);

  // Inicializar cámara
  const initializeCamera = useCallback(async () => {
    try {
      setError(null);
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'environment'
        }
      });

      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          setIsStreaming(true);
        };
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
      setError('Error al acceder a la cámara: ' + err.message);
    }
  }, []);

  // Detener cámara
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
  }, []);

  // Capturar imagen y enviar al backend
  const captureAndPredict = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !isStreaming || isProcessing) return;

    setIsProcessing(true);
    
    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');

      // Configurar canvas
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Dibujar video en canvas
      context.drawImage(video, 0, 0);
      
      // Convertir a base64
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      
      // Enviar al backend
      const response = await fetch(import.meta.env.DEV ? 'http://localhost:5000/predict' : 'https://tu-backend-render.onrender.com/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData.split(',')[1] }) // Remover data:image/jpeg;base64,
      });

      if (!response.ok) {
        throw new Error('Error en la predicción');
      }

      const result = await response.json();
      setPrediction(result);
      
      // Notificar al componente padre
      if (onLetterDetected) {
        onLetterDetected(result.letter, result.confidence);
      }
      
    } catch (err) {
      console.error('Error en predicción:', err);
      setError('Error al procesar la imagen: ' + err.message);
    } finally {
      setIsProcessing(false);
    }
  }, [isStreaming, isProcessing]);

  // Iniciar captura automática
  const startAutoCapture = useCallback(() => {
    if (captureInterval) return;
    
    const interval = setInterval(() => {
      captureAndPredict();
    }, 2000); // Capturar cada 2 segundos
    
    setCaptureInterval(interval);
    setAutoCapture(true);
  }, [captureAndPredict, captureInterval]);

  // Detener captura automática
  const stopAutoCapture = useCallback(() => {
    if (captureInterval) {
      clearInterval(captureInterval);
      setCaptureInterval(null);
    }
    setAutoCapture(false);
  }, [captureInterval]);

  // Efectos
  useEffect(() => {
    initializeCamera();
    return () => {
      stopCamera();
      stopAutoCapture();
    };
  }, [initializeCamera, stopCamera, stopAutoCapture]);

  // Limpiar intervalo al desmontar
  useEffect(() => {
    return () => {
      if (captureInterval) {
        clearInterval(captureInterval);
      }
    };
  }, [captureInterval]);

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-100';
    if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl p-6">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto bg-red-100 rounded-full flex items-center justify-center mb-4">
            <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-red-800 mb-2">Error</h3>
          <p className="text-red-600 mb-4">{error}</p>
          <button 
            onClick={initializeCamera}
            className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg"
          >
            Reintentar
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white/80 backdrop-blur-xl rounded-3xl p-6 shadow-2xl border border-white/20">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Cámara en Tiempo Real</h2>
        <p className="text-gray-600">Detecta letras braille automáticamente</p>
      </div>

      <div className="relative rounded-2xl overflow-hidden shadow-xl">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-auto max-h-96 object-cover"
        />
        
        {/* Overlay de detección */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Marco de detección */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-4/5 h-3/5">
            <div className="w-full h-full border-2 border-dashed border-white/80 rounded-xl relative">
              {/* Esquinas decorativas */}
              <div className="absolute -top-1 -left-1 w-6 h-6 border-l-4 border-t-4 border-yellow-400 rounded-tl-lg"></div>
              <div className="absolute -top-1 -right-1 w-6 h-6 border-r-4 border-t-4 border-yellow-400 rounded-tr-lg"></div>
              <div className="absolute -bottom-1 -left-1 w-6 h-6 border-l-4 border-b-4 border-yellow-400 rounded-bl-lg"></div>
              <div className="absolute -bottom-1 -right-1 w-6 h-6 border-r-4 border-b-4 border-yellow-400 rounded-br-lg"></div>
            </div>
          </div>
          
          {/* Indicador de estado */}
          <div className="absolute top-4 right-4">
            <div className="flex items-center space-x-2 bg-black/50 backdrop-blur-sm rounded-full px-3 py-1">
              <div className={`w-2 h-2 rounded-full ${isStreaming ? 'bg-green-400 animate-pulse' : 'bg-yellow-400'}`}></div>
              <span className="text-white text-xs font-medium">
                {isStreaming ? 'EN VIVO' : 'CONECTANDO...'}
              </span>
            </div>
          </div>

          {/* Efecto de procesamiento */}
          {isProcessing && (
            <div className="absolute inset-0 bg-white/30 animate-pulse flex items-center justify-center">
              <div className="bg-black/70 text-white px-4 py-2 rounded-full text-sm">
                Procesando...
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Controles */}
      <div className="mt-6 flex flex-wrap gap-3 justify-center">
        <button
          onClick={captureAndPredict}
          disabled={!isStreaming || isProcessing}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-6 py-3 rounded-xl font-semibold transition-colors"
        >
          {isProcessing ? 'Procesando...' : 'Capturar'}
        </button>
        
        <button
          onClick={autoCapture ? stopAutoCapture : startAutoCapture}
          disabled={!isStreaming}
          className={`px-6 py-3 rounded-xl font-semibold transition-colors ${
            autoCapture 
              ? 'bg-red-600 hover:bg-red-700 text-white' 
              : 'bg-green-600 hover:bg-green-700 text-white'
          }`}
        >
          {autoCapture ? 'Detener Auto' : 'Auto Captura'}
        </button>
      </div>

      {/* Resultado de predicción */}
      {prediction && (
        <div className="mt-6 bg-gradient-to-br from-green-50 to-emerald-100 rounded-2xl p-6 border border-green-200">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-800">Letra Detectada</h3>
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(prediction.confidence)}`}>
              {(prediction.confidence * 100).toFixed(1)}% confianza
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-6xl font-bold text-green-600 mb-2">
              {prediction.letter}
            </div>
            <p className="text-gray-600">
              Letra braille detectada con alta precisión
            </p>
          </div>
        </div>
      )}

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
};

export default BrailleCamera;
