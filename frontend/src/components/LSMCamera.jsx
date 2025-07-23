import React, { useRef, useState, useEffect, useCallback } from 'react';

const LSMCamera = ({ onSignDetected }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [autoCapture, setAutoCapture] = useState(false);
  const [captureInterval, setCaptureInterval] = useState(null);

  // Datos simulados para LSM
  const LSM_SIGNS = {
    words: [
      { id: 'hola', text: 'Hola', confidence: 0.95 },
      { id: 'como_estas', text: 'Cómo estás', confidence: 0.92 },
      { id: 'bien', text: 'Bien', confidence: 0.88 },
      { id: 'mal', text: 'Mal', confidence: 0.85 },
      { id: 'gracias', text: 'Gracias', confidence: 0.90 }
    ],
    letters: [
      { id: 'A', text: 'A', confidence: 0.93 },
      { id: 'B', text: 'B', confidence: 0.91 },
      { id: 'C', text: 'C', confidence: 0.89 },
      { id: 'D', text: 'D', confidence: 0.87 },
      { id: 'E', text: 'E', confidence: 0.94 }
    ]
  };

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
          facingMode: 'user' // Cámara frontal para LSM
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

  // Simular detección de señas
  const simulateSignDetection = useCallback(() => {
    const allSigns = [...LSM_SIGNS.words, ...LSM_SIGNS.letters];
    const randomSign = allSigns[Math.floor(Math.random() * allSigns.length)];
    
    // Agregar algo de variación en la confianza
    const confidence = randomSign.confidence + (Math.random() - 0.5) * 0.1;
    
    return {
      text: randomSign.text,
      confidence: Math.max(0.7, Math.min(0.99, confidence)),
      type: LSM_SIGNS.words.includes(randomSign) ? 'word' : 'letter'
    };
  }, []);

  // Capturar imagen y simular predicción
  const captureAndPredict = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !isStreaming || isProcessing) return;

    setIsProcessing(true);
    
    try {
      // Simular tiempo de procesamiento
      await new Promise(resolve => setTimeout(resolve, 800));
      
      const result = simulateSignDetection();
      setPrediction(result);
      
      // Notificar al componente padre
      if (onSignDetected) {
        onSignDetected(result.text, result.confidence, result.type);
      }
      
    } catch (err) {
      console.error('Error en predicción:', err);
      setError('Error al procesar la imagen: ' + err.message);
    } finally {
      setIsProcessing(false);
    }
  }, [isStreaming, isProcessing, simulateSignDetection, onSignDetected]);

  // Iniciar captura automática
  const startAutoCapture = useCallback(() => {
    if (captureInterval) return;
    
    const interval = setInterval(() => {
      captureAndPredict();
    }, 3000); // Capturar cada 3 segundos para LSM
    
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
    <div className="bg-white rounded-2xl p-6 shadow-xl">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-black mb-2">Cámara en vivo</h2>
        <p className="text-black">Reconocimiento en tiempo real de señas mediante puntos clave de la mano.</p>
      </div>

      <div className="relative rounded-2xl overflow-hidden shadow-xl bg-gray-800">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-auto max-h-96 object-cover"
        />
        
        {/* Overlay de detección con puntos simulados */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Simulación de puntos de detección de mano */}
          {isStreaming && (
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
              {/* Puntos rojos simulando detección de mano */}
              <div className="relative w-32 h-40">
                {/* Dedos */}
                <div className="absolute top-2 left-6 w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                <div className="absolute top-4 left-10 w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                <div className="absolute top-6 left-14 w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                <div className="absolute top-8 left-18 w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                <div className="absolute top-10 left-20 w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                
                {/* Líneas verdes conectando puntos */}
                <svg className="absolute inset-0 w-full h-full" viewBox="0 0 128 160">
                  <path d="M24 8 L40 16 L56 24 L72 32 L80 40" stroke="#10B981" strokeWidth="2" fill="none" className="animate-pulse" />
                  <path d="M24 8 L32 48 L40 80 L48 120" stroke="#10B981" strokeWidth="2" fill="none" className="animate-pulse" />
                </svg>
              </div>
            </div>
          )}
          
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
            <div className="absolute inset-0 bg-blue-500/20 animate-pulse flex items-center justify-center">
              <div className="bg-black/70 text-white px-4 py-2 rounded-full text-sm">
                Analizando seña...
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
          {isProcessing ? 'Analizando...' : 'Detectar Seña'}
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
          {autoCapture ? 'Detener Auto' : 'Auto Detección'}
        </button>
      </div>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
};

export default LSMCamera;