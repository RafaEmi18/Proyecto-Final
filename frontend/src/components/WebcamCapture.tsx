import React, { useRef, useState, useEffect } from 'react';

interface WebcamCaptureProps {
  onCapture: (imageData: string) => void;
}

const WebcamCapture: React.FC<WebcamCaptureProps> = ({ onCapture }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    initializeCamera();
    return () => {
      stopCamera();
    };
  }, []);

  const initializeCamera = async () => {
    try {
      setError(null);
      setHasPermission(null);
      
      // Detener stream anterior si existe
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'environment'
        }
      });

      streamRef.current = stream;
      setHasPermission(true);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        
        // Manejar múltiples eventos para asegurar que funcione en todos los navegadores
        const handleVideoReady = () => {
          if (videoRef.current) {
            videoRef.current.play()
              .then(() => {
                setIsStreaming(true);
                console.log('Video playing successfully');
              })
              .catch((playError) => {
                console.error('Error playing video:', playError);
                setError('No se pudo reproducir el video de la cámara');
              });
          }
        };

        // Usar múltiples eventos para mayor compatibilidad
        videoRef.current.addEventListener('loadedmetadata', handleVideoReady);
        videoRef.current.addEventListener('canplay', handleVideoReady);
        
        // Timeout de seguridad
        setTimeout(() => {
          if (!isStreaming && videoRef.current && videoRef.current.readyState >= 2) {
            handleVideoReady();
          }
        }, 1000);
      }
    } catch (error: any) {
      console.error('Error accessing camera:', error);
      setHasPermission(false);
      
      // Mensajes de error más específicos
      if (error.name === 'NotAllowedError') {
        setError('Permiso de cámara denegado. Por favor, permite el acceso a la cámara.');
      } else if (error.name === 'NotFoundError') {
        setError('No se encontró ninguna cámara en tu dispositivo.');
      } else if (error.name === 'NotReadableError') {
        setError('La cámara está siendo usada por otra aplicación.');
      } else {
        setError('Error al acceder a la cámara: ' + error.message);
      }
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
  };

  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current || !isStreaming) return;

    setIsCapturing(true);
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (context) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0);
      
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      onCapture(imageData);
      
      setTimeout(() => setIsCapturing(false), 500);
    }
  };

  // Mostrar error específico
  if (error) {
    return (
      <div className="bg-gradient-to-br from-red-50 to-orange-50 rounded-3xl p-8 shadow-2xl border border-red-100">
        <div className="text-center space-y-6">
          <div className="w-20 h-20 mx-auto bg-gradient-to-br from-red-400 to-orange-500 rounded-full flex items-center justify-center shadow-lg">
            <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <div>
            <h3 className="text-2xl font-bold text-gray-800 mb-3">Error con la cámara</h3>
            <p className="text-gray-600 mb-6 max-w-md mx-auto">
              {error}
            </p>
          </div>
          <button 
            onClick={initializeCamera}
            className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold py-4 px-8 rounded-2xl shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 flex items-center space-x-3 mx-auto"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            <span>Intentar nuevamente</span>
          </button>
        </div>
      </div>
    );
  }

  if (hasPermission === null) {
    return (
      <div className="bg-white/80 backdrop-blur-xl rounded-3xl p-8 shadow-2xl border border-white/20">
        <div className="flex flex-col items-center justify-center space-y-6">
          <div className="relative">
            <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
            <div className="absolute inset-0 w-16 h-16 border-4 border-transparent border-t-purple-400 rounded-full animate-spin animation-delay-150"></div>
          </div>
          <div className="text-center">
            <h3 className="text-xl font-semibold text-gray-800 mb-2">Inicializando cámara</h3>
            <p className="text-gray-600">Preparando el sistema de captura...</p>
          </div>
        </div>
      </div>
    );
  }

  if (hasPermission === false) {
    return (
      <div className="bg-gradient-to-br from-red-50 to-orange-50 rounded-3xl p-8 shadow-2xl border border-red-100">
        <div className="text-center space-y-6">
          <div className="w-20 h-20 mx-auto bg-gradient-to-br from-red-400 to-orange-500 rounded-full flex items-center justify-center shadow-lg">
            <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </div>
          <div>
            <h3 className="text-2xl font-bold text-gray-800 mb-3">Acceso a cámara requerido</h3>
            <p className="text-gray-600 mb-6 max-w-md mx-auto">
              Para usar el traductor de braille, necesitamos acceso a tu cámara para capturar las imágenes del texto.
            </p>
          </div>
          <button 
            onClick={initializeCamera}
            className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold py-4 px-8 rounded-2xl shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 flex items-center space-x-3 mx-auto"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            <span>Permitir acceso a cámara</span>
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white/80 backdrop-blur-xl rounded-3xl p-6 shadow-2xl border border-white/20 overflow-hidden">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2 flex items-center space-x-3">
          <span>Captura de Imagen</span>
        </h2>
        <p className="text-gray-600">Posiciona el texto braille dentro del marco de captura</p>
      </div>

      <div className={`relative rounded-2xl overflow-hidden shadow-xl transition-all duration-300 ${isCapturing ? 'scale-98 shadow-glow' : ''}`}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-auto max-h-96 object-cover"
        />
        
        {/* Overlay de captura */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Marco de guía */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-4/5 h-3/5">
            <div className="w-full h-full border-2 border-dashed border-white/80 rounded-xl relative">
              {/* Esquinas decorativas */}
              <div className="absolute -top-1 -left-1 w-6 h-6 border-l-4 border-t-4 border-yellow-400 rounded-tl-lg"></div>
              <div className="absolute -top-1 -right-1 w-6 h-6 border-r-4 border-t-4 border-yellow-400 rounded-tr-lg"></div>
              <div className="absolute -bottom-1 -left-1 w-6 h-6 border-l-4 border-b-4 border-yellow-400 rounded-bl-lg"></div>
              <div className="absolute -bottom-1 -right-1 w-6 h-6 border-r-4 border-b-4 border-yellow-400 rounded-br-lg"></div>
            </div>
          </div>
          
          {/* Instrucciones */}
          <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2">
            <div className="bg-black/70 backdrop-blur-sm text-white px-4 py-2 rounded-full text-sm font-medium">
              Apunta hacia el texto en braille
            </div>
          </div>

          {/* Efecto de captura */}
          {isCapturing && (
            <div className="absolute inset-0 bg-white/30 animate-pulse"></div>
          )}
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
      </div>
      
      <div className="mt-6 text-center">
        <button
          onClick={captureImage}
          disabled={!isStreaming || isCapturing}
          className={`
            relative overflow-hidden group
            ${isCapturing 
              ? 'bg-gradient-to-r from-orange-500 to-red-500 shadow-glow' 
              : 'bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700'
            }
            text-white font-bold py-4 px-8 rounded-2xl shadow-xl
            transform transition-all duration-200
            ${!isCapturing && isStreaming ? 'hover:scale-105 hover:shadow-glow-green' : ''}
            disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none
            flex items-center space-x-3 mx-auto
          `}
          aria-label="Capturar imagen para traducir braille"
        >
          <div className="relative z-10 flex items-center space-x-3">
            <div className={`text-2xl transition-transform duration-200 ${isCapturing ? 'animate-bounce' : 'group-hover:scale-110'}`}>
              {isCapturing ? '' : ''}
            </div>
            <span className="text-lg">
              {isCapturing ? 'Capturando...' : isStreaming ? 'Capturar Imagen' : 'Preparando...'}
            </span>
          </div>
          
          {/* Efecto shimmer */}
          {!isCapturing && isStreaming && (
            <div className="absolute inset-0 -top-2 -bottom-2 bg-gradient-to-r from-transparent via-white/20 to-transparent transform -skew-x-12 group-hover:animate-shimmer"></div>
          )}
        </button>
      </div>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
};

export default WebcamCapture;