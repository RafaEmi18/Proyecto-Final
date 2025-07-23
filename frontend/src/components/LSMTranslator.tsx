import React, { useState, useEffect } from 'react';

interface LSMTranslation {
  id: string;
  originalSign: string;
  translatedText: string;
  timestamp: Date;
  confidence: number;
  type: 'word' | 'letter';
}

interface LSMTranslatorProps {
  detectedSign?: { text: string; confidence: number; type: 'word' | 'letter' } | null;
}

const LSMTranslator: React.FC<LSMTranslatorProps> = ({ detectedSign }) => {
  const [translations, setTranslations] = useState<LSMTranslation[]>([]);
  const [voiceEnabled, setVoiceEnabled] = useState(false);

  // Función para convertir texto a voz
  const speakText = (text: string) => {
    if (voiceEnabled && 'speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'es-ES';
      utterance.rate = 0.8;
      speechSynthesis.speak(utterance);
    }
  };

  // Agregar seña detectada al historial
  useEffect(() => {
    if (detectedSign && detectedSign.confidence > 0.7) {
      const translation: LSMTranslation = {
        id: Date.now().toString(),
        originalSign: detectedSign.text,
        translatedText: detectedSign.text,
        timestamp: new Date(),
        confidence: detectedSign.confidence * 100,
        type: detectedSign.type
      };
      
      setTranslations(prev => [translation, ...prev.slice(0, 9)]);
      
      // Reproducir voz si está habilitada
      if (voiceEnabled) {
        speakText(detectedSign.text);
      }
    }
  }, [detectedSign, voiceEnabled]);

  const clearHistory = () => {
    setTranslations([]);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'text-green-600 bg-green-100';
    if (confidence >= 80) return 'text-yellow-600 bg-yellow-100';
    return 'text-orange-600 bg-orange-100';
  };

  const getTypeIcon = (type: 'word' | 'letter') => {
    return type === 'word' ? '💬' : '🔤';
  };

  const getTypeColor = (type: 'word' | 'letter') => {
    return type === 'word' ? 'bg-blue-100 text-blue-800' : 'bg-purple-100 text-purple-800';
  };

  return (
    <div className="bg-gray-800 rounded-2xl p-6 shadow-xl text-white">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
            <span>Resultado</span>
          </h2>
          <p className="text-gray-300 mt-1">Traducción de lengua de señas a texto</p>
        </div>
        
        <div className="flex items-center space-x-3">
          {/* Control de voz */}
          <button
            onClick={() => setVoiceEnabled(!voiceEnabled)}
            className={`px-4 py-2 rounded-xl transition-colors duration-200 flex items-center space-x-2 ${
              voiceEnabled 
                ? 'bg-blue-600 hover:bg-blue-700 text-white' 
                : 'bg-gray-600 hover:bg-gray-700 text-gray-300'
            }`}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M9 12a3 3 0 106 0v-5a3 3 0 00-6 0v5z" />
            </svg>
            <span className="hidden sm:inline">{voiceEnabled ? 'Activar voz' : 'Desactivar voz'}</span>
          </button>
          
          {translations.length > 0 && (
            <button 
              onClick={clearHistory}
              className="bg-gray-600 hover:bg-gray-700 text-gray-300 px-4 py-2 rounded-xl transition-colors duration-200 flex items-center space-x-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
              <span className="hidden sm:inline">Limpiar</span>
            </button>
          )}
        </div>
      </div>

      {/* Resultado actual */}
      {detectedSign ? (
        <div className="mb-6 animate-fade-in">
          <div className="bg-gray-700 rounded-2xl p-6 border border-gray-600">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Seña Detectada</h3>
              <div className="flex items-center space-x-2">
                <div className={`px-3 py-1 rounded-full text-sm font-medium ${getTypeColor(detectedSign.type)}`}>
                  {getTypeIcon(detectedSign.type)} {detectedSign.type === 'word' ? 'Palabra' : 'Letra'}
                </div>
                <div className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(detectedSign.confidence * 100)}`}>
                  {(detectedSign.confidence * 100).toFixed(1)}%
                </div>
              </div>
            </div>
            
            <div className="text-center">
              <div className="text-6xl font-bold text-blue-400 mb-4">
                {detectedSign.text}
              </div>
              <p className="text-gray-300 mb-4">
                {detectedSign.type === 'word' ? 'Palabra' : 'Letra'} detectada por reconocimiento de gestos
              </p>
              
              <div className="bg-gray-600 rounded-xl p-4 border border-gray-500">
                <p className="text-sm text-gray-300 mb-2">Confianza: ---%</p>
                <div className="flex justify-center space-x-4">
                  <button
                    onClick={() => setVoiceEnabled(true)}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
                  >
                    Activar voz
                  </button>
                  <button
                    onClick={() => setVoiceEnabled(false)}
                    className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors"
                  >
                    Desactivar voz
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="mb-6">
          <div className="bg-gray-700 rounded-2xl p-6 border border-gray-600 text-center">
            <div className="text-4xl text-gray-500 mb-4">---</div>
            <p className="text-gray-400">Esperando detección de seña...</p>
          </div>
        </div>
      )}

      {/* Historial */}
      {translations.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
            <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Historial reciente</span>
          </h3>
          
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {translations.map((translation, index) => (
              <div 
                key={translation.id} 
                className="bg-gray-700 hover:bg-gray-600 rounded-xl p-4 transition-all duration-200 border border-gray-600 hover:border-gray-500 animate-slide-up"
                style={{animationDelay: `${index * 100}ms`}}
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <p className="font-medium text-white text-lg">{translation.translatedText}</p>
                      <div className={`px-2 py-1 rounded-full text-xs font-medium ${getTypeColor(translation.type)}`}>
                        {getTypeIcon(translation.type)}
                      </div>
                    </div>
                    <div className="flex items-center space-x-3 text-xs text-gray-400">
                      <span>{translation.timestamp.toLocaleTimeString()}</span>
                      <div className={`px-2 py-1 rounded-full ${getConfidenceColor(translation.confidence)}`}>
                        {translation.confidence.toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2 ml-3">
                    <button 
                      onClick={() => speakText(translation.translatedText)}
                      className="p-2 hover:bg-gray-500 rounded-lg transition-colors duration-200"
                      title="Reproducir voz"
                    >
                      <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M9 12a3 3 0 106 0v-5a3 3 0 00-6 0v5z" />
                      </svg>
                    </button>
                    <button 
                      onClick={() => copyToClipboard(translation.translatedText)}
                      className="p-2 hover:bg-gray-500 rounded-lg transition-colors duration-200"
                      title="Copiar texto"
                    >
                      <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Estado vacío */}
      {translations.length === 0 && !detectedSign && (
        <div className="text-center py-12 space-y-6">
          <div className="w-24 h-24 mx-auto bg-gradient-to-br from-gray-700 to-gray-800 rounded-full flex items-center justify-center">
            <svg className="w-12 h-12 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4V2a1 1 0 011-1h8a1 1 0 011 1v2m0 0V1a1 1 0 011-1h2a1 1 0 011 1v18a1 1 0 01-1 1H4a1 1 0 01-1-1V1a1 1 0 011-1h2a1 1 0 011 1v3z" />
            </svg>
          </div>
          <div>
            <h3 className="text-xl font-semibold text-white mb-2">Listo para traducir</h3>
            <p className="text-gray-400 max-w-md mx-auto">
              Realiza una seña frente a la cámara para comenzar la traducción automática de LSM a texto
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default LSMTranslator;