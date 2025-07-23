import React, { useState, useEffect } from 'react';

interface Translation {
  id: string;
  originalText: string;
  brailleText: string;
  timestamp: Date;
  confidence: number;
}

interface BrailleTranslatorProps {
  detectedLetter?: { letter: string; confidence: number } | null;
}

const BrailleTranslator: React.FC<BrailleTranslatorProps> = ({ detectedLetter }) => {
  const [translations, setTranslations] = useState<Translation[]>([]);


  const convertToBraille = (text: string): string => {
    const brailleMap: { [key: string]: string } = {
      'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑',
      'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚',
      'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕',
      'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞',
      'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵',
      ' ': '⠀', 'á': '⠷', 'é': '⠮', 'í': '⠌', 'ó': '⠬', 'ú': '⠾', 'ñ': '⠻'
    };
    
    return text.toLowerCase().split('').map(char => brailleMap[char] || char).join('');
  };

  // Agregar letra detectada al historial
  useEffect(() => {
    if (detectedLetter && detectedLetter.confidence > 0.7) {
      const translation: Translation = {
        id: Date.now().toString(),
        originalText: detectedLetter.letter,
        brailleText: convertToBraille(detectedLetter.letter),
        timestamp: new Date(),
        confidence: detectedLetter.confidence * 100
      };
      setTranslations(prev => [translation, ...prev.slice(0, 9)]);
    }
  }, [detectedLetter]);

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

  return (
    <div className="bg-white/80 backdrop-blur-xl rounded-3xl p-6 shadow-2xl border border-white/20">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-800 flex items-center space-x-3">
            <span>Traductor</span>
          </h2>
          <p className="text-gray-600 mt-1">Conversión inteligente de braille a texto</p>
        </div>
        
        {translations.length > 0 && (
          <button 
            onClick={clearHistory}
            className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-4 py-2 rounded-xl transition-colors duration-200 flex items-center space-x-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            <span>Limpiar</span>
          </button>
        )}
      </div>

      {detectedLetter && (
        <div className="mb-6 animate-fade-in">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-100 rounded-2xl p-6 border border-blue-200">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800">Letra Detectada</h3>
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(detectedLetter.confidence * 100)}`}>
                {(detectedLetter.confidence * 100).toFixed(1)}% confianza
              </div>
            </div>
            
            <div className="text-center">
              <div className="text-8xl font-bold text-blue-600 mb-4">
                {detectedLetter.letter}
              </div>
              <p className="text-gray-600 mb-4">
                Letra braille detectada por el modelo entrenado
              </p>
              
              <div className="bg-white rounded-xl p-4 border border-blue-200">
                <p className="text-sm text-gray-600 mb-2">Representación en braille:</p>
                <p className="text-3xl font-mono text-blue-800">
                  {convertToBraille(detectedLetter.letter)}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {translations.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 flex items-center space-x-2">
            <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Historial reciente</span>
          </h3>
          
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {translations.map((translation, index) => (
              <div 
                key={translation.id} 
                className="bg-gray-50 hover:bg-gray-100 rounded-xl p-4 transition-all duration-200 border border-gray-200 hover:border-gray-300 animate-slide-up"
                style={{animationDelay: `${index * 100}ms`}}
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <p className="font-medium text-gray-800 mb-1">{translation.originalText}</p>
                    <p className="text-sm font-mono text-gray-600 mb-2">{translation.brailleText}</p>
                    <div className="flex items-center space-x-3 text-xs text-gray-500">
                      <span>{translation.timestamp.toLocaleTimeString()}</span>
                      <div className={`px-2 py-1 rounded-full ${getConfidenceColor(translation.confidence)}`}>
                        {translation.confidence}%
                      </div>
                    </div>
                  </div>
                  <button 
                    onClick={() => copyToClipboard(translation.originalText)}
                    className="ml-3 p-2 hover:bg-gray-200 rounded-lg transition-colors duration-200"
                    title="Copiar texto"
                  >
                    <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Estado vacío */}
      {translations.length === 0 && !detectedLetter && (
        <div className="text-center py-12 space-y-6">
          <div className="w-24 h-24 mx-auto bg-gradient-to-br from-gray-100 to-gray-200 rounded-full flex items-center justify-center">
            <svg className="w-12 h-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <div>
            <h3 className="text-xl font-semibold text-gray-800 mb-2">Listo para traducir</h3>
            <p className="text-gray-600 max-w-md mx-auto">
              Captura una imagen con texto en braille usando la cámara para comenzar la traducción automática
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default BrailleTranslator;