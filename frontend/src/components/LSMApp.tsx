import React, { useState } from 'react';
import NavBar from './NavBar';
import LSMCamera from './LSMCamera';
import LSMTranslator from './LSMTranslator';

const LSMApp: React.FC = () => {
  const [detectedSign, setDetectedSign] = useState<{ text: string; confidence: number; type: 'word' | 'letter' } | null>(null);

  const handleSignDetection = (text: string, confidence: number, type: 'word' | 'letter') => {
    setDetectedSign({ text, confidence, type });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-indigo-900">
      {/* NavBar */}
      <NavBar />

      {/* Header específico para LSM */}
      <div className="container mx-auto px-6 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-4">
            LSM Traductor - Universidad Politécnica de Tapachula
          </h1>
          <p className="text-gray-300 text-lg max-w-3xl mx-auto">
            Sistema de reconocimiento de Lengua de Señas Mexicana usando inteligencia artificial 
            para detectar gestos y convertirlos a texto en tiempo real
          </p>
        </div>

        {/* Área de traducción principal */}
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 min-h-[600px]">
            {/* Sección de cámara */}
            <div className="space-y-6">
              <LSMCamera onSignDetected={handleSignDetection} />
            </div>
            
            {/* Sección de traducción */}
            <div className="space-y-6">
              <LSMTranslator detectedSign={detectedSign} />
            </div>
          </div>
        </div>

        {/* Ejemplos de señas ilustradas */}
        <section className="mt-16 max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white mb-4">Ejemplos de señas ilustradas</h2>
            <p className="text-gray-300 max-w-2xl mx-auto">
              Algunas de las señas que nuestro sistema puede reconocer
            </p>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
            {[
              { name: 'Hola', emoji: '👋', description: 'Saludo básico' },
              { name: 'Gracias', emoji: '🙏', description: 'Expresión de gratitud' },
              { name: 'Bien', emoji: '👍', description: 'Estado positivo' },
              { name: 'Mal', emoji: '👎', description: 'Estado negativo' },
              { name: 'Cómo estás', emoji: '🤔', description: 'Pregunta sobre estado' }
            ].map((sign, index) => (
              <div 
                key={index}
                className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 text-center hover:transform hover:scale-105 transition-all duration-300 border border-white/20"
              >
                <div className="text-6xl mb-4">{sign.emoji}</div>
                <h3 className="text-lg font-bold text-white mb-2">{sign.name}</h3>
                <p className="text-gray-300 text-sm">{sign.description}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Sección de letras del abecedario */}
        <section className="mt-16 max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white mb-4">Letras del Abecedario LSM</h2>
            <p className="text-gray-300 max-w-2xl mx-auto">
              Letras básicas del abecedario en Lengua de Señas Mexicana
            </p>
          </div>
          
          <div className="grid grid-cols-5 gap-4">
            {['A', 'B', 'C', 'D', 'E'].map((letter, index) => (
              <div 
                key={index}
                className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 text-center hover:transform hover:scale-105 transition-all duration-300 border border-white/20"
              >
                <div className="text-4xl font-bold text-blue-400 mb-2">{letter}</div>
                <p className="text-gray-300 text-sm">Letra {letter}</p>
              </div>
            ))}
          </div>
        </section>
      </div>

      {/* Footer */}
      <footer className="mt-20 bg-black/30 backdrop-blur-sm text-white border-t border-white/10">
        <div className="container mx-auto px-6 py-8">
          <div className="text-center space-y-4">
            <div className="flex justify-center">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                <span className="text-2xl">🤟</span>
              </div>
            </div>
            
            <div>
              <h3 className="text-xl font-bold mb-2">Tecnología Inclusiva LSM</h3>
              <p className="text-gray-300 max-w-2xl mx-auto text-sm">
                Desarrollado para hacer la comunicación más accesible mediante el reconocimiento 
                inteligente de Lengua de Señas Mexicana.
              </p>
            </div>
            
            <div className="border-t border-gray-700 pt-4">
              <p className="text-gray-400 text-xs">
                Universidad Politécnica de Tapachula &copy; 2025
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LSMApp;