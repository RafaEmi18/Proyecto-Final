import React, { useState } from 'react';
import NavBar from './NavBar';
import WebcamCapture from './WebcamCapture';
import BrailleTranslator from './BrailleTranslator';

const BrailleApp: React.FC = () => {
  const [capturedImage, setCapturedImage] = useState<string | null>(null);

  const handleCapture = (imageData: string) => {
    setCapturedImage(imageData);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* NavBar */}
      <NavBar />

      {/* Contenido principal - Estilo Google Translate */}
      <main className="container mx-auto px-6 py-8">
        <div className="max-w-6xl mx-auto">
          {/* Área de traducción principal */}
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 overflow-hidden">
            <div className="grid grid-cols-1 lg:grid-cols-2 min-h-[500px]">
              {/* Sección de cámara */}
              <div className="p-8 border-r border-gray-200/50">
                <div className="h-full flex flex-col">
                  <div className="mb-6">
                    <h2 className="text-xl font-semibold text-gray-800 mb-2">Camara Web</h2>
                    <p className="text-gray-600 text-sm">
                      Usa tu cámara para capturar texto en braille
                    </p>
                  </div>
                  <div className="flex-1">
                    <WebcamCapture onCapture={handleCapture} />
                  </div>
                </div>
              </div>
              
              {/* Sección de traducción */}
              <div className="p-8">
                <div className="h-full flex flex-col">
                  <div className="mb-6">
                    <h2 className="text-xl font-semibold text-gray-800 mb-2">Traducción</h2>
                    <p className="text-gray-600 text-sm">
                      Resultado de la traducción del texto braille
                    </p>
                  </div>
                  <div className="flex-1">
                    <BrailleTranslator capturedImage={capturedImage} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Sección de características */}
        <section className="mt-16 max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-800 mb-4">Características principales</h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Nuestra tecnología combina inteligencia artificial avanzada con un diseño accesible 
              para ofrecer la mejor experiencia de traducción braille
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-8 shadow-xl border border-white/20 text-center hover:transform hover:scale-105 transition-all duration-300">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="text-xl font-bold text-gray-800 mb-3">Traducción Instantánea</h3>
              <p className="text-gray-600">
                Procesamiento en tiempo real con resultados precisos en segundos
              </p>
            </div>
            
            <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-8 shadow-xl border border-white/20 text-center hover:transform hover:scale-105 transition-all duration-300">
              <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-xl font-bold text-gray-800 mb-3">Alta Precisión</h3>
              <p className="text-gray-600">
                Algoritmos de IA entrenados para reconocer patrones braille con 95%+ de precisión
              </p>
            </div>
            
            <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-8 shadow-xl border border-white/20 text-center hover:transform hover:scale-105 transition-all duration-300">
              <div className="w-16 h-16 bg-gradient-to-br from-orange-500 to-red-500 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="text-xl font-bold text-gray-800 mb-3">Totalmente Responsivo</h3>
              <p className="text-gray-600">
                Funciona perfectamente en dispositivos móviles, tablets y computadoras
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* Footer elegante */}
      <footer className="mt-20 bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 text-white">
        <div className="container mx-auto px-6 py-12">
          <div className="text-center space-y-6">
            <div className="flex justify-center">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                <span className="text-2xl">⠠⠊⠁</span>
              </div>
            </div>
            
            <div>
              <h3 className="text-2xl font-bold mb-2">Tecnología Inclusiva</h3>
              <p className="text-gray-300 max-w-2xl mx-auto">
                Desarrollado con el compromiso de hacer la tecnología más accesible para todos.
              </p>
            </div>
            
            <div className="border-t border-gray-700 pt-6">
              <p className="text-gray-400 text-sm">
                Derechos reservados &copy; 2025
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default BrailleApp;