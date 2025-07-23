import React, { useState } from 'react';

const NavBar: React.FC = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <nav className="bg-white/20 shadow-lg relative">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo y nombre */}
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-gray-300 backdrop-blur-sm rounded-xl flex items-center justify-center shadow-lg">
              <img src="/logo_braille_señas.svg" alt="Logo" className="w-12 h-12" />
            </div>
            <h1 className="text-xl md:text-2xl font-bold text-black">
              <span className="hidden md:inline">Intérprete Inteligente Multisensorial de Braille y LSM</span>
              <span className="md:hidden">IIMB y LSM</span>
            </h1>
          </div>
          
          {/* Menú desktop */}
          <div className="hidden md:flex items-center space-x-4">
            <a 
              href="/" 
              className="bg-gray-400 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors"
            >
              Traductor Braille
            </a>
            <a 
              href="/lsm" 
              className="bg-gray-400 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors"
            >
              Traductor LSM
            </a>
          </div>
          
          {/* Botón hamburguesa móvil */}
          <button 
            onClick={toggleMenu}
            className="md:hidden p-2 rounded-lg hover:bg-white/20 transition-colors"
          >
            <svg className="w-6 h-6 text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              {isMenuOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>
        
        {/* Menú móvil desplegable */}
        {isMenuOpen && (
          <div className="flex items-center space-x-4">
            <div className="md:hidden absolute top-full left-0 right-0 bg-white/95 backdrop-blur-sm shadow-lg border-t border-white/20 z-50">
              <div className="container mx-auto px-6 py-4 space-y-3">
                <a 
                  href="/" 
                  className="block bg-blue-600 hover:bg-blue-700 text-white px-4 py-3 rounded-lg transition-colors text-center"
                  onClick={() => setIsMenuOpen(false)}
                >
                  Traductor Braille
                </a>
                <a 
                  href="/lsm" 
                  className="block bg-green-600 hover:bg-green-700 text-white px-4 py-3 rounded-lg transition-colors text-center"
                  onClick={() => setIsMenuOpen(false)}
                >
                  Traductor LSM
                </a>
              </div>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export default NavBar;