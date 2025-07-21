import React from 'react';

const NavBar: React.FC = () => {
  return (
    <nav className="bg-white/20 shadow-lg">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo y nombre */}
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-gray-300 backdrop-blur-sm rounded-xl flex items-center justify-center shadow-lg">
              <span className="text-2xl text-zinc-500">⠠⠊⠁</span>
            </div>
            <h1 className="text-2xl font-bold text-black">
            Intérprete Inteligente Multisensorial de Braille y LSM
            </h1>
          </div>
          
          {/* Espacio para futuras opciones de navegación */}
          <div className="flex items-center space-x-4">
            {/* Espacio para botón de Guante Traductor */}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;