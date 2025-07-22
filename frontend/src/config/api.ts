// Configuración de la API para desarrollo y producción
const isDevelopment = import.meta.env.DEV;

export const API_BASE_URL = isDevelopment 
  ? 'http://localhost:5000' 
  : 'https://api.socialmeet.site'; // URL del VPS

export const API_ENDPOINTS = {
  PREDICT: `${API_BASE_URL}/predict`,
}; 