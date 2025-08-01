import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Dict, Optional
import time
import json
import os
from datetime import datetime

class BrailleStatisticalEstimator:
    """
    Clase para realizar estimación puntual y por intervalos en el sistema de Braille
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Inicializar el estimador estadístico
        
        Args:
            confidence_level: Nivel de confianza para los intervalos (default: 0.95)
        """
        self.confidence_level = confidence_level
        self.prediction_history = []
        self.response_times = []
        self.accuracy_history = []
        self.confidence_scores = []
        
    def add_prediction(self, predicted_letter: str, true_letter: str, 
                      confidence: float, response_time: float):
        """
        Agregar una nueva predicción al historial para análisis estadístico
        
        Args:
            predicted_letter: Letra predicha por el modelo
            true_letter: Letra real (ground truth)
            confidence: Nivel de confianza de la predicción
            response_time: Tiempo de respuesta en segundos
        """
        prediction_data = {
            'timestamp': datetime.now().isoformat(),
            'predicted': predicted_letter,
            'true': true_letter,
            'confidence': confidence,
            'response_time': response_time,
            'correct': predicted_letter == true_letter
        }
        
        self.prediction_history.append(prediction_data)
        self.response_times.append(response_time)
        self.confidence_scores.append(confidence)
        
        if predicted_letter == true_letter:
            self.accuracy_history.append(1)
        else:
            self.accuracy_history.append(0)
    
    def estimate_accuracy_point(self) -> float:
        """
        Estimación puntual de la precisión del modelo
        
        Returns:
            float: Precisión media del modelo
        """
        if not self.accuracy_history:
            return 0.0
        return np.mean(self.accuracy_history)
    
    def estimate_accuracy_interval(self) -> Tuple[float, float]:
        """
        Intervalo de confianza para la precisión del modelo
        
        Returns:
            Tuple[float, float]: (límite inferior, límite superior) del intervalo
        """
        if len(self.accuracy_history) < 2:
            return (0.0, 1.0)
        
        n = len(self.accuracy_history)
        p_hat = np.mean(self.accuracy_history)
        
        # Usar aproximación normal para proporciones
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        margin_of_error = z_score * np.sqrt((p_hat * (1 - p_hat)) / n)
        
        lower_bound = max(0, p_hat - margin_of_error)
        upper_bound = min(1, p_hat + margin_of_error)
        
        return (lower_bound, upper_bound)
    
    def estimate_confidence_point(self) -> float:
        """
        Estimación puntual del nivel de confianza promedio
        
        Returns:
            float: Confianza media del modelo
        """
        if not self.confidence_scores:
            return 0.0
        return np.mean(self.confidence_scores)
    
    def estimate_confidence_interval(self) -> Tuple[float, float]:
        """
        Intervalo de confianza para el nivel de confianza promedio
        
        Returns:
            Tuple[float, float]: (límite inferior, límite superior) del intervalo
        """
        if len(self.confidence_scores) < 2:
            return (0.0, 1.0)
        
        confidence_mean = np.mean(self.confidence_scores)
        confidence_std = np.std(self.confidence_scores, ddof=1)
        n = len(self.confidence_scores)
        
        # Intervalo de confianza t-student
        t_score = stats.t.ppf((1 + self.confidence_level) / 2, df=n-1)
        margin_of_error = t_score * (confidence_std / np.sqrt(n))
        
        lower_bound = max(0, confidence_mean - margin_of_error)
        upper_bound = min(1, confidence_mean + margin_of_error)
        
        return (lower_bound, upper_bound)
    
    def estimate_response_time_point(self) -> float:
        """
        Estimación puntual del tiempo de respuesta promedio
        
        Returns:
            float: Tiempo de respuesta medio en segundos
        """
        if not self.response_times:
            return 0.0
        return np.mean(self.response_times)
    
    def estimate_response_time_interval(self) -> Tuple[float, float]:
        """
        Intervalo de confianza para el tiempo de respuesta promedio
        
        Returns:
            Tuple[float, float]: (límite inferior, límite superior) del intervalo
        """
        if len(self.response_times) < 2:
            return (0.0, float('inf'))
        
        time_mean = np.mean(self.response_times)
        time_std = np.std(self.response_times, ddof=1)
        n = len(self.response_times)
        
        # Intervalo de confianza t-student
        t_score = stats.t.ppf((1 + self.confidence_level) / 2, df=n-1)
        margin_of_error = t_score * (time_std / np.sqrt(n))
        
        lower_bound = max(0, time_mean - margin_of_error)
        upper_bound = time_mean + margin_of_error
        
        return (lower_bound, upper_bound)
    
    def estimate_accuracy_by_letter(self) -> Dict[str, Dict]:
        """
        Estimación de precisión por letra individual
        
        Returns:
            Dict: Diccionario con precisión puntual e intervalos por letra
        """
        letter_stats = {}
        
        for prediction in self.prediction_history:
            true_letter = prediction['true']
            if true_letter not in letter_stats:
                letter_stats[true_letter] = {'correct': 0, 'total': 0}
            
            letter_stats[true_letter]['total'] += 1
            if prediction['correct']:
                letter_stats[true_letter]['correct'] += 1
        
        results = {}
        for letter, stats in letter_stats.items():
            if stats['total'] >= 1:
                p_hat = stats['correct'] / stats['total']
                
                # Intervalo de confianza para proporción
                z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
                margin_of_error = z_score * np.sqrt((p_hat * (1 - p_hat)) / stats['total'])
                
                lower_bound = max(0, p_hat - margin_of_error)
                upper_bound = min(1, p_hat + margin_of_error)
                
                results[letter] = {
                    'point_estimate': p_hat,
                    'confidence_interval': (lower_bound, upper_bound),
                    'sample_size': stats['total']
                }
        
        return results
    
    def estimate_confidence_by_letter(self) -> Dict[str, Dict]:
        """
        Estimación de confianza promedio por letra
        
        Returns:
            Dict: Diccionario con confianza puntual e intervalos por letra
        """
        letter_confidences = {}
        
        for prediction in self.prediction_history:
            predicted_letter = prediction['predicted']
            confidence = prediction['confidence']
            
            if predicted_letter not in letter_confidences:
                letter_confidences[predicted_letter] = []
            
            letter_confidences[predicted_letter].append(confidence)
        
        results = {}
        for letter, confidences in letter_confidences.items():
            if len(confidences) >= 1:
                confidence_mean = np.mean(confidences)
                
                if len(confidences) >= 2:
                    confidence_std = np.std(confidences, ddof=1)
                    t_score = stats.t.ppf((1 + self.confidence_level) / 2, df=len(confidences)-1)
                    margin_of_error = t_score * (confidence_std / np.sqrt(len(confidences)))
                    
                    lower_bound = max(0, confidence_mean - margin_of_error)
                    upper_bound = min(1, confidence_mean + margin_of_error)
                else:
                    lower_bound = upper_bound = confidence_mean
                
                results[letter] = {
                    'point_estimate': confidence_mean,
                    'confidence_interval': (lower_bound, upper_bound),
                    'sample_size': len(confidences)
                }
        
        return results
    
    def get_comprehensive_estimates(self) -> Dict:
        """
        Obtener todas las estimaciones puntuales y por intervalos
        
        Returns:
            Dict: Diccionario completo con todas las estimaciones
        """
        return {
            'overall_accuracy': {
                'point_estimate': self.estimate_accuracy_point(),
                'confidence_interval': self.estimate_accuracy_interval(),
                'sample_size': len(self.accuracy_history)
            },
            'overall_confidence': {
                'point_estimate': self.estimate_confidence_point(),
                'confidence_interval': self.estimate_confidence_interval(),
                'sample_size': len(self.confidence_scores)
            },
            'response_time': {
                'point_estimate': self.estimate_response_time_point(),
                'confidence_interval': self.estimate_response_time_interval(),
                'sample_size': len(self.response_times)
            },
            'accuracy_by_letter': self.estimate_accuracy_by_letter(),
            'confidence_by_letter': self.estimate_confidence_by_letter(),
            'metadata': {
                'confidence_level': self.confidence_level,
                'total_predictions': len(self.prediction_history),
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def save_estimates(self, filename: str = None):
        """
        Guardar las estimaciones en un archivo JSON
        
        Args:
            filename: Nombre del archivo (opcional)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"braille_estimates_{timestamp}.json"
        
        estimates = self.get_comprehensive_estimates()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(estimates, f, indent=2, ensure_ascii=False)
        
        print(f"Estimaciones guardadas en: {filename}")
    
    def load_estimates(self, filename: str) -> Dict:
        """
        Cargar estimaciones desde un archivo JSON
        
        Args:
            filename: Nombre del archivo a cargar
            
        Returns:
            Dict: Estimaciones cargadas
        """
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def print_summary(self):
        """
        Imprimir un resumen de las estimaciones
        """
        estimates = self.get_comprehensive_estimates()
        
        print("=" * 60)
        print("ESTIMACIONES ESTADÍSTICAS DEL SISTEMA BRAILLE")
        print("=" * 60)
        
        # Precisión general
        acc = estimates['overall_accuracy']
        print(f"\n📊 PRECISIÓN GENERAL:")
        print(f"   Estimación puntual: {acc['point_estimate']:.3f} ({acc['point_estimate']*100:.1f}%)")
        print(f"   Intervalo de confianza ({self.confidence_level*100:.0f}%): "
              f"[{acc['confidence_interval'][0]:.3f}, {acc['confidence_interval'][1]:.3f}]")
        print(f"   Tamaño de muestra: {acc['sample_size']}")
        
        # Confianza general
        conf = estimates['overall_confidence']
        print(f"\n🎯 CONFIANZA GENERAL:")
        print(f"   Estimación puntual: {conf['point_estimate']:.3f} ({conf['point_estimate']*100:.1f}%)")
        print(f"   Intervalo de confianza ({self.confidence_level*100:.0f}%): "
              f"[{conf['confidence_interval'][0]:.3f}, {conf['confidence_interval'][1]:.3f}]")
        print(f"   Tamaño de muestra: {conf['sample_size']}")
        
        # Tiempo de respuesta
        time_est = estimates['response_time']
        print(f"\n⏱️  TIEMPO DE RESPUESTA:")
        print(f"   Estimación puntual: {time_est['point_estimate']:.3f} segundos")
        print(f"   Intervalo de confianza ({self.confidence_level*100:.0f}%): "
              f"[{time_est['confidence_interval'][0]:.3f}, {time_est['confidence_interval'][1]:.3f}] segundos")
        print(f"   Tamaño de muestra: {time_est['sample_size']}")
        
        # Precisión por letra
        if estimates['accuracy_by_letter']:
            print(f"\n📝 PRECISIÓN POR LETRA:")
            for letter, stats in estimates['accuracy_by_letter'].items():
                print(f"   {letter}: {stats['point_estimate']:.3f} "
                      f"[{stats['confidence_interval'][0]:.3f}, {stats['confidence_interval'][1]:.3f}] "
                      f"(n={stats['sample_size']})")
        
        print("\n" + "=" * 60)

# Función de utilidad para crear datos de prueba
def create_sample_data(estimator: BrailleStatisticalEstimator, n_samples: int = 100):
    """
    Crear datos de muestra para probar las estimaciones
    
    Args:
        estimator: Instancia del estimador
        n_samples: Número de muestras a generar
    """
    import random
    
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    # Simular diferentes niveles de precisión por letra
    letter_accuracy = {
        'A': 0.95, 'B': 0.92, 'C': 0.88, 'D': 0.90, 'E': 0.93,
        'F': 0.87, 'G': 0.91, 'H': 0.89, 'I': 0.94, 'J': 0.86,
        'K': 0.90, 'L': 0.93, 'M': 0.88, 'N': 0.91, 'O': 0.95,
        'P': 0.87, 'Q': 0.89, 'R': 0.92, 'S': 0.90, 'T': 0.94,
        'U': 0.88, 'V': 0.91, 'W': 0.93, 'X': 0.86, 'Y': 0.90, 'Z': 0.92
    }
    
    for _ in range(n_samples):
        true_letter = random.choice(letters)
        confidence = random.uniform(0.7, 0.99)
        response_time = random.uniform(0.1, 0.5)
        
        # Determinar si la predicción es correcta basada en la precisión de la letra
        accuracy = letter_accuracy[true_letter]
        is_correct = random.random() < accuracy
        
        if is_correct:
            predicted_letter = true_letter
        else:
            # Predicción incorrecta
            other_letters = [l for l in letters if l != true_letter]
            predicted_letter = random.choice(other_letters)
        
        estimator.add_prediction(predicted_letter, true_letter, confidence, response_time)

if __name__ == "__main__":
    # Ejemplo de uso
    estimator = BrailleStatisticalEstimator(confidence_level=0.95)
    
    # Crear datos de muestra
    create_sample_data(estimator, n_samples=200)
    
    # Mostrar resumen
    estimator.print_summary()
    
    # Guardar estimaciones
    estimator.save_estimates() 