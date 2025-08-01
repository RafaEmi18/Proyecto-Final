#!/usr/bin/env python3
"""
Script para probar el sistema de estimación estadística del proyecto Braille
"""

import requests
import json
import time
import random
from datetime import datetime

# Configuración
BACKEND_URL = "http://localhost:5000"
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def test_backend_connection():
    """Probar conexión con el backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/")
        if response.status_code == 200:
            print("✅ Conexión con el backend exitosa")
            return True
        else:
            print(f"❌ Error de conexión: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False

def add_sample_predictions(n_samples=50):
    """Agregar predicciones de muestra para análisis estadístico"""
    print(f"\n📊 Agregando {n_samples} predicciones de muestra...")
    
    # Simular diferentes niveles de precisión por letra
    letter_accuracy = {
        'A': 0.95, 'B': 0.92, 'C': 0.88, 'D': 0.90, 'E': 0.93,
        'F': 0.87, 'G': 0.91, 'H': 0.89, 'I': 0.94, 'J': 0.86,
        'K': 0.90, 'L': 0.93, 'M': 0.88, 'N': 0.91, 'O': 0.95,
        'P': 0.87, 'Q': 0.89, 'R': 0.92, 'S': 0.90, 'T': 0.94,
        'U': 0.88, 'V': 0.91, 'W': 0.93, 'X': 0.86, 'Y': 0.90, 'Z': 0.92
    }
    
    successful_adds = 0
    
    for i in range(n_samples):
        try:
            # Seleccionar letra real
            true_letter = random.choice(LETTERS)
            
            # Generar predicción basada en la precisión de la letra
            accuracy = letter_accuracy[true_letter]
            is_correct = random.random() < accuracy
            
            if is_correct:
                predicted_letter = true_letter
            else:
                # Predicción incorrecta
                other_letters = [l for l in LETTERS if l != true_letter]
                predicted_letter = random.choice(other_letters)
            
            # Generar confianza y tiempo de respuesta realistas
            confidence = random.uniform(0.7, 0.99)
            response_time = random.uniform(0.1, 0.5)
            
            # Enviar predicción al backend
            prediction_data = {
                'predicted_letter': predicted_letter,
                'true_letter': true_letter,
                'confidence': confidence,
                'response_time': response_time
            }
            
            response = requests.post(
                f"{BACKEND_URL}/statistics/add-prediction",
                json=prediction_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                successful_adds += 1
                if (i + 1) % 10 == 0:
                    print(f"   ✅ Agregadas {i + 1} predicciones...")
            else:
                print(f"   ❌ Error al agregar predicción {i + 1}: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error en predicción {i + 1}: {e}")
    
    print(f"✅ Se agregaron {successful_adds}/{n_samples} predicciones exitosamente")
    return successful_adds

def get_statistics_summary():
    """Obtener resumen de estadísticas"""
    try:
        response = requests.get(f"{BACKEND_URL}/statistics/summary")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error al obtener estadísticas: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error al obtener estadísticas: {e}")
        return None

def get_detailed_statistics():
    """Obtener estadísticas detalladas"""
    try:
        response = requests.get(f"{BACKEND_URL}/statistics")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error al obtener estadísticas detalladas: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error al obtener estadísticas detalladas: {e}")
        return None

def save_statistics():
    """Guardar estadísticas en archivo"""
    try:
        filename = f"braille_test_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        response = requests.post(
            f"{BACKEND_URL}/statistics/save",
            json={'filename': filename},
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Estadísticas guardadas en: {result.get('filename', filename)}")
            return True
        else:
            print(f"❌ Error al guardar estadísticas: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error al guardar estadísticas: {e}")
        return False

def print_statistics_summary(stats):
    """Imprimir resumen de estadísticas de forma legible"""
    if not stats:
        print("❌ No hay estadísticas disponibles")
        return
    
    print("\n" + "="*60)
    print("📊 RESUMEN DE ESTIMACIONES ESTADÍSTICAS")
    print("="*60)
    
    # Precisión general
    acc = stats.get('overall_accuracy', {})
    print(f"\n📈 PRECISIÓN GENERAL:")
    print(f"   Estimación puntual: {acc.get('point_estimate', 'N/A')}")
    print(f"   Intervalo de confianza: {acc.get('confidence_interval', 'N/A')}")
    print(f"   Tamaño de muestra: {acc.get('sample_size', 0)}")
    
    # Confianza general
    conf = stats.get('overall_confidence', {})
    print(f"\n🎯 CONFIANZA GENERAL:")
    print(f"   Estimación puntual: {conf.get('point_estimate', 'N/A')}")
    print(f"   Intervalo de confianza: {conf.get('confidence_interval', 'N/A')}")
    print(f"   Tamaño de muestra: {conf.get('sample_size', 0)}")
    
    # Tiempo de respuesta
    time_est = stats.get('response_time', {})
    print(f"\n⏱️  TIEMPO DE RESPUESTA:")
    print(f"   Estimación puntual: {time_est.get('point_estimate', 'N/A')}")
    print(f"   Intervalo de confianza: {time_est.get('confidence_interval', 'N/A')}")
    print(f"   Tamaño de muestra: {time_est.get('sample_size', 0)}")
    
    # Total de predicciones
    total = stats.get('total_predictions', 0)
    print(f"\n📋 TOTAL DE PREDICCIONES: {total}")
    
    print("\n" + "="*60)

def print_detailed_statistics(stats):
    """Imprimir estadísticas detalladas"""
    if not stats:
        print("❌ No hay estadísticas detalladas disponibles")
        return
    
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS DETALLADAS")
    print("="*60)
    
    # Precisión por letra
    accuracy_by_letter = stats.get('accuracy_by_letter', {})
    if accuracy_by_letter:
        print(f"\n📝 PRECISIÓN POR LETRA:")
        for letter, data in sorted(accuracy_by_letter.items()):
            point_est = data.get('point_estimate', 0)
            interval = data.get('confidence_interval', [0, 0])
            sample_size = data.get('sample_size', 0)
            print(f"   {letter}: {point_est:.3f} [{interval[0]:.3f}, {interval[1]:.3f}] (n={sample_size})")
    
    # Confianza por letra
    confidence_by_letter = stats.get('confidence_by_letter', {})
    if confidence_by_letter:
        print(f"\n🎯 CONFIANZA POR LETRA:")
        for letter, data in sorted(confidence_by_letter.items()):
            point_est = data.get('point_estimate', 0)
            interval = data.get('confidence_interval', [0, 0])
            sample_size = data.get('sample_size', 0)
            print(f"   {letter}: {point_est:.3f} [{interval[0]:.3f}, {interval[1]:.3f}] (n={sample_size})")
    
    # Metadatos
    metadata = stats.get('metadata', {})
    if metadata:
        print(f"\n📋 METADATOS:")
        print(f"   Nivel de confianza: {metadata.get('confidence_level', 0)*100:.0f}%")
        print(f"   Total predicciones: {metadata.get('total_predictions', 0)}")
        print(f"   Timestamp: {metadata.get('timestamp', 'N/A')}")
    
    print("\n" + "="*60)

def main():
    """Función principal del script de prueba"""
    print("🚀 INICIANDO PRUEBAS DEL SISTEMA DE ESTIMACIÓN ESTADÍSTICA")
    print("="*60)
    
    # Probar conexión
    if not test_backend_connection():
        print("❌ No se puede continuar sin conexión al backend")
        return
    
    # Agregar predicciones de muestra
    n_samples = 100
    successful_adds = add_sample_predictions(n_samples)
    
    if successful_adds == 0:
        print("❌ No se pudieron agregar predicciones. Verificar que el backend esté funcionando.")
        return
    
    # Esperar un momento para que se procesen los datos
    print("\n⏳ Procesando datos...")
    time.sleep(2)
    
    # Obtener y mostrar estadísticas
    print("\n📊 Obteniendo estadísticas...")
    
    # Resumen
    summary_stats = get_statistics_summary()
    print_statistics_summary(summary_stats)
    
    # Detalladas
    detailed_stats = get_detailed_statistics()
    print_detailed_statistics(detailed_stats)
    
    # Guardar estadísticas
    print("\n💾 Guardando estadísticas...")
    save_statistics()
    
    print("\n✅ PRUEBAS COMPLETADAS EXITOSAMENTE")
    print("\n🎯 PRÓXIMOS PASOS:")
    print("   1. Abrir http://localhost:4321/statistics en el navegador")
    print("   2. Verificar que las estadísticas se muestren correctamente")
    print("   3. Probar las funcionalidades del panel de estadísticas")

if __name__ == "__main__":
    main() 