import React, { useState, useEffect } from 'react';

interface StatisticsData {
  overall_accuracy: {
    point_estimate: string;
    confidence_interval: string;
    sample_size: number;
  };
  overall_confidence: {
    point_estimate: string;
    confidence_interval: string;
    sample_size: number;
  };
  response_time: {
    point_estimate: string;
    confidence_interval: string;
    sample_size: number;
  };
  total_predictions: number;
}

interface DetailedStatistics {
  overall_accuracy: {
    point_estimate: number;
    confidence_interval: [number, number];
    sample_size: number;
  };
  overall_confidence: {
    point_estimate: number;
    confidence_interval: [number, number];
    sample_size: number;
  };
  response_time: {
    point_estimate: number;
    confidence_interval: [number, number];
    sample_size: number;
  };
  accuracy_by_letter: Record<string, {
    point_estimate: number;
    confidence_interval: [number, number];
    sample_size: number;
  }>;
  confidence_by_letter: Record<string, {
    point_estimate: number;
    confidence_interval: [number, number];
    sample_size: number;
  }>;
  metadata: {
    confidence_level: number;
    total_predictions: number;
    timestamp: string;
  };
}

const StatisticsPanel: React.FC = () => {
  const [summaryData, setSummaryData] = useState<StatisticsData | null>(null);
  const [detailedData, setDetailedData] = useState<DetailedStatistics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'summary' | 'detailed' | 'charts'>('summary');

  const fetchSummary = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:5000/statistics/summary');
      if (!response.ok) throw new Error('Error al cargar estadísticas');
      const data = await response.json();
      setSummaryData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido');
    } finally {
      setLoading(false);
    }
  };

  const fetchDetailed = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:5000/statistics');
      if (!response.ok) throw new Error('Error al cargar estadísticas detalladas');
      const data = await response.json();
      setDetailedData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido');
    } finally {
      setLoading(false);
    }
  };

  const saveStatistics = async () => {
    try {
      const response = await fetch('http://localhost:5000/statistics/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: `braille_stats_${Date.now()}.json` })
      });
      if (!response.ok) throw new Error('Error al guardar estadísticas');
      alert('Estadísticas guardadas exitosamente');
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Error al guardar');
    }
  };

  useEffect(() => {
    fetchSummary();
  }, []);

  const renderSummaryTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Precisión General */}
        <div className="bg-white rounded-lg shadow-md p-6 border-l-4 border-blue-500">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">📊 Precisión General</h3>
          <div className="text-2xl font-bold text-blue-600 mb-2">
            {summaryData?.overall_accuracy.point_estimate || 'N/A'}
          </div>
          <div className="text-sm text-gray-600 mb-2">
            Intervalo: {summaryData?.overall_accuracy.confidence_interval || 'N/A'}
          </div>
          <div className="text-xs text-gray-500">
            Muestras: {summaryData?.overall_accuracy.sample_size || 0}
          </div>
        </div>

        {/* Confianza General */}
        <div className="bg-white rounded-lg shadow-md p-6 border-l-4 border-green-500">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">🎯 Confianza General</h3>
          <div className="text-2xl font-bold text-green-600 mb-2">
            {summaryData?.overall_confidence.point_estimate || 'N/A'}
          </div>
          <div className="text-sm text-gray-600 mb-2">
            Intervalo: {summaryData?.overall_confidence.confidence_interval || 'N/A'}
          </div>
          <div className="text-xs text-gray-500">
            Muestras: {summaryData?.overall_confidence.sample_size || 0}
          </div>
        </div>

        {/* Tiempo de Respuesta */}
        <div className="bg-white rounded-lg shadow-md p-6 border-l-4 border-purple-500">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">⏱️ Tiempo de Respuesta</h3>
          <div className="text-2xl font-bold text-purple-600 mb-2">
            {summaryData?.response_time.point_estimate || 'N/A'}
          </div>
          <div className="text-sm text-gray-600 mb-2">
            Intervalo: {summaryData?.response_time.confidence_interval || 'N/A'}
          </div>
          <div className="text-xs text-gray-500">
            Muestras: {summaryData?.response_time.sample_size || 0}
          </div>
        </div>
      </div>

      {/* Resumen General */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-semibold text-gray-800 mb-4">📈 Resumen General</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-3xl font-bold text-blue-600">
              {summaryData?.total_predictions || 0}
            </div>
            <div className="text-sm text-gray-600">Total Predicciones</div>
          </div>
          <div>
            <div className="text-3xl font-bold text-green-600">
              {summaryData?.overall_accuracy.sample_size || 0}
            </div>
            <div className="text-sm text-gray-600">Muestras Evaluadas</div>
          </div>
          <div>
            <div className="text-3xl font-bold text-purple-600">
              {summaryData?.response_time.sample_size || 0}
            </div>
            <div className="text-sm text-gray-600">Tiempos Medidos</div>
          </div>
          <div>
            <div className="text-3xl font-bold text-orange-600">
              {summaryData?.overall_confidence.sample_size || 0}
            </div>
            <div className="text-sm text-gray-600">Niveles de Confianza</div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderDetailedTab = () => (
    <div className="space-y-6">
      {detailedData && (
        <>
          {/* Precisión por Letra */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">📝 Precisión por Letra</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {Object.entries(detailedData.accuracy_by_letter).map(([letter, stats]) => (
                <div key={letter} className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="text-lg font-bold text-blue-600">{letter}</div>
                  <div className="text-sm font-semibold">
                    {(stats.point_estimate * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500">
                    [{stats.confidence_interval[0].toFixed(3)}, {stats.confidence_interval[1].toFixed(3)}]
                  </div>
                  <div className="text-xs text-gray-400">n={stats.sample_size}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Confianza por Letra */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">🎯 Confianza por Letra</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {Object.entries(detailedData.confidence_by_letter).map(([letter, stats]) => (
                <div key={letter} className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="text-lg font-bold text-green-600">{letter}</div>
                  <div className="text-sm font-semibold">
                    {(stats.point_estimate * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500">
                    [{stats.confidence_interval[0].toFixed(3)}, {stats.confidence_interval[1].toFixed(3)}]
                  </div>
                  <div className="text-xs text-gray-400">n={stats.sample_size}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Metadatos */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">📋 Metadatos</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <div className="text-sm text-gray-600">Nivel de Confianza</div>
                <div className="text-lg font-semibold">
                  {(detailedData.metadata.confidence_level * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-600">Total Predicciones</div>
                <div className="text-lg font-semibold">
                  {detailedData.metadata.total_predictions}
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-600">Última Actualización</div>
                <div className="text-lg font-semibold">
                  {new Date(detailedData.metadata.timestamp).toLocaleString()}
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );

  const renderChartsTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-semibold text-gray-800 mb-4">📊 Gráficos de Estimación</h3>
        <div className="text-center text-gray-500 py-8">
          <div className="text-4xl mb-4">📈</div>
          <p>Los gráficos interactivos estarán disponibles en la próxima versión</p>
          <p className="text-sm mt-2">
            Incluirán histogramas de distribución, gráficos de barras por letra,
            y visualizaciones de intervalos de confianza
          </p>
        </div>
      </div>
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          📊 Panel de Estadísticas
        </h1>
        <p className="text-gray-600">
          Estimaciones puntuales y por intervalos del sistema de reconocimiento Braille
        </p>
      </div>

      {/* Controles */}
      <div className="flex flex-wrap gap-4 mb-6">
        <button
          onClick={fetchSummary}
          disabled={loading}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
        >
          {loading ? 'Cargando...' : '🔄 Actualizar'}
        </button>
        
        <button
          onClick={fetchDetailed}
          disabled={loading}
          className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:opacity-50"
        >
          📊 Datos Detallados
        </button>
        
        <button
          onClick={saveStatistics}
          className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600"
        >
          💾 Guardar Estadísticas
        </button>
      </div>

      {/* Tabs */}
      <div className="mb-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {[
              { id: 'summary', label: '📋 Resumen', icon: '📋' },
              { id: 'detailed', label: '📊 Detallado', icon: '📊' },
              { id: 'charts', label: '📈 Gráficos', icon: '📈' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <div className="flex">
            <div className="text-red-400">⚠️</div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Error</h3>
              <div className="text-sm text-red-700 mt-1">{error}</div>
            </div>
          </div>
        </div>
      )}

      {/* Contenido de las tabs */}
      <div className="min-h-[400px]">
        {activeTab === 'summary' && renderSummaryTab()}
        {activeTab === 'detailed' && renderDetailedTab()}
        {activeTab === 'charts' && renderChartsTab()}
      </div>

      {/* Información adicional */}
      <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-blue-800 mb-2">ℹ️ Información sobre las Estimaciones</h3>
        <div className="text-sm text-blue-700 space-y-1">
          <p><strong>Estimación Puntual:</strong> Valor medio calculado a partir de las muestras disponibles.</p>
          <p><strong>Intervalo de Confianza:</strong> Rango donde se espera que esté el verdadero valor con 95% de confianza.</p>
          <p><strong>Tamaño de Muestra:</strong> Número de observaciones utilizadas para cada estimación.</p>
          <p><strong>Nivel de Confianza:</strong> 95% - probabilidad de que el intervalo contenga el verdadero valor.</p>
        </div>
      </div>
    </div>
  );
};

export default StatisticsPanel; 