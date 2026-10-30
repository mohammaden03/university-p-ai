import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { RecognitionTab } from './components/RecognitionTab';
import { PreprocessingTab } from './components/PreprocessingTab';
import { DatasetTab } from './components/DatasetTab';
import { EvaluationTab } from './components/EvaluationTab';
import { TrainingTab } from './components/TrainingTab';
import { ApiDocsTab } from './components/ApiDocsTab';
import { HealthResponse, ReadinessResponse, ModelInfo } from './types/htr';

export default function App() {
  const [activeTab, setActiveTab] = useState<string>('recognize');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [defaultModelId, setDefaultModelId] = useState<string>('persian-arabic-crnn-v1');

  const fetchServiceMetadata = async () => {
    try {
      // 1. Health
      const hResp = await fetch('/health');
      if (hResp.ok) {
        const hData = await hResp.json();
        setHealth(hData);
      }

      // 2. Readiness
      const rResp = await fetch('/ready');
      if (rResp.ok) {
        const rData = await rResp.json();
        setReadiness(rData);
      }

      // 3. Models
      const mResp = await fetch('/v1/models');
      if (mResp.ok) {
        const mData = await mResp.json();
        setModels(mData.models || []);
        if (mData.default_model_id) {
          setDefaultModelId(mData.default_model_id);
        }
      }
    } catch {
      // Service may be initializing or proxied
    }
  };

  useEffect(() => {
    fetchServiceMetadata();
    const interval = setInterval(() => {
      fetchServiceMetadata();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-amber-500 selection:text-white">
      {/* Top Header */}
      <Header
        health={health}
        readiness={readiness}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      {/* Main Workspace Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6">
        {activeTab === 'recognize' && (
          <RecognitionTab models={models} defaultModelId={defaultModelId} />
        )}
        {activeTab === 'preprocessing' && <PreprocessingTab />}
        {activeTab === 'dataset' && <DatasetTab />}
        {activeTab === 'evaluation' && (
          <EvaluationTab models={models} defaultModelId={defaultModelId} />
        )}
        {activeTab === 'training' && (
          <TrainingTab models={models} refreshModels={fetchServiceMetadata} />
        )}
        {activeTab === 'api' && <ApiDocsTab />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-4 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>
            University HTR Project: Intelligent Recognition & Reading for Persian & Arabic Manuscripts with Islamic Calligraphy
          </span>
          <span className="font-mono text-[11px] text-slate-600">
            PyTorch • FastAPI • CRNN + CTC • Levenshtein CER/WER
          </span>
        </div>
      </footer>
    </div>
  );
}
