import React from 'react';
import { Activity, CheckCircle2, AlertTriangle, Cpu, Database, BookOpen, Layers, BarChart3, Terminal, Settings } from 'lucide-react';
import { HealthResponse, ReadinessResponse } from '../types/htr';

interface HeaderProps {
  health: HealthResponse | null;
  readiness: ReadinessResponse | null;
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  readiness,
  activeTab,
  setActiveTab,
}) => {
  const isReady = readiness?.ready ?? false;
  const modelStatus = readiness?.default_model_status ?? 'UNKNOWN';

  const navItems = [
    { id: 'recognize', label: 'Manuscript HTR', icon: BookOpen },
    { id: 'preprocessing', label: 'Image Preprocessing', icon: Layers },
    { id: 'dataset', label: 'Datasets & Manifests', icon: Database },
    { id: 'evaluation', label: 'CER & WER Metrics', icon: BarChart3 },
    { id: 'training', label: 'Model Hub & Training', icon: Settings },
    { id: 'api', label: 'API & NestJS Contract', icon: Terminal },
  ];

  return (
    <header className="bg-slate-900 border-b border-slate-800 text-slate-100 sticky top-0 z-50 shadow-md">
      {/* Top Banner */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-amber-600/20 border border-amber-500/30 flex items-center justify-center text-amber-400 font-bold text-xl shadow-inner">
            خط
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-semibold tracking-tight text-amber-100">
                Persian & Arabic Manuscript HTR/OCR AI Service
              </h1>
              <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-amber-400 border border-amber-500/20 font-mono">
                v{health?.version || '1.0.0'}
              </span>
            </div>
            <p className="text-xs text-slate-400 font-sans" dir="rtl">
              سامانه هوشمند بازشناسی خطوط و نسخ خطی فارسی و عربی با خوشنویسی اسلامی
            </p>
          </div>
        </div>

        {/* Live Service Telemetry Badges */}
        <div className="flex flex-wrap items-center gap-2 self-stretch md:self-auto text-xs">
          {/* Service Liveness */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700">
            <Activity className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
            <span className="text-slate-300">FastAPI:</span>
            <span className="text-emerald-400 font-medium">{health?.status === 'ok' ? 'Online' : 'Starting...'}</span>
          </div>

          {/* Model Readiness */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700">
            {isReady ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
            )}
            <span className="text-slate-300">CRNN Engine:</span>
            <span className={isReady ? 'text-emerald-400 font-medium' : 'text-amber-400 font-medium'}>
              {modelStatus}
            </span>
          </div>

          {/* Device */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700">
            <Cpu className="w-3.5 h-3.5 text-sky-400" />
            <span className="text-slate-300">Device:</span>
            <span className="text-sky-300 font-mono uppercase">
              {readiness?.dependencies?.pytorch?.device || 'CPU'}
            </span>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <nav className="flex space-x-1 overflow-x-auto scrollbar-none py-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-t-md text-xs sm:text-sm font-medium transition-colors whitespace-nowrap border-b-2 ${
                  active
                    ? 'border-amber-500 text-amber-300 bg-slate-800/60'
                    : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/30'
                }`}
              >
                <Icon className={`w-4 h-4 ${active ? 'text-amber-400' : 'text-slate-400'}`} />
                {item.label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
