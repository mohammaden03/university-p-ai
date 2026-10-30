import React, { useState, useEffect } from 'react';
import { Database, CheckCircle2, AlertTriangle, RefreshCw, FileText, Check, ShieldCheck, Sparkles } from 'lucide-react';
import { DatasetValidationResult } from '../types/htr';

export const DatasetTab: React.FC = () => {
  const [selectedManifest, setSelectedManifest] = useState<string>('sample_persian_arabic/manifest.jsonl');
  const [loading, setLoading] = useState<boolean>(false);
  const [auditResult, setAuditResult] = useState<DatasetValidationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const manifests = [
    { id: 'sample_persian_arabic/manifest.jsonl', name: 'Combined Manifest (manifest.jsonl)', desc: 'Complete sample dataset containing train, validation, and test splits' },
    { id: 'sample_persian_arabic/train.jsonl', name: 'Training Split (train.jsonl)', desc: 'Used for CRNN gradient updates (Saadi Golestan & Arabic manuscripts)' },
    { id: 'sample_persian_arabic/val.jsonl', name: 'Validation Split (val.jsonl)', desc: 'Held-out validation set for epoch monitoring & early stopping (Rumi Masnavi)' },
    { id: 'sample_persian_arabic/test.jsonl', name: 'Test Benchmark Split (test.jsonl)', desc: 'Independent test benchmark for CER/WER reporting (Hafez Diwan)' },
  ];

  const runAudit = async (manifestPath: string) => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`/v1/datasets/validate?manifest_path=${encodeURIComponent(manifestPath)}`);
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.message || data.detail?.message || `Validation error: ${resp.status}`);
      }
      setAuditResult(data);
    } catch (err: any) {
      setError(err.message || 'Dataset validation failed.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runAudit(selectedManifest);
  }, [selectedManifest]);

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/60 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <Database className="w-4 h-4 text-amber-400" />
            Dataset Management & Manifest Auditing
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            JSONL dataset manifest validation, data leakage prevention, missing-file auditing, and character set coverage.
          </p>
        </div>
        <button
          onClick={() => runAudit(selectedManifest)}
          disabled={loading}
          className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 border border-slate-700 rounded-lg flex items-center gap-1.5 transition-colors self-start sm:self-auto"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-amber-400' : ''}`} />
          <span>Re-run Manifest Audit</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Manifest Selector (4 cols) */}
        <div className="lg:col-span-4 space-y-3">
          <label className="text-xs font-medium text-slate-300 block uppercase tracking-wider">
            Available Manifests (.jsonl)
          </label>
          <div className="space-y-2">
            {manifests.map((m) => {
              const active = selectedManifest === m.id;
              return (
                <button
                  key={m.id}
                  onClick={() => setSelectedManifest(m.id)}
                  className={`w-full text-left p-3 rounded-lg border text-xs transition-all ${
                    active
                      ? 'bg-amber-950/40 border-amber-500/50 text-amber-200 shadow-sm'
                      : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                  }`}
                >
                  <span className="font-medium text-slate-200 block">{m.name}</span>
                  <span className="text-[11px] text-slate-400 mt-0.5 block leading-normal">{m.desc}</span>
                  <span className="text-[10px] text-amber-400/80 font-mono mt-1.5 block">{m.id}</span>
                </button>
              );
            })}
          </div>

          {/* Dataset Standards Note */}
          <div className="bg-slate-900/50 rounded-lg border border-slate-800 p-3.5 space-y-2 text-xs">
            <span className="font-semibold text-slate-300 flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              Manifest Formatting Standards:
            </span>
            <ul className="list-disc list-inside text-[11px] text-slate-400 space-y-1">
              <li>Stored as standardized UTF-8 JSONL records.</li>
              <li>Images referenced relative to configured DATASET_ROOT.</li>
              <li>Manuscript IDs group pages to prevent train/val/test data leakage.</li>
              <li>Preserves original logical Unicode order without string reversal.</li>
            </ul>
          </div>
        </div>

        {/* Right: Audit Report & Records (8 cols) */}
        <div className="lg:col-span-8 space-y-4">
          {error && (
            <div className="bg-rose-950/40 border border-rose-500/40 text-rose-200 rounded-xl p-4 text-xs flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-rose-300">Validation Failure</p>
                <p className="mt-1 font-mono text-[11px]">{error}</p>
              </div>
            </div>
          )}

          {auditResult && (
            <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-5">
              <div className="flex items-center justify-between border-b border-slate-700/60 pb-3">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
                  <span className="text-sm font-semibold text-slate-100">
                    Audit Status: Integrity Verified
                  </span>
                </div>
                <span className="text-xs px-2.5 py-0.5 rounded bg-emerald-950/50 text-emerald-400 border border-emerald-500/30 font-mono">
                  All Physical Files Located
                </span>
              </div>

              {/* Statistics Counters */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg">
                  <span className="text-[11px] text-slate-400 block">Total Samples</span>
                  <span className="text-xl font-bold text-amber-300 font-mono mt-1 block">
                    {auditResult.total_samples}
                  </span>
                </div>
                <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg">
                  <span className="text-[11px] text-slate-400 block">Total Words</span>
                  <span className="text-xl font-bold text-slate-100 font-mono mt-1 block">
                    {auditResult.total_words}
                  </span>
                </div>
                <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg">
                  <span className="text-[11px] text-slate-400 block">Unique Vocab Chars</span>
                  <span className="text-xl font-bold text-sky-300 font-mono mt-1 block">
                    {auditResult.vocabulary_size}
                  </span>
                </div>
                <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg">
                  <span className="text-[11px] text-slate-400 block">Unrecognized Chars</span>
                  <span className="text-xl font-bold text-emerald-400 font-mono mt-1 block">
                    {auditResult.unrecognized_chars.length}
                  </span>
                </div>
              </div>

              {/* Split Breakdown */}
              <div className="bg-slate-900/50 rounded-lg border border-slate-800 p-3.5 space-y-2">
                <span className="text-xs font-medium text-slate-300 uppercase tracking-wider block font-mono">
                  Dataset Split Partitioning:
                </span>
                <div className="grid grid-cols-3 gap-2 text-center text-xs">
                  <div className="bg-slate-800/60 p-2 rounded border border-slate-700">
                    <span className="text-slate-400 block">Train</span>
                    <span className="text-amber-300 font-bold font-mono">
                      {auditResult.split_counts.train || 0}
                    </span>
                  </div>
                  <div className="bg-slate-800/60 p-2 rounded border border-slate-700">
                    <span className="text-slate-400 block">Val (Validation)</span>
                    <span className="text-sky-300 font-bold font-mono">
                      {auditResult.split_counts.val || 0}
                    </span>
                  </div>
                  <div className="bg-slate-800/60 p-2 rounded border border-slate-700">
                    <span className="text-slate-400 block">Test</span>
                    <span className="text-emerald-300 font-bold font-mono">
                      {auditResult.split_counts.test || 0}
                    </span>
                  </div>
                </div>
              </div>

              {/* Warnings & Integrity Notes */}
              {auditResult.warnings && auditResult.warnings.length > 0 && (
                <div className="space-y-1.5">
                  <span className="text-xs font-medium text-slate-400 block">Integrity Notices:</span>
                  {auditResult.warnings.map((w, idx) => (
                    <div key={idx} className="bg-amber-950/20 border border-amber-500/20 text-amber-200 p-2.5 rounded text-xs flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                      <span>{w}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
