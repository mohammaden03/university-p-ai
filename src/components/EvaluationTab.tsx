import React, { useState } from 'react';
import { BarChart3, Play, CheckCircle2, AlertCircle, FileCheck, Layers, Calculator, HelpCircle } from 'lucide-react';
import { EvaluationResponse, ModelInfo } from '../types/htr';

interface EvaluationTabProps {
  models: ModelInfo[];
  defaultModelId: string;
}

export const EvaluationTab: React.FC<EvaluationTabProps> = ({ models, defaultModelId }) => {
  const [modelId, setModelId] = useState<string>(defaultModelId || 'persian-arabic-crnn-v1');
  const [manifest, setManifest] = useState<string>('sample_persian_arabic/test.jsonl');
  const [split, setSplit] = useState<string>('test');
  const [normalize, setNormalize] = useState<boolean>(false);

  const [loading, setLoading] = useState<boolean>(false);
  const [evalResult, setEvalResult] = useState<EvaluationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Interactive Live Calculator
  const [calcRef, setCalcRef] = useState<string>('به نام خداوند جان و خرد');
  const [calcHyp, setCalcHyp] = useState<string>('به نام پروردگار جان و خرد');
  const [calcResult, setCalcResult] = useState<{ cer: number; wer: number } | null>(null);

  const runEvaluation = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch('/v1/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_manifest: manifest,
          model_id: modelId,
          split: split,
          normalize_text: normalize,
        }),
      });

      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.message || data.detail?.message || `HTTP ${resp.status}`);
      }
      setEvalResult(data);
    } catch (err: any) {
      setError(err.message || 'Evaluation request failed.');
    } finally {
      setLoading(false);
    }
  };

  const calculateQuickDistance = () => {
    // Quick client-side demonstration of Levenshtein
    const r = calcRef.trim();
    const h = calcHyp.trim();
    const rChars = Array.from(r);
    const hChars = Array.from(h);

    const dp = Array.from({ length: rChars.length + 1 }, () => Array(hChars.length + 1).fill(0));
    for (let i = 0; i <= rChars.length; i++) dp[i][0] = i;
    for (let j = 0; j <= hChars.length; j++) dp[0][j] = j;

    for (let i = 1; i <= rChars.length; i++) {
      for (let j = 1; j <= hChars.length; j++) {
        if (rChars[i - 1] === hChars[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1];
        } else {
          dp[i][j] = Math.min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1;
        }
      }
    }

    const dist = dp[rChars.length][hChars.length];
    const cer = rChars.length > 0 ? dist / rChars.length : 0;

    const rWords = r.split(/\s+/).filter(Boolean);
    const hWords = h.split(/\s+/).filter(Boolean);
    const wDp = Array.from({ length: rWords.length + 1 }, () => Array(hWords.length + 1).fill(0));
    for (let i = 0; i <= rWords.length; i++) wDp[i][0] = i;
    for (let j = 0; j <= hWords.length; j++) wDp[0][j] = j;
    for (let i = 1; i <= rWords.length; i++) {
      for (let j = 1; j <= hWords.length; j++) {
        if (rWords[i - 1] === hWords[j - 1]) wDp[i][j] = wDp[i - 1][j - 1];
        else wDp[i][j] = Math.min(wDp[i - 1][j - 1], wDp[i - 1][j], wDp[i][j - 1]) + 1;
      }
    }
    const wDist = wDp[rWords.length][hWords.length];
    const wer = rWords.length > 0 ? wDist / rWords.length : 0;

    setCalcResult({ cer: parseFloat(cer.toFixed(4)), wer: parseFloat(wer.toFixed(4)) });
  };

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/60 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-amber-400" />
            Character Error Rate (CER) & Word Error Rate (WER) Evaluation
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Standard edit-distance metric calculations conforming to university research benchmarks.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Configuration Form (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-4 space-y-4">
            <span className="text-xs font-medium text-slate-300 block uppercase tracking-wider">
              1. Benchmark Parameters
            </span>

            {/* Model Selection */}
            <div>
              <label className="text-xs text-slate-400 block mb-1">Model to Evaluate:</label>
              <select
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 text-xs text-amber-300 rounded-lg p-2 focus:ring-1 focus:ring-amber-500 font-mono"
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.id} ({m.status})
                  </option>
                ))}
              </select>
            </div>

            {/* Manifest Path */}
            <div>
              <label className="text-xs text-slate-400 block mb-1">Dataset Manifest:</label>
              <select
                value={manifest}
                onChange={(e) => setManifest(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 text-xs text-slate-200 rounded-lg p-2 focus:ring-1 focus:ring-amber-500 font-mono"
              >
                <option value="sample_persian_arabic/test.jsonl">sample_persian_arabic/test.jsonl (Held-out Test)</option>
                <option value="sample_persian_arabic/val.jsonl">sample_persian_arabic/val.jsonl (Validation)</option>
                <option value="sample_persian_arabic/manifest.jsonl">sample_persian_arabic/manifest.jsonl (All)</option>
              </select>
            </div>

            {/* Split Selector */}
            <div>
              <label className="text-xs text-slate-400 block mb-1">Split Partition:</label>
              <div className="grid grid-cols-3 gap-2">
                {['test', 'val', 'all'].map((s) => (
                  <button
                    key={s}
                    onClick={() => setSplit(s)}
                    className={`py-1.5 text-xs rounded border font-medium uppercase font-mono ${
                      split === s
                        ? 'bg-amber-600 text-white border-amber-500'
                        : 'bg-slate-900/60 border-slate-700 text-slate-400 hover:bg-slate-800'
                    }`}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>

            {/* Normalization Toggle */}
            <div className="pt-2 border-t border-slate-700/50">
              <label className="flex items-start gap-2 p-2 rounded-lg bg-slate-900/50 border border-slate-800 cursor-pointer text-xs">
                <input
                  type="checkbox"
                  checked={normalize}
                  onChange={(e) => setNormalize(e.target.checked)}
                  className="rounded border-slate-700 text-amber-600 focus:ring-0 mt-0.5"
                />
                <div>
                  <span className="text-slate-200 font-medium block">Evaluation Normalization</span>
                  <span className="text-[11px] text-slate-400 block mt-0.5">
                    Unify Alif, Arabic/Persian Kaf and Yeh, and strip decorative diacritics.
                  </span>
                </div>
              </label>
            </div>

            {/* Run Button */}
            <button
              onClick={runEvaluation}
              disabled={loading}
              className="w-full py-2.5 px-4 bg-amber-600 hover:bg-amber-500 disabled:bg-slate-700 text-white text-xs font-semibold rounded-lg shadow-md transition-all flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Evaluating Dataset Samples...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-white" />
                  <span>Execute Evaluation Benchmark</span>
                </>
              )}
            </button>
          </div>

          {/* Quick Interactive Edit Distance Sandbox */}
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-4 space-y-3">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5 font-mono">
              <Calculator className="w-3.5 h-3.5 text-amber-400" />
              Live CER/WER Metric Calculator
            </span>
            <div className="space-y-2 text-xs">
              <div>
                <label className="text-[11px] text-slate-400 block mb-1">Ground Truth (Reference):</label>
                <input
                  type="text"
                  value={calcRef}
                  onChange={(e) => setCalcRef(e.target.value)}
                  dir="rtl"
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-slate-100 font-serif"
                />
              </div>
              <div>
                <label className="text-[11px] text-slate-400 block mb-1">Hypothesis (Prediction):</label>
                <input
                  type="text"
                  value={calcHyp}
                  onChange={(e) => setCalcHyp(e.target.value)}
                  dir="rtl"
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-slate-100 font-serif"
                />
              </div>
              <button
                onClick={calculateQuickDistance}
                className="w-full py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 text-xs font-medium"
              >
                Compute Edit Distance
              </button>

              {calcResult && (
                <div className="grid grid-cols-2 gap-2 pt-2 font-mono text-center">
                  <div className="bg-slate-900 p-2 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">CER</span>
                    <span className="text-sm font-bold text-amber-400">{(calcResult.cer * 100).toFixed(2)}%</span>
                  </div>
                  <div className="bg-slate-900 p-2 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">WER</span>
                    <span className="text-sm font-bold text-sky-400">{(calcResult.wer * 100).toFixed(2)}%</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right: Results Dashboard (8 cols) */}
        <div className="lg:col-span-8 space-y-4">
          {error && (
            <div className="bg-rose-950/40 border border-rose-500/40 text-rose-200 rounded-xl p-4 text-xs flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-rose-300">Evaluation Error</p>
                <p className="mt-1 font-mono text-[11px]">{error}</p>
              </div>
            </div>
          )}

          {evalResult ? (
            <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-5">
              <div className="flex items-center justify-between border-b border-slate-700/60 pb-3">
                <div>
                  <span className="text-xs font-mono text-amber-400 uppercase tracking-wider block">
                    Model: {evalResult.model_id} (v{evalResult.model_version})
                  </span>
                  <span className="text-[11px] text-slate-400">
                    Manifest: {evalResult.dataset_manifest} (Split: {evalResult.split})
                  </span>
                </div>
                <span className="text-xs px-2.5 py-0.5 rounded bg-emerald-950/50 text-emerald-400 border border-emerald-500/30 font-mono">
                  {evalResult.metrics.evaluated_samples} evaluated
                </span>
              </div>

              {/* Main Gauges */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-lg text-center">
                  <span className="text-[11px] text-slate-400 block font-medium">Character Error Rate</span>
                  <span className="text-2xl font-bold text-amber-400 font-mono mt-1 block">
                    {(evalResult.metrics.cer * 100).toFixed(2)}%
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono">
                    (S+D+I)/{evalResult.metrics.total_chars}
                  </span>
                </div>

                <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-lg text-center">
                  <span className="text-[11px] text-slate-400 block font-medium">Word Error Rate</span>
                  <span className="text-2xl font-bold text-sky-400 font-mono mt-1 block">
                    {(evalResult.metrics.wer * 100).toFixed(2)}%
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono">
                    {evalResult.metrics.word_errors}/{evalResult.metrics.total_words} words
                  </span>
                </div>

                <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-lg text-center">
                  <span className="text-[11px] text-slate-400 block font-medium">Substitutions (S)</span>
                  <span className="text-2xl font-bold text-rose-400 font-mono mt-1 block">
                    {evalResult.metrics.char_substitutions}
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono">Mismatched chars</span>
                </div>

                <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-lg text-center">
                  <span className="text-[11px] text-slate-400 block font-medium">Deletions / Insertions</span>
                  <span className="text-2xl font-bold text-slate-200 font-mono mt-1 block">
                    {evalResult.metrics.char_deletions} / {evalResult.metrics.char_insertions}
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono">Missed / Extra</span>
                </div>
              </div>

              {/* Per-Language Breakdown */}
              {Object.keys(evalResult.per_language_results).length > 0 && (
                <div className="space-y-2">
                  <span className="text-xs font-medium text-slate-300 block uppercase font-mono">
                    Per-Language Performance:
                  </span>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                    {Object.entries(evalResult.per_language_results).map(([lang, data]) => (
                      <div key={lang} className="bg-slate-900/60 p-3 rounded-lg border border-slate-800 text-xs">
                        <span className="font-semibold text-amber-300 block font-mono">{lang}</span>
                        <div className="flex justify-between mt-1 text-slate-400">
                          <span>CER:</span>
                          <span className="font-mono text-slate-100 font-medium">{(data.cer * 100).toFixed(2)}%</span>
                        </div>
                        <div className="flex justify-between mt-0.5 text-slate-400">
                          <span>WER:</span>
                          <span className="font-mono text-slate-100 font-medium">{(data.wer * 100).toFixed(2)}%</span>
                        </div>
                        <div className="flex justify-between mt-0.5 text-slate-500 text-[10px]">
                          <span>Samples:</span>
                          <span>{data.samples} ({data.total_chars} chars)</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Sample Predictions Comparison Table */}
              <div className="space-y-2 pt-2">
                <span className="text-xs font-medium text-slate-300 block uppercase font-mono">
                  Sample Predictions vs Ground Truth:
                </span>
                <div className="overflow-x-auto rounded-lg border border-slate-800">
                  <table className="w-full text-left text-xs text-slate-300">
                    <thead className="bg-slate-900 text-slate-400 text-[10px] uppercase font-mono">
                      <tr>
                        <th className="p-2.5">Sample ID</th>
                        <th className="p-2.5">CER</th>
                        <th className="p-2.5">WER</th>
                        <th className="p-2.5 text-right">Ground Truth Reference</th>
                        <th className="p-2.5 text-right">Model Prediction</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
                      {evalResult.sample_results.map((s) => (
                        <tr key={s.sample_id} className="hover:bg-slate-800/40 transition-colors">
                          <td className="p-2.5 font-mono text-slate-400">{s.sample_id}</td>
                          <td className="p-2.5 font-mono text-amber-400">{(s.cer * 100).toFixed(1)}%</td>
                          <td className="p-2.5 font-mono text-sky-400">{(s.wer * 100).toFixed(1)}%</td>
                          <td className="p-2.5 text-right font-serif text-slate-200" dir="rtl">
                            {s.ground_truth}
                          </td>
                          <td className="p-2.5 text-right font-serif text-amber-200" dir="rtl">
                            {s.prediction || '<empty>'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-slate-800/20 border border-slate-800 rounded-xl p-12 text-center text-slate-500">
              <BarChart3 className="w-10 h-10 mx-auto mb-3 opacity-30 text-amber-400" />
              <p className="text-sm font-medium text-slate-400">No Evaluation Executed Yet</p>
              <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
                Select your model and dataset manifest on the left, then click "Execute Evaluation Benchmark" to compute Character Error Rate (CER) and Word Error Rate (WER).
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
