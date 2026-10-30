import React, { useState, useEffect } from 'react';
import { Settings, Play, CheckCircle2, AlertTriangle, Clock, RefreshCw, Cpu, Layers, HardDrive } from 'lucide-react';
import { ModelInfo, TrainingJobStatus } from '../types/htr';

interface TrainingTabProps {
  models: ModelInfo[];
  refreshModels: () => void;
}

export const TrainingTab: React.FC<TrainingTabProps> = ({ models, refreshModels }) => {
  // Training form parameters
  const [modelName, setModelName] = useState<string>('persian-crnn-v2');
  const [trainManifest, setTrainManifest] = useState<string>('sample_persian_arabic/train.jsonl');
  const [valManifest, setValManifest] = useState<string>('sample_persian_arabic/val.jsonl');
  const [epochs, setEpochs] = useState<number>(3);
  const [batchSize, setBatchSize] = useState<number>(4);
  const [learningRate, setLearningRate] = useState<number>(0.001);

  const [starting, setStarting] = useState<boolean>(false);
  const [startError, setStartError] = useState<string | null>(null);

  // Job tracking
  const [jobs, setJobs] = useState<TrainingJobStatus[]>([]);
  const [polling, setPolling] = useState<boolean>(false);

  const fetchJobs = async () => {
    try {
      const resp = await fetch('/v1/train');
      if (resp.ok) {
        const data = await resp.json();
        setJobs(data);
      }
    } catch {
      // ignore transient polling errors
    }
  };

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(() => {
      fetchJobs();
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleStartTraining = async (e: React.FormEvent) => {
    e.preventDefault();
    setStarting(true);
    setStartError(null);

    try {
      const resp = await fetch('/v1/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName,
          train_manifest: trainManifest,
          val_manifest: valManifest,
          epochs: Number(epochs),
          batch_size: Number(batchSize),
          learning_rate: Number(learningRate),
          random_seed: 42,
        }),
      });

      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.message || data.detail?.message || `Failed to start job: ${resp.status}`);
      }

      await fetchJobs();
      refreshModels();
    } catch (err: any) {
      setStartError(err.message || 'Failed to dispatch training job.');
    } finally {
      setStarting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/60 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <Settings className="w-4 h-4 text-amber-400" />
            Model Registry & Checkpoint Training Hub
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Manage HTR model lifecycles, monitor PyTorch CRNN training jobs, and inspect honest availability states.
          </p>
        </div>
        <button
          onClick={() => {
            fetchJobs();
            refreshModels();
          }}
          className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 border border-slate-700 rounded-lg flex items-center gap-1.5 transition-colors self-start sm:self-auto"
        >
          <RefreshCw className="w-3.5 h-3.5 text-amber-400" />
          <span>Refresh Registry</span>
        </button>
      </div>

      {/* Model Cards Grid */}
      <div className="space-y-3">
        <span className="text-xs font-medium text-slate-300 block uppercase tracking-wider font-mono">
          Configured Recognition Engines & Availability:
        </span>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {models.map((m) => {
            const isReady = m.status === 'MODEL_READY';
            return (
              <div
                key={m.id}
                className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-4 flex flex-col justify-between space-y-3"
              >
                <div>
                  <div className="flex items-start justify-between gap-2">
                    <span className="font-semibold text-slate-100 text-sm">{m.name}</span>
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded font-mono font-medium border ${
                        isReady
                          ? 'bg-emerald-950/60 text-emerald-400 border-emerald-500/30'
                          : 'bg-amber-950/60 text-amber-400 border-amber-500/30'
                      }`}
                    >
                      {m.status}
                    </span>
                  </div>
                  <span className="text-[11px] font-mono text-amber-300 block mt-1">{m.id}</span>
                  <p className="text-xs text-slate-400 mt-2 line-clamp-3">{m.known_limitations || 'No limitations recorded.'}</p>
                </div>

                <div className="pt-2 border-t border-slate-700/50 space-y-1 text-[11px]">
                  <div className="flex justify-between text-slate-400">
                    <span>Engine Type:</span>
                    <span className="font-mono text-slate-200 uppercase">{m.engine_type}</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Languages:</span>
                    <span className="text-amber-200 font-medium">{m.supported_languages.join(', ')}</span>
                  </div>
                  {m.checkpoint_path && (
                    <div className="text-slate-500 font-mono text-[10px] truncate" title={m.checkpoint_path}>
                      Checkpoint: {m.checkpoint_path}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Dispatch Training Form (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <form
            onSubmit={handleStartTraining}
            className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-4"
          >
            <span className="text-xs font-semibold text-slate-200 block uppercase tracking-wider font-mono">
              Launch Asynchronous Model Training:
            </span>

            {startError && (
              <div className="bg-rose-950/40 border border-rose-500/40 text-rose-200 rounded-lg p-3 text-xs flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <span>{startError}</span>
              </div>
            )}

            <div>
              <label className="text-xs text-slate-400 block mb-1">New Model Identifier:</label>
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                required
                className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs text-amber-300 font-mono"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-slate-400 block mb-1">Epochs:</label>
                <input
                  type="number"
                  min="1"
                  max="50"
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                  required
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs text-slate-200 font-mono"
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 block mb-1">Batch Size:</label>
                <input
                  type="number"
                  min="1"
                  max="32"
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  required
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs text-slate-200 font-mono"
                />
              </div>
            </div>

            <div>
              <label className="text-xs text-slate-400 block mb-1">Learning Rate (Adam):</label>
              <input
                type="number"
                step="0.0001"
                min="0.0001"
                max="0.05"
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                required
                className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs text-slate-200 font-mono"
              />
            </div>

            <div>
              <label className="text-xs text-slate-400 block mb-1">Training Manifest:</label>
              <input
                type="text"
                value={trainManifest}
                onChange={(e) => setTrainManifest(e.target.value)}
                required
                className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs text-slate-300 font-mono text-[11px]"
              />
            </div>

            <div>
              <label className="text-xs text-slate-400 block mb-1">Validation Manifest:</label>
              <input
                type="text"
                value={valManifest}
                onChange={(e) => setValManifest(e.target.value)}
                required
                className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs text-slate-300 font-mono text-[11px]"
              />
            </div>

            <button
              type="submit"
              disabled={starting}
              className="w-full py-2.5 px-4 bg-amber-600 hover:bg-amber-500 disabled:bg-slate-700 text-white text-xs font-semibold rounded-lg shadow-md transition-all flex items-center justify-center gap-2"
            >
              {starting ? (
                <>
                  <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Submitting Training Job...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-white" />
                  <span>Start PyTorch CRNN Training Run</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Right: Live Training Jobs Monitor (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200 uppercase tracking-wider font-mono">
                Training Jobs & Loss Progress:
              </span>
              <span className="text-[11px] text-slate-400 font-mono">{jobs.length} recorded job(s)</span>
            </div>

            {jobs.length === 0 ? (
              <div className="text-center py-10 text-slate-500 text-xs">
                <Clock className="w-8 h-8 mx-auto mb-2 opacity-30 text-amber-400" />
                <span>No active training jobs. Launch a training run to monitor loss and validation CER.</span>
              </div>
            ) : (
              <div className="space-y-3">
                {jobs.map((j) => {
                  const isRunning = j.status === 'RUNNING';
                  const isCompleted = j.status === 'COMPLETED';
                  const isFailed = j.status === 'FAILED';

                  return (
                    <div
                      key={j.job_id}
                      className="bg-slate-900/80 border border-slate-800 rounded-lg p-4 space-y-3"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-amber-300 font-medium text-xs">{j.job_id}</span>
                          <span className="text-xs text-slate-300">({j.model_name})</span>
                        </div>
                        <span
                          className={`text-[10px] px-2 py-0.5 rounded font-mono font-medium border ${
                            isCompleted
                              ? 'bg-emerald-950/60 text-emerald-400 border-emerald-500/30'
                              : isRunning
                              ? 'bg-sky-950/60 text-sky-400 border-sky-500/30 animate-pulse'
                              : isFailed
                              ? 'bg-rose-950/60 text-rose-400 border-rose-500/30'
                              : 'bg-slate-800 text-slate-400 border-slate-700'
                          }`}
                        >
                          {j.status}
                        </span>
                      </div>

                      {/* Progress Metrics */}
                      <div className="grid grid-cols-3 gap-2 text-center text-xs">
                        <div className="bg-slate-950 p-2 rounded border border-slate-800">
                          <span className="text-[10px] text-slate-500 block">Epoch</span>
                          <span className="font-mono text-slate-200 font-medium">
                            {j.current_epoch} / {j.total_epochs}
                          </span>
                        </div>
                        <div className="bg-slate-950 p-2 rounded border border-slate-800">
                          <span className="text-[10px] text-slate-500 block">Train Loss</span>
                          <span className="font-mono text-amber-400 font-medium">
                            {j.train_loss !== null ? j.train_loss.toFixed(4) : '—'}
                          </span>
                        </div>
                        <div className="bg-slate-950 p-2 rounded border border-slate-800">
                          <span className="text-[10px] text-slate-500 block">Val CER</span>
                          <span className="font-mono text-emerald-400 font-medium">
                            {j.val_cer !== null ? `${(j.val_cer * 100).toFixed(1)}%` : '—'}
                          </span>
                        </div>
                      </div>

                      {/* Checkpoint or Error */}
                      {j.checkpoint_path && (
                        <div className="text-[10px] text-slate-400 font-mono truncate" title={j.checkpoint_path}>
                          Saved: {j.checkpoint_path}
                        </div>
                      )}
                      {j.error_message && (
                        <div className="text-[10px] text-rose-400 font-mono">Error: {j.error_message}</div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
