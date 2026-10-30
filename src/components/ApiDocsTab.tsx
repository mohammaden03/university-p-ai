import React, { useState } from 'react';
import { Terminal, Copy, Check, ExternalLink, Code2, Shield, Network } from 'lucide-react';

export const ApiDocsTab: React.FC = () => {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const copyToClipboard = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const curlSnippets = [
    {
      title: 'Health Check (Liveness)',
      method: 'GET',
      path: '/health',
      curl: `curl -s http://localhost:8001/health`,
      response: `{\n  "status": "ok",\n  "version": "1.0.0",\n  "ready": true,\n  "timestamp": "2026-10-30T08:35:01Z"\n}`,
    },
    {
      title: 'Readiness Probe (Dependencies & Models)',
      method: 'GET',
      path: '/ready',
      curl: `curl -s http://localhost:8001/ready`,
      response: `{\n  "ready": true,\n  "service_status": "ready",\n  "default_model_status": "MODEL_READY",\n  "dependencies": {\n    "pytorch": { "version": "2.15.0+cpu", "cuda_available": false },\n    "default_model": { "id": "persian-arabic-crnn-v1", "status": "MODEL_READY" }\n  }\n}`,
    },
    {
      title: 'List Configured Models',
      method: 'GET',
      path: '/v1/models',
      curl: `curl -s http://localhost:8001/v1/models`,
      response: `{\n  "default_model_id": "persian-arabic-crnn-v1",\n  "models": [\n    {\n      "id": "persian-arabic-crnn-v1",\n      "name": "Persian-Arabic Deep CRNN HTR Model",\n      "version": "1.0.0",\n      "supported_languages": ["PERSIAN", "ARABIC", "MIXED"],\n      "status": "MODEL_READY"\n    }\n  ]\n}`,
    },
    {
      title: 'Recognize Uploaded Image (Multipart)',
      method: 'POST',
      path: '/v1/recognize',
      curl: `curl -X POST http://localhost:8001/v1/recognize \\\n  -F "file=@/path/to/manuscript_line.jpg" \\\n  -F "language=PERSIAN" \\\n  -F "contrast=true" \\\n  -F "layout_analysis=true"`,
      response: `{\n  "status": "success",\n  "request_id": "fe3a2457-978c-4b40-ab6d-f03a72a6a313",\n  "text": "به نام خداوند جان و خرد",\n  "language": "PERSIAN",\n  "model": { "id": "persian-arabic-crnn-v1", "version": "1.0.0" },\n  "processing_time_ms": 28.4,\n  "confidence": 0.942,\n  "lines": [\n    {\n      "line_id": 1,\n      "text": "به نام خداوند جان و خرد",\n      "confidence": 0.942,\n      "bbox": { "x": 0, "y": 0, "w": 500, "h": 64 },\n      "reading_order": 1\n    }\n  ]\n}`,
    },
    {
      title: 'Recognize Internal Image Reference (NestJS Contract)',
      method: 'POST',
      path: '/v1/recognize',
      curl: `curl -X POST http://localhost:8001/v1/recognize \\\n  -F "image_ref=sample_persian_arabic/images/ms_golestan_001_l1.png" \\\n  -F "language=PERSIAN" \\\n  -F "model_id=persian-arabic-crnn-v1"`,
      response: `{\n  "status": "success",\n  "request_id": "a9108c12-32b1-4091-a12b-312019481231",\n  "text": "به نام خداوند جان و خرد",\n  "language": "PERSIAN",\n  "processing_time_ms": 19.8\n}`,
    },
    {
      title: 'Run CER & WER Evaluation Benchmark',
      method: 'POST',
      path: '/v1/evaluate',
      curl: `curl -X POST http://localhost:8001/v1/evaluate \\\n  -H "Content-Type: application/json" \\\n  -d '{\n    "dataset_manifest": "sample_persian_arabic/test.jsonl",\n    "split": "test",\n    "normalize_text": true\n  }'`,
      response: `{\n  "dataset_manifest": "sample_persian_arabic/test.jsonl",\n  "metrics": {\n    "cer": 0.082,\n    "wer": 0.125,\n    "total_chars": 74,\n    "char_substitutions": 4,\n    "char_deletions": 1,\n    "char_insertions": 1\n  }\n}`,
    },
    {
      title: 'Initiate Asynchronous Model Training',
      method: 'POST',
      path: '/v1/train',
      curl: `curl -X POST http://localhost:8001/v1/train \\\n  -H "Content-Type: application/json" \\\n  -d '{\n    "model_name": "persian-crnn-custom",\n    "train_manifest": "sample_persian_arabic/train.jsonl",\n    "val_manifest": "sample_persian_arabic/val.jsonl",\n    "epochs": 5,\n    "batch_size": 4\n  }'`,
      response: `{\n  "job_id": "job-b46c5129",\n  "status": "RUNNING",\n  "model_name": "persian-crnn-custom",\n  "total_epochs": 5\n}`,
    },
    {
      title: 'Poll Training Job Progress',
      method: 'GET',
      path: '/v1/train/job-b46c5129',
      curl: `curl -s http://localhost:8001/v1/train/job-b46c5129`,
      response: `{\n  "job_id": "job-b46c5129",\n  "status": "COMPLETED",\n  "current_epoch": 5,\n  "train_loss": 3.412,\n  "val_cer": 0.091,\n  "checkpoint_path": "/app/storage/checkpoints/persian-crnn-custom_best.pt"\n}`,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/60 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <Terminal className="w-4 h-4 text-amber-400" />
            HTTP API Specification & NestJS Integration
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Strict microservice contract connecting the NestJS backend with this Python HTR AI Service.
          </p>
        </div>
      </div>

      {/* Architectural Separation Card */}
      <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-4">
        <span className="text-xs font-semibold text-slate-200 uppercase tracking-wider font-mono flex items-center gap-2">
          <Network className="w-4 h-4 text-sky-400" />
          Division of Responsibilities (Microservice Boundaries)
        </span>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          {/* Python Service */}
          <div className="bg-slate-900/80 rounded-lg border border-amber-500/30 p-4 space-y-2">
            <div className="flex items-center gap-2 text-amber-300 font-semibold">
              <span className="w-2 h-2 rounded-full bg-amber-400" />
              Python AI Service Owns:
            </div>
            <ul className="list-disc list-inside text-slate-300 space-y-1 text-[11px] leading-relaxed">
              <li>Image decoding, size validation & EXIF orientation.</li>
              <li>Manuscript preprocessing (deskew, contrast, binarization).</li>
              <li>Text-line valley segmentation & bounding box coordinates.</li>
              <li>PyTorch CRNN deep learning inference & CTC greedy decoding.</li>
              <li>Preservation of logical Unicode RTL order without manual reversal.</li>
              <li>Dataset integrity validation & data leakage prevention.</li>
              <li>Model training loops, checkpoint management, and CER/WER evaluation.</li>
            </ul>
          </div>

          {/* NestJS Backend */}
          <div className="bg-slate-900/80 rounded-lg border border-sky-500/30 p-4 space-y-2">
            <div className="flex items-center gap-2 text-sky-300 font-semibold">
              <span className="w-2 h-2 rounded-full bg-sky-400" />
              NestJS Backend Owns:
            </div>
            <ul className="list-disc list-inside text-slate-300 space-y-1 text-[11px] leading-relaxed">
              <li>User authentication, JWT tokens, RBAC permissions.</li>
              <li>Manuscript metadata (cataloging, authors, calligraphy period).</li>
              <li>Page digital archive records & cloud bucket storage.</li>
              <li>Transcription audit history & human-in-the-loop corrections.</li>
              <li>Full-text search indexing with normalized Persian/Arabic lemmas.</li>
              <li>Public client-facing web application endpoints.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Curl Endpoints Reference */}
      <div className="space-y-4">
        <span className="text-xs font-semibold text-slate-200 uppercase tracking-wider font-mono block">
          Documented Endpoints & Sample Requests:
        </span>

        <div className="space-y-4">
          {curlSnippets.map((item, idx) => (
            <div
              key={idx}
              className="bg-slate-900/90 border border-slate-800 rounded-xl overflow-hidden shadow-sm"
            >
              {/* Header */}
              <div className="bg-slate-950 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span
                    className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold uppercase ${
                      item.method === 'GET'
                        ? 'bg-sky-950 text-sky-400 border border-sky-500/30'
                        : 'bg-emerald-950 text-emerald-400 border border-emerald-500/30'
                    }`}
                  >
                    {item.method}
                  </span>
                  <span className="font-mono text-xs text-slate-200 font-semibold">{item.path}</span>
                  <span className="text-xs text-slate-400 hidden sm:inline">— {item.title}</span>
                </div>
                <button
                  onClick={() => copyToClipboard(item.curl, idx)}
                  className="flex items-center gap-1 px-2.5 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors"
                >
                  {copiedIndex === idx ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      <span>Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>Copy cURL</span>
                    </>
                  )}
                </button>
              </div>

              {/* cURL Command */}
              <div className="p-4 bg-slate-950/70 border-b border-slate-850">
                <pre className="font-mono text-xs text-amber-300 overflow-x-auto whitespace-pre-wrap leading-relaxed select-all">
                  {item.curl}
                </pre>
              </div>

              {/* Sample Response */}
              <div className="p-4 bg-slate-900/40">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider font-mono block mb-1">
                  Expected JSON Response (200 OK):
                </span>
                <pre className="font-mono text-[11px] text-slate-300 overflow-x-auto whitespace-pre-wrap leading-relaxed bg-slate-950/80 p-3 rounded border border-slate-800/80">
                  {item.response}
                </pre>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
