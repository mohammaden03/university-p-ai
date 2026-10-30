import React, { useState } from 'react';
import { Upload, Play, Copy, Check, Info, Sliders, Clock, AlertCircle, FileText, CheckCircle2 } from 'lucide-react';
import { Language, ModelInfo, RecognitionResponse } from '../types/htr';

interface RecognitionTabProps {
  models: ModelInfo[];
  defaultModelId: string;
}

const SAMPLE_MANUSCRIPTS = [
  {
    id: 'sample_golestan_1',
    name: "Saadi's Golestan (Persian - Nasta'liq)",
    imageRef: 'sample_persian_arabic/images/ms_golestan_001_l1.png',
    lang: 'PERSIAN' as Language,
    description: "Classical verse: 'به نام خداوند جان و خرد'",
  },
  {
    id: 'sample_golestan_2',
    name: "Saadi's Golestan Prose (Persian - Naskh)",
    imageRef: 'sample_persian_arabic/images/ms_golestan_002_l1.png',
    lang: 'PERSIAN' as Language,
    description: "Classical prose: 'منت خدای را عز و جل...'",
  },
  {
    id: 'sample_fatihah_1',
    name: 'Surah Al-Fatihah (Arabic - Thuluth Calligraphy)',
    imageRef: 'sample_persian_arabic/images/ms_arabic_001_l1.png',
    lang: 'ARABIC' as Language,
    description: "Basmalah: 'بسم الله الرحمن الرحيم'",
  },
  {
    id: 'sample_hafez_1',
    name: 'Hafez Ghazal 1 (Persian & Arabic Mixed)',
    imageRef: 'sample_persian_arabic/images/ms_hafez_001_l1.png',
    lang: 'MIXED' as Language,
    description: "Hemistich: 'الا یا ایها الساقی ادر کأسا و ناولها'",
  },
];

export const RecognitionTab: React.FC<RecognitionTabProps> = ({ models, defaultModelId }) => {
  const [selectedSample, setSelectedSample] = useState<string>(SAMPLE_MANUSCRIPTS[0].id);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [language, setLanguage] = useState<Language>('PERSIAN');
  const [modelId, setModelId] = useState<string>(defaultModelId || 'persian-arabic-crnn-v1');

  // Preprocessing toggles
  const [deskew, setDeskew] = useState(false);
  const [contrast, setContrast] = useState(true);
  const [denoise, setDenoise] = useState(false);
  const [binarize, setBinarize] = useState(false);
  const [layoutAnalysis, setLayoutAnalysis] = useState(true);

  // Execution state
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RecognitionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [activeLineHover, setActiveLineHover] = useState<number | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setUploadedFile(file);
      setSelectedSample('');
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleSelectSample = (sampleId: string) => {
    setSelectedSample(sampleId);
    setUploadedFile(null);
    setPreviewUrl(null);
    const s = SAMPLE_MANUSCRIPTS.find((item) => item.id === sampleId);
    if (s) {
      setLanguage(s.lang);
    }
    setResult(null);
    setError(null);
  };

  const runRecognition = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('language', language);
    formData.append('model_id', modelId);
    formData.append('deskew', deskew ? 'true' : 'false');
    formData.append('contrast', contrast ? 'true' : 'false');
    formData.append('denoise', denoise ? 'true' : 'false');
    formData.append('binarize', binarize ? 'true' : 'false');
    formData.append('layout_analysis', layoutAnalysis ? 'true' : 'false');

    if (uploadedFile) {
      formData.append('file', uploadedFile);
    } else {
      const s = SAMPLE_MANUSCRIPTS.find((item) => item.id === selectedSample);
      if (s) {
        formData.append('image_ref', s.imageRef);
      }
    }

    try {
      const resp = await fetch('/v1/recognize', {
        method: 'POST',
        body: formData,
      });

      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.message || data.detail?.message || `HTTP ${resp.status}`);
      }

      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Recognition request failed.');
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = () => {
    if (result?.text) {
      navigator.clipboard.writeText(result.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner & Instructions */}
      <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/60 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <FileText className="w-4 h-4 text-amber-400" />
            Handwritten Manuscript Recognition (HTR Inference)
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Process Persian & Arabic manuscripts through the active deep learning model adapter with real-time text-line segmentation.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">Target Model:</span>
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="bg-slate-900 border border-slate-700 text-xs text-amber-300 rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-amber-500 font-mono"
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.id} ({m.status})
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Input Selection & Preprocessing Controls (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          {/* Sample Manuscripts Carousel */}
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-4 space-y-3">
            <label className="text-xs font-medium text-slate-300 block uppercase tracking-wider">
              1. Select Sample Manuscript or Upload
            </label>

            <div className="grid grid-cols-1 gap-2">
              {SAMPLE_MANUSCRIPTS.map((s) => (
                <button
                  key={s.id}
                  onClick={() => handleSelectSample(s.id)}
                  className={`text-left p-3 rounded-lg border text-xs transition-all flex flex-col justify-between ${
                    selectedSample === s.id && !uploadedFile
                      ? 'bg-amber-950/30 border-amber-500/50 text-amber-100 shadow-sm'
                      : 'bg-slate-900/60 border-slate-800 text-slate-300 hover:bg-slate-800/40 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-slate-100">{s.name}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-amber-400 border border-amber-500/20 font-mono">
                      {s.lang}
                    </span>
                  </div>
                  <span className="text-[11px] text-slate-400 mt-1" dir="rtl">
                    {s.description}
                  </span>
                </button>
              ))}
            </div>

            {/* Custom Upload Area */}
            <div className="pt-2 border-t border-slate-700/50">
              <label
                className={`border-2 border-dashed rounded-lg p-3 text-center cursor-pointer block transition-colors ${
                  uploadedFile
                    ? 'border-emerald-500/50 bg-emerald-950/20 text-emerald-200'
                    : 'border-slate-700 hover:border-slate-500 bg-slate-900/30 text-slate-400'
                }`}
              >
                <Upload className="w-5 h-5 mx-auto mb-1 text-slate-400" />
                <span className="text-xs font-medium block">
                  {uploadedFile ? uploadedFile.name : 'Upload Custom Manuscript Image'}
                </span>
                <span className="text-[10px] text-slate-500 block">
                  PNG, JPEG, WebP, or TIFF (up to 25MB)
                </span>
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/webp,image/tiff"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </label>
            </div>
          </div>

          {/* Configuration & Preprocessing Options */}
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-4 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-amber-400" />
                2. Recognition Options
              </span>
              <span className="text-[10px] text-slate-400 font-mono">API Form Fields</span>
            </div>

            {/* Language Selector */}
            <div>
              <label className="text-xs text-slate-400 block mb-1.5">Script / Language Target:</label>
              <div className="grid grid-cols-3 gap-2">
                {(['PERSIAN', 'ARABIC', 'MIXED'] as Language[]).map((l) => (
                  <button
                    key={l}
                    onClick={() => setLanguage(l)}
                    className={`py-1.5 text-xs rounded-md font-medium border transition-colors ${
                      language === l
                        ? 'bg-amber-600 text-white border-amber-500 shadow-sm'
                        : 'bg-slate-900/70 border-slate-700 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    {l}
                  </button>
                ))}
              </div>
            </div>

            {/* Preprocessing Toggles */}
            <div className="space-y-2 pt-2 border-t border-slate-700/50">
              <span className="text-[11px] font-medium text-slate-400 block">Pipeline Filters:</span>

              <label className="flex items-center justify-between p-2 rounded-lg bg-slate-900/50 border border-slate-800 cursor-pointer text-xs">
                <span className="text-slate-300">Calligraphy Contrast Enhancement</span>
                <input
                  type="checkbox"
                  checked={contrast}
                  onChange={(e) => setContrast(e.target.checked)}
                  className="rounded border-slate-700 text-amber-600 focus:ring-0"
                />
              </label>

              <label className="flex items-center justify-between p-2 rounded-lg bg-slate-900/50 border border-slate-800 cursor-pointer text-xs">
                <span className="text-slate-300">Projection Profile Deskew</span>
                <input
                  type="checkbox"
                  checked={deskew}
                  onChange={(e) => setDeskew(e.target.checked)}
                  className="rounded border-slate-700 text-amber-600 focus:ring-0"
                />
              </label>

              <label className="flex items-center justify-between p-2 rounded-lg bg-slate-900/50 border border-slate-800 cursor-pointer text-xs">
                <span className="text-slate-300">Median Despeckle / Denoising</span>
                <input
                  type="checkbox"
                  checked={denoise}
                  onChange={(e) => setDenoise(e.target.checked)}
                  className="rounded border-slate-700 text-amber-600 focus:ring-0"
                />
              </label>

              <label className="flex items-center justify-between p-2 rounded-lg bg-slate-900/50 border border-slate-800 cursor-pointer text-xs">
                <span className="text-slate-300">Otsu Binarization (Black & White)</span>
                <input
                  type="checkbox"
                  checked={binarize}
                  onChange={(e) => setBinarize(e.target.checked)}
                  className="rounded border-slate-700 text-amber-600 focus:ring-0"
                />
              </label>

              <label className="flex items-center justify-between p-2 rounded-lg bg-slate-900/50 border border-slate-800 cursor-pointer text-xs">
                <span className="text-slate-300">Document Layout & Line Extraction</span>
                <input
                  type="checkbox"
                  checked={layoutAnalysis}
                  onChange={(e) => setLayoutAnalysis(e.target.checked)}
                  className="rounded border-slate-700 text-amber-600 focus:ring-0"
                />
              </label>
            </div>

            {/* Run Button */}
            <button
              onClick={runRecognition}
              disabled={loading}
              className="w-full py-2.5 px-4 bg-amber-600 hover:bg-amber-500 disabled:bg-slate-700 text-white text-sm font-semibold rounded-lg shadow-md transition-all flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Running HTR Inference...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  <span>Execute Recognition Pipeline</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: Visualization, Recognized Text & Lines (7 cols) */}
        <div className="lg:col-span-7 space-y-5">
          {/* Error Banner */}
          {error && (
            <div className="bg-rose-950/40 border border-rose-500/40 text-rose-200 rounded-xl p-4 text-xs flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-rose-300">Inference Error</p>
                <p className="mt-1 font-mono text-[11px]">{error}</p>
              </div>
            </div>
          )}

          {/* Manuscript Preview & Bounding Box Viewer */}
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-300 uppercase tracking-wider">
                Manuscript Image & Layout Bounding Boxes
              </span>
              <span className="text-[11px] text-slate-400 font-mono">
                {result?.lines ? `${result.lines.length} segmented line(s)` : 'Awaiting recognition'}
              </span>
            </div>

            <div className="bg-slate-950/80 rounded-lg border border-slate-800 p-3 flex items-center justify-center min-h-[140px] relative overflow-hidden">
              {/* Synthetic/Sample line display or uploaded image */}
              <div className="relative inline-block max-w-full">
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Uploaded Manuscript"
                    className="max-h-64 rounded shadow-md object-contain border border-slate-800"
                  />
                ) : (
                  <div className="w-full max-w-lg h-24 bg-[#f2ebd7] rounded border border-amber-900/30 flex items-center justify-center shadow-inner relative p-4 text-center">
                    <span
                      className="text-2xl text-slate-900 font-serif select-none"
                      dir="rtl"
                      style={{ fontFamily: 'Tahoma, Arial, sans-serif' }}
                    >
                      {SAMPLE_MANUSCRIPTS.find((s) => s.id === selectedSample)?.description || 'نسخه خطی'}
                    </span>
                    <div className="absolute inset-x-4 top-1/2 h-0.5 bg-amber-900/20 -translate-y-1/2 pointer-events-none" />
                  </div>
                )}

                {/* Overlaid bounding box highlights */}
                {result?.lines &&
                  result.lines.map((line) => {
                    if (!line.bbox) return null;
                    const isHovered = activeLineHover === line.line_id;
                    return (
                      <div
                        key={line.line_id}
                        onMouseEnter={() => setActiveLineHover(line.line_id)}
                        onMouseLeave={() => setActiveLineHover(null)}
                        className={`absolute border transition-colors cursor-pointer ${
                          isHovered
                            ? 'border-amber-400 bg-amber-400/20 z-20'
                            : 'border-emerald-500/70 bg-emerald-500/10 z-10'
                        }`}
                        style={{
                          left: `${line.bbox.x}px`,
                          top: `${line.bbox.y}px`,
                          width: `${line.bbox.w}px`,
                          height: `${line.bbox.h}px`,
                        }}
                        title={`Line ${line.line_id} (${line.bbox.w}x${line.bbox.h})`}
                      />
                    );
                  })}
              </div>
            </div>
          </div>

          {/* Recognition Output Card */}
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-4 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-slate-200 uppercase tracking-wider">
                  Recognized Text (Logical RTL Unicode)
                </span>
                {result && (
                  <span className="text-[11px] px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-500/30 flex items-center gap-1 font-mono">
                    <CheckCircle2 className="w-3 h-3" />
                    Success
                  </span>
                )}
              </div>
              {result?.text && (
                <button
                  onClick={copyToClipboard}
                  className="flex items-center gap-1 px-2.5 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copied ? 'Copied!' : 'Copy Text'}</span>
                </button>
              )}
            </div>

            {/* Transcribed Text Display */}
            <div className="bg-slate-950 rounded-lg border border-slate-800 p-4 min-h-[100px] flex items-center justify-center">
              {result?.text ? (
                <div className="w-full text-right" dir="rtl">
                  <p className="text-xl sm:text-2xl text-amber-200 font-serif leading-relaxed whitespace-pre-wrap select-all">
                    {result.text}
                  </p>
                </div>
              ) : (
                <div className="text-center text-slate-500 text-xs">
                  <FileText className="w-6 h-6 mx-auto mb-2 opacity-40" />
                  <span>Execute recognition above to transcribe manuscript text in logical Unicode order.</span>
                </div>
              )}
            </div>

            {/* Metadata Bar */}
            {result && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
                <div className="bg-slate-900/60 border border-slate-800 p-2 rounded-lg">
                  <span className="text-slate-400 block">Latency:</span>
                  <span className="font-mono text-amber-300 font-medium flex items-center gap-1">
                    <Clock className="w-3 h-3 text-amber-400" />
                    {result.processing_time_ms} ms
                  </span>
                </div>
                <div className="bg-slate-900/60 border border-slate-800 p-2 rounded-lg">
                  <span className="text-slate-400 block">Confidence:</span>
                  <span className="font-mono text-emerald-300 font-medium">
                    {result.confidence !== null ? `${(result.confidence * 100).toFixed(1)}%` : 'Null (Untrained)'}
                  </span>
                </div>
                <div className="bg-slate-900/60 border border-slate-800 p-2 rounded-lg">
                  <span className="text-slate-400 block">Executing Model:</span>
                  <span className="font-mono text-sky-300 truncate block" title={result.model.id}>
                    {result.model.id}
                  </span>
                </div>
                <div className="bg-slate-900/60 border border-slate-800 p-2 rounded-lg">
                  <span className="text-slate-400 block">Lines Detected:</span>
                  <span className="font-mono text-slate-200 font-medium">{result.lines.length}</span>
                </div>
              </div>
            )}

            {/* Line-Level Table */}
            {result?.lines && result.lines.length > 0 && (
              <div className="space-y-2 pt-2">
                <span className="text-xs font-medium text-slate-300 block">Line-Level Spatial Metadata:</span>
                <div className="overflow-x-auto rounded-lg border border-slate-800">
                  <table className="w-full text-left text-xs text-slate-300">
                    <thead className="bg-slate-900 text-slate-400 text-[10px] uppercase font-mono">
                      <tr>
                        <th className="p-2">Line #</th>
                        <th className="p-2">Reading Order</th>
                        <th className="p-2">Bounding Box (x,y,w,h)</th>
                        <th className="p-2">Confidence</th>
                        <th className="p-2 text-right">Recognized Text</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
                      {result.lines.map((l) => (
                        <tr
                          key={l.line_id}
                          onMouseEnter={() => setActiveLineHover(l.line_id)}
                          onMouseLeave={() => setActiveLineHover(null)}
                          className={`hover:bg-slate-800/40 transition-colors ${
                            activeLineHover === l.line_id ? 'bg-amber-950/30' : ''
                          }`}
                        >
                          <td className="p-2 font-mono text-amber-400">{l.line_id}</td>
                          <td className="p-2 font-mono text-slate-400">{l.reading_order}</td>
                          <td className="p-2 font-mono text-slate-400 text-[11px]">
                            {l.bbox ? `[${l.bbox.x}, ${l.bbox.y}, ${l.bbox.w}, ${l.bbox.h}]` : 'N/A'}
                          </td>
                          <td className="p-2 font-mono">
                            {l.confidence !== null ? (
                              <span className="text-emerald-400">{(l.confidence * 100).toFixed(1)}%</span>
                            ) : (
                              <span className="text-slate-500">null</span>
                            )}
                          </td>
                          <td className="p-2 text-right font-serif text-slate-100" dir="rtl">
                            {l.text}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
