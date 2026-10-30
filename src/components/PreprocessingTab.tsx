import React, { useState } from 'react';
import { Layers, Sliders, CheckCircle2, AlertTriangle, ArrowRight, Eye, ShieldCheck } from 'lucide-react';

export const PreprocessingTab: React.FC = () => {
  const [activeStage, setActiveStage] = useState<number>(0);

  const stages = [
    {
      id: 'original',
      title: '1. Raw Manuscript Acquisition & EXIF',
      desc: 'Decodes JPEG/PNG/WebP/TIFF byte streams, corrects camera EXIF orientation, and validates size limits (25MB, 100Mpx).',
      principle: 'Historical documents are acquired from flatbed scanners or archival cameras. Orientation tags must be normalized before tensor operations.',
      badge: 'Decoding',
    },
    {
      id: 'grayscale',
      title: '2. Color Space & Luminance Normalization',
      desc: 'Converts RGB color channels into standardized 8-bit single-channel grayscale (Luminance L = 0.299R + 0.587G + 0.114B).',
      principle: 'Historical parchment often exhibits yellowing, water stains, and fading. Grayscale conversion isolates ink carbon pigments from parchment tint.',
      badge: 'Luminance',
    },
    {
      id: 'deskew',
      title: '3. Projection Profile Variance Deskew',
      desc: 'Rotates image across test angles (-10° to +10°) to maximize the variance of horizontal projection profiles.',
      principle: 'Manuscript leaves may have scan slants. Aligning baselines horizontally is critical for elongated 1D time-step recurrent recognition.',
      badge: 'Geometry',
    },
    {
      id: 'contrast',
      title: '4. Calligraphy Ink Contrast Enhancement',
      desc: 'Non-linear contrast stretching and adaptive histogram scaling to emphasize faded iron-gall or lampblack ink.',
      principle: 'Centuries-old Islamic calligraphy features delicate pen strokes and fine vowel marks (harakat) that fade into aged paper.',
      badge: 'Contrast',
    },
    {
      id: 'denoise',
      title: '5. Median Despeckle & Paper Grain Filter',
      desc: 'Applies 3x3 median filter to eliminate salt-and-pepper noise, paper fiber specks, and scanner sensor artifacts.',
      principle: 'Aggressive Gaussian blurring would blend diacritic dots (nuqteh) together, whereas median filtering preserves sharp stroke edges.',
      badge: 'Filtering',
    },
    {
      id: 'binarize',
      title: '6. Otsu Adaptive Binarization',
      desc: 'Calculates bimodal gray-level histogram variance threshold to segment foreground ink from background parchment.',
      principle: 'Optional step: While clean print OCR favors pure black-and-white, manuscript HTR often preserves grayscale gradients to distinguish overlapping ligatures.',
      badge: 'Thresholding',
    },
    {
      id: 'segment',
      title: '7. Text-Line Valley Projection & Slicing',
      desc: 'Calculates row-wise dark ink density to identify inter-line valleys and bounding boxes for text-line CRNN input.',
      principle: 'Historical Persian scripts like Nasta\'liq and Shikasteh have sloping baselines and interlinear overlaps that require valley-guided region extraction.',
      badge: 'Layout',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/60 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <Layers className="w-4 h-4 text-amber-400" />
            Historical Manuscript Preprocessing Pipeline
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Documented image transformation stages adhering to historical preservation discipline (non-destructive, aspect-ratio preserving).
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs px-2.5 py-1 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300 font-mono flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-amber-400" />
            Source Image Immutable
          </span>
        </div>
      </div>

      {/* Interactive Stages Navigator */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
        {/* Left: Stage Selector (4 cols) */}
        <div className="md:col-span-5 space-y-2">
          {stages.map((stage, idx) => {
            const isActive = activeStage === idx;
            return (
              <button
                key={stage.id}
                onClick={() => setActiveStage(idx)}
                className={`w-full text-left p-3 rounded-lg border text-xs transition-all flex items-start justify-between gap-3 ${
                  isActive
                    ? 'bg-amber-950/40 border-amber-500/50 text-amber-200 shadow-sm'
                    : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                }`}
              >
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-slate-200 text-xs">{stage.title}</span>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">{stage.desc}</p>
                </div>
                <span
                  className={`text-[9px] uppercase font-mono px-1.5 py-0.5 rounded shrink-0 ${
                    isActive
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                      : 'bg-slate-800 text-slate-500 border border-slate-700'
                  }`}
                >
                  {stage.badge}
                </span>
              </button>
            );
          })}
        </div>

        {/* Right: Stage Detail & Visualization (7 cols) */}
        <div className="md:col-span-7 space-y-4">
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-amber-300 uppercase tracking-wider font-mono">
                Active Transformation: {stages[activeStage].badge}
              </span>
              <span className="text-xs text-slate-400">Step {activeStage + 1} of {stages.length}</span>
            </div>

            <h3 className="text-lg font-semibold text-slate-100">{stages[activeStage].title}</h3>

            <div className="bg-slate-950/80 rounded-lg border border-slate-800 p-4 space-y-3">
              <span className="text-xs font-medium text-slate-300 uppercase font-mono block">Operation Description:</span>
              <p className="text-xs text-slate-300 leading-relaxed font-sans">{stages[activeStage].desc}</p>
            </div>

            {/* Historical Preservation Guideline */}
            <div className="bg-amber-950/20 rounded-lg border border-amber-500/20 p-4 space-y-2">
              <span className="text-xs font-semibold text-amber-300 flex items-center gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                Manuscript Discipline Rationale:
              </span>
              <p className="text-xs text-amber-200/90 leading-relaxed font-sans">
                {stages[activeStage].principle}
              </p>
            </div>

            {/* Visual Transformation Simulation Canvas */}
            <div className="pt-2">
              <span className="text-xs font-medium text-slate-300 block mb-2">Simulated Visual Output:</span>
              <div className="bg-slate-950 rounded-lg border border-slate-800 p-4 flex flex-col items-center justify-center min-h-[120px]">
                {activeStage === 0 && (
                  <div className="w-full max-w-md h-20 bg-[#e8dcbd] rounded border border-amber-900/30 flex items-center justify-center shadow-inner text-amber-950 font-serif text-lg">
                    [RGB 24-bit Natural Parchment & Faded Sepia Ink]
                  </div>
                )}
                {activeStage === 1 && (
                  <div className="w-full max-w-md h-20 bg-[#c8c5bc] rounded border border-slate-600 flex items-center justify-center shadow-inner text-slate-900 font-serif text-lg">
                    [Luminance Grayscale: L = 0.299R + 0.587G + 0.114B]
                  </div>
                )}
                {activeStage === 2 && (
                  <div className="w-full max-w-md h-20 bg-[#c8c5bc] rounded border border-slate-600 flex items-center justify-center shadow-inner text-slate-900 font-serif text-lg transform rotate-0">
                    [Deskewed Baseline: Optimal Rotation θ = -2.4° applied]
                  </div>
                )}
                {activeStage === 3 && (
                  <div className="w-full max-w-md h-20 bg-[#dedad0] rounded border border-slate-600 flex items-center justify-center shadow-inner text-black font-serif text-lg font-bold">
                    [Contrast Enhanced: Ink Peaks Scaled (Factor 1.3)]
                  </div>
                )}
                {activeStage === 4 && (
                  <div className="w-full max-w-md h-20 bg-[#dedad0] rounded border border-slate-600 flex items-center justify-center shadow-inner text-black font-serif text-lg">
                    [3x3 Median Filtered: Paper Grain Smoothed, Harakat Preserved]
                  </div>
                )}
                {activeStage === 5 && (
                  <div className="w-full max-w-md h-20 bg-white rounded border border-slate-400 flex items-center justify-center shadow-inner text-black font-serif text-lg font-mono">
                    [Binary Otsu: Threshold T=118, Foreground Ink Segments]
                  </div>
                )}
                {activeStage === 6 && (
                  <div className="w-full max-w-md h-20 bg-slate-900 rounded border border-emerald-500/50 flex flex-col items-center justify-center p-2 text-emerald-400 font-mono text-xs">
                    <span>Line 1: [y=0, h=32, w=500] ━━━━ Valley: 0.05 ━━━━</span>
                    <span className="mt-1">Line 2: [y=32, h=32, w=500] ━━━━ Baseline Segmented ━━━━</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
