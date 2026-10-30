export type Language = 'PERSIAN' | 'ARABIC' | 'MIXED';

export type ModelStatus = 'MODEL_READY' | 'MODEL_NOT_CONFIGURED' | 'MODEL_LOADING' | 'MODEL_LOAD_FAILED';

export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface RecognizedLine {
  line_id: number;
  text: string;
  confidence: number | null;
  bbox: BoundingBox | null;
  reading_order: number;
}

export interface RecognitionResponse {
  status: string;
  request_id: string;
  text: string;
  language: Language;
  model: {
    id: string;
    version: string;
  };
  processing_time_ms: number;
  confidence: number | null;
  lines: RecognizedLine[];
  warnings: string[];
}

export interface ModelInfo {
  id: string;
  name: string;
  version: string;
  supported_languages: string[];
  engine_type: string;
  status: ModelStatus;
  capabilities: string[];
  source?: string;
  license?: string;
  checkpoint_path?: string | null;
  known_limitations?: string;
  evaluation_status?: string;
}

export interface ModelsListResponse {
  default_model_id: string;
  models: ModelInfo[];
}

export interface HealthResponse {
  status: string;
  version: string;
  ready: boolean;
  timestamp: string;
}

export interface ReadinessResponse {
  ready: boolean;
  service_status: string;
  default_model_status: string;
  dependencies: {
    pytorch: {
      version: string;
      cuda_available: boolean;
      device: string;
    };
    storage: {
      model_dir_exists: boolean;
      dataset_root_exists: boolean;
      checkpoint_dir_exists: boolean;
    };
    default_model: {
      id: string;
      status: string;
      ready: boolean;
    };
  };
  timestamp: string;
}

export interface EvaluationMetricSummary {
  cer: number;
  wer: number;
  total_samples: number;
  evaluated_samples: number;
  skipped_samples: number;
  total_chars: number;
  char_substitutions: number;
  char_deletions: number;
  char_insertions: number;
  total_words: number;
  word_errors: number;
}

export interface SamplePredictionResult {
  sample_id: string;
  ground_truth: string;
  prediction: string;
  cer: number;
  wer: number;
}

export interface EvaluationResponse {
  dataset_manifest: string;
  model_id: string;
  model_version: string;
  normalized: boolean;
  split: string;
  metrics: EvaluationMetricSummary;
  per_language_results: Record<string, { cer: number; wer: number; samples: number; total_chars: number }>;
  sample_results: SamplePredictionResult[];
  created_at: string;
}

export interface TrainingJobStatus {
  job_id: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  model_name: string;
  current_epoch: number;
  total_epochs: number;
  train_loss: number | null;
  val_loss: number | null;
  val_cer: number | null;
  checkpoint_path: string | null;
  error_message: string | null;
  started_at: string;
  finished_at: string | null;
  metrics_history: Array<{
    epoch: number;
    train_loss: number;
    val_loss: number;
    val_cer: number;
    checkpoint_path: string;
  }>;
}

export interface DatasetValidationResult {
  valid: boolean;
  manifest_path: string;
  total_samples: number;
  split_counts: Record<string, number>;
  total_words: number;
  vocabulary_size: number;
  unrecognized_chars: string[];
  warnings: string[];
}
