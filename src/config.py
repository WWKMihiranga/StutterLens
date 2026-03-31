import os
import json
import torch
from pathlib import Path


class Config:
    # Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
    DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    DATASET_PATH = DATA_RAW

    # Audio
    SAMPLE_RATE = 16_000
    CLIP_DURATION = 3.0
    MAX_AUDIO_LENGTH = int(SAMPLE_RATE * CLIP_DURATION)

    # Classes
    STUTTER_TYPES = [
        "interjection", "prolongation", "word_repetition",
    ]
    NUM_CLASSES = len(STUTTER_TYPES)
    STUTTER_CLASS_MAPPING = {
        "Interjection": "interjection",
        "Prolongation": "prolongation",
        "WordRep": "word_repetition",
    }
    NON_STUTTER_CLASSES = [
        "Music", "NoSpeech", "NoStutteredWords",
        "PoorAudioQuality", "NaturalPause", "DifficultToUnderstand",
    ]
    # Include a fraction of NoStutteredWords as negative examples
    INCLUDE_NEGATIVES = True
    NEGATIVE_RATIO = 0.25  # negatives as fraction of total stutter samples (raised from 0.15)

    # Model
    WAV2VEC2_MODEL = "facebook/wav2vec2-base-960h"
    HIDDEN_DIM = 768
    TEMPORAL_HIDDEN_DIM = 256          # increased from 192 for more capacity
    TEMPORAL_NUM_LAYERS = 2
    NUM_RULES = 3                      # one rule per class
    RULE_PROJECTION_DIM = 64           # increased from 48
    GATE_HIDDEN_DIM = 128              # increased from 96
    DROPOUT = 0.3                      # increased from 0.25 for stronger reg
    MIL_POOLING = "mixed"
    MIL_MIXED_ALPHA = 0.7

    # Encoder fine-tuning: unfreeze last N transformer layers
    UNFREEZE_ENCODER_LAYERS = 2        # fine-tune top 2 layers of Wav2Vec2

    # Stage 0
    STAGE0_NUM_EPOCHS = 20             # increased from 15
    STAGE0_LR = 5e-4
    STAGE0_SYNTHETIC_SAMPLES = 5000    # increased from 3000

    # Stage 1
    BATCH_SIZE = 16                    # increased from 8 for more stable gradients
    LEARNING_RATE = 2e-5               # lower LR since encoder is partially unfrozen
    ENCODER_LR = 5e-6                  # separate LR for encoder layers
    WEIGHT_DECAY = 0.02                # slightly higher
    NUM_EPOCHS_STAGE1 = 25             # full training (set lower for quick testing)
    WARMUP_STEPS = 200                 # increased from 100
    EARLY_STOPPING_PATIENCE = 8        # increased from 6
    GRAD_CLIP_NORM = 1.0
    USE_BALANCED_SAMPLING = True
    USE_AUGMENTATION = True
    FOCAL_GAMMA = 2.0                  # focal loss gamma
    LABEL_SMOOTHING = 0.05             # smooth hard labels
    CLASS_WEIGHT_CAP = 15.0            # raised from 10.0 for stronger minority class boost

    # Stage 2
    NUM_EPOCHS_STAGE2 = 15             # full training (set lower for quick testing)
    STAGE2_LR = 1e-5                   # lower for stability
    EMA_DECAY = 0.999                  # slower EMA for better teacher
    CONFIDENCE_THRESHOLD = 0.5
    MIN_EVENT_LENGTH = 3
    BOUNDARY_LOSS_ALPHA = 1.0
    BOUNDARY_LOSS_BETA = 0.1
    CONSISTENCY_TEMPERATURE = 1.0
    LAMBDA_SUPERVISED = 1.0
    LAMBDA_CONSISTENCY = 0.5           # increased from 0.3
    LAMBDA_WEAK = 0.7                  # increased from 0.5
    PSEUDO_LABEL_PERCENTILE = 65       # raised from 40 for higher quality pseudo-labels
    PSEUDO_LABEL_MIN_THRESH = 0.35     # minimum threshold floor

    # Gate entropy annealing
    GATE_ENTROPY_LAMBDA_START = 0.5    # strong entropy reg at start
    GATE_ENTROPY_LAMBDA_END = 0.15     # anneal down to allow specialisation

    # Event post-processing
    EVENT_MEDIAN_FILTER_SIZE = 5       # smooth frame probs before detection
    EVENT_MERGE_GAP = 2                # merge events within N frames

    # Splits
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Device
    DEVICE = torch.device("cpu")
    SEED = 42

    def __init__(self):
        for d in [self.DATA_RAW, self.DATA_PROCESSED, self.CHECKPOINT_DIR,
                  self.OUTPUT_DIR, self.LOG_DIR,
                  os.path.join(self.OUTPUT_DIR, "visualizations")]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def save(self, path=None):
        if path is None:
            path = os.path.join(self.CHECKPOINT_DIR, "config.json")
        cfg = {}
        for attr in dir(self):
            if attr.startswith("_") or callable(getattr(self, attr)):
                continue
            val = getattr(self, attr)
            if isinstance(val, torch.device):
                val = str(val)
            cfg[attr] = val
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            cfg = json.load(f)
        instance = cls()
        for k, v in cfg.items():
            if k == "DEVICE":
                v = torch.device(v)
            setattr(instance, k, v)
        return instance

    @staticmethod
    def select_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
