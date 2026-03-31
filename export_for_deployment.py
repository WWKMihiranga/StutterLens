import os
import sys
import torch
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.models.neurosymbolic import NeuroSymbolicStutterDetector
from transformers import Wav2Vec2Model


def export_model():
    config = Config()
    config.DEVICE = torch.device("cpu")

    checkpoint_dir = config.CHECKPOINT_DIR

    # 1. Locate the best checkpoint
    stage2_path = os.path.join(checkpoint_dir, "stage2_final.pth")
    stage1_path = os.path.join(checkpoint_dir, "stage1_best.pth")

    if os.path.exists(stage2_path):
        ckpt_path = stage2_path
        print(f"Found Stage 2 checkpoint: {ckpt_path}")
    elif os.path.exists(stage1_path):
        ckpt_path = stage1_path
        print(f"Found Stage 1 checkpoint: {ckpt_path}")
    else:
        print("ERROR: No checkpoint found. Run training first.")
        print(f"  Looked in: {checkpoint_dir}")
        return

    # 2. Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Handle different save formats from the notebook
    if isinstance(checkpoint, dict):
        # Stage 2 format: 'student_state_dict'
        if "student_state_dict" in checkpoint:
            state_dict = checkpoint["student_state_dict"]
            print("  Loaded student_state_dict (Stage 2 format)")
        # Stage 1 format: 'model_state_dict'
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("  Loaded model_state_dict (Stage 1 format)")
        # Raw state dict
        elif any(k.startswith("encoder.") for k in checkpoint.keys()):
            state_dict = checkpoint
            print("  Loaded raw state dict")
        else:
            print(f"  ERROR: Unrecognised checkpoint format. Keys: {list(checkpoint.keys())[:5]}")
            return
    else:
        state_dict = checkpoint
        print("  Loaded as raw tensor dict")

    # 3. Extract calibration info
    calibration = None

    # Stage 2 embeds calibration directly
    if isinstance(checkpoint, dict) and "calibration" in checkpoint:
        cal_raw = checkpoint["calibration"]
        calibration = {
            "temperatures": cal_raw.get("temperatures", {}),
            "thresholds": cal_raw.get("thresholds", {}),
        }
        print(f"  Found calibration in checkpoint")
        print(f"    Temperatures: {calibration['temperatures']}")
        print(f"    Thresholds:   {calibration['thresholds']}")

    # Also check for calibration_info key (alternative format)
    if calibration is None and isinstance(checkpoint, dict) and "calibration_info" in checkpoint:
        calibration = checkpoint["calibration_info"]
        print(f"  Found calibration_info in checkpoint")

    # Try loading from separate JSON file
    if calibration is None:
        cal_path = os.path.join(checkpoint_dir, "calibration_info.json")
        if os.path.exists(cal_path):
            with open(cal_path) as f:
                calibration = json.load(f)
            print(f"  Loaded calibration from {cal_path}")

    if calibration is None:
        print("  WARNING: No calibration found. Model will use fixed 0.5 thresholds.")

    # 4. Load label mappings
    mappings_path = os.path.join(config.DATA_PROCESSED, "label_mappings.json")
    if os.path.exists(mappings_path):
        with open(mappings_path) as f:
            label_mappings = json.load(f)
        print(f"  Loaded label mappings: {label_mappings.get('classes', [])}")
    else:
        label_mappings = {
            "label2idx": {c: i for i, c in enumerate(config.STUTTER_TYPES)},
            "idx2label": {str(i): c for i, c in enumerate(config.STUTTER_TYPES)},
            "classes": config.STUTTER_TYPES,
        }
        print(f"  Using default label mappings: {config.STUTTER_TYPES}")

    # 5. Build model and load weights
    print("\nBuilding model architecture...")
    print(f"  NUM_CLASSES = {config.NUM_CLASSES}")
    print(f"  NUM_RULES = {config.NUM_RULES}")
    print(f"  TEMPORAL_HIDDEN_DIM = {config.TEMPORAL_HIDDEN_DIM}")
    print(f"  RULE_PROJECTION_DIM = {config.RULE_PROJECTION_DIM}")

    wav2vec2 = Wav2Vec2Model.from_pretrained(config.WAV2VEC2_MODEL)
    model = NeuroSymbolicStutterDetector(
        config, wav2vec2_model=wav2vec2, freeze_encoder=True
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # Filter out expected missing keys (encoder layers that were frozen)
        real_missing = [k for k in missing if not k.startswith("encoder.")]
        if real_missing:
            print(f"  WARNING: {len(real_missing)} missing non-encoder keys:")
            for k in real_missing[:5]:
                print(f"    - {k}")
        else:
            print(f"  OK: {len(missing)} missing keys (all encoder — expected)")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys")

    # 6. Verify with a test forward pass
    print("\nVerifying model...")
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, config.MAX_AUDIO_LENGTH)  # 48000 samples
        logits, details = model(dummy, return_details=True)
        print(f"  Input:  (1, {config.MAX_AUDIO_LENGTH})")
        print(f"  Output: logits {tuple(logits.shape)}, "
              f"gates {tuple(details['gate_weights'].shape)}, "
              f"rules {tuple(details['rule_scores'].shape)}")
        expected = (1, 149, config.NUM_CLASSES)
        assert logits.shape == torch.Size(expected), \
            f"Shape mismatch: got {tuple(logits.shape)}, expected {expected}"
        print(f"  Shape check PASSED")

    # 7. Save deployment checkpoint
    deploy_config = {
        "SAMPLE_RATE": config.SAMPLE_RATE,
        "CLIP_DURATION": config.CLIP_DURATION,
        "MAX_AUDIO_LENGTH": config.MAX_AUDIO_LENGTH,
        "HIDDEN_DIM": config.HIDDEN_DIM,
        "TEMPORAL_HIDDEN_DIM": config.TEMPORAL_HIDDEN_DIM,
        "TEMPORAL_NUM_LAYERS": config.TEMPORAL_NUM_LAYERS,
        "NUM_CLASSES": config.NUM_CLASSES,
        "NUM_RULES": config.NUM_RULES,
        "RULE_PROJECTION_DIM": config.RULE_PROJECTION_DIM,
        "GATE_HIDDEN_DIM": config.GATE_HIDDEN_DIM,
        "DROPOUT": config.DROPOUT,
        "STUTTER_TYPES": config.STUTTER_TYPES,
        "WAV2VEC2_MODEL": config.WAV2VEC2_MODEL,
        "UNFREEZE_ENCODER_LAYERS": 0,  # always freeze for deployment
    }

    output_path = os.path.join(checkpoint_dir, "cpu_final_model.pth")

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": deploy_config,
        "label_mappings": label_mappings,
        "calibration_info": calibration,
    }, output_path)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"  EXPORT SUCCESSFUL")
    print(f"{'='*60}")
    print(f"  File:    {output_path}")
    print(f"  Size:    {file_size_mb:.1f} MB")
    print(f"  Classes: {config.STUTTER_TYPES}")
    print(f"  Rules:   {config.NUM_RULES}")
    if calibration:
        print(f"  Calibration: included")
    else:
        print(f"  Calibration: NOT included (will use fixed 0.5 thresholds)")


if __name__ == "__main__":
    export_model()
