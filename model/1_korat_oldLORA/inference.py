import torch
import librosa
from espnet2.bin.asr_inference import Speech2Text
from espnet2.layers.create_adapter_fn import create_lora_adapter

DEVICE = "cuda"

# 1. PATHS (Same as in your train.py)
CONFIG_PATH = "[path to thai-central]/config.yaml"
MODEL_PATH = "[path to thai-central]/valid.acc.ave_10best.pth"

# Path to your new LoRA file
ADAPTER_FILE = "./exp/finetune/[model].pth" # Check exact filename in your exp dir!
AUDIO_FILE = "[your audio path]"
LORA_TARGET = ["w_1", "w_2", "merge_proj"]

print("1. Loading Base Model...")
# Use manual loading to match train.py logic
speech2text = Speech2Text(
    asr_train_config=CONFIG_PATH,
    asr_model_file=MODEL_PATH,
    device=DEVICE,
    minlenratio=0.0,
    maxlenratio=0.0,
    beam_size=10,
    ctc_weight=0.3,
)

model = speech2text.asr_model

# 2. CRITICAL FIX: Disable Normalization
# (The model was trained without it, so inference must also be without it)
if hasattr(model, "normalize"):
    print("   -> Disabling normalization layer (matching training setup)")
    model.normalize = None

# 3. Attach LoRA Hooks
print("2. Attaching LoRA adapters...")
create_lora_adapter(model, target_modules=LORA_TARGET)

# 4. Load LoRA Weights
print(f"3. Loading weights from {ADAPTER_FILE}...")
try:
    # ESPnet-EZ saves adapters inside a 'lora_state' key usually,
    # but sometimes raw state dict depending on version.
    state_dict = torch.load(ADAPTER_FILE, map_location=DEVICE)
    
    # Handle different save formats
    if "lora_state" in state_dict:
        lora_state = state_dict["lora_state"]
    elif "state_dict" in state_dict:
        lora_state = state_dict["state_dict"]
    else:
        lora_state = state_dict

    # Load with strict=False (we only want to load the matched LoRA layers)
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    
    # Verify meaningful keys were loaded
    loaded_lora_keys = [k for k in lora_state.keys() if "lora" in k]
    if len(loaded_lora_keys) > 0:
        print(f"   -> Successfully loaded {len(loaded_lora_keys)} LoRA layers.")
    else:
        print("   -> WARNING: No LoRA keys found in checkpoint!")

except Exception as e:
    print(f"   -> Error loading adapter: {e}")
    exit()

model.eval()

# 5. Audio Processing
print(f"4. Transcribing {AUDIO_FILE}...")
# Ensure 16k sample rate (standard for this model)
audio, rate = librosa.load(AUDIO_FILE, sr=16000)

# 6. Inference
with torch.no_grad():
    # speech2text call handles tokenization/decoding automatically
    results = speech2text(audio)

# Extract text (results is a list of hypotheses, index 0 is best)
if results:
    text = results[0][0]
    # Remove special tokens if any (usually not needed for this model type)
    print("\n------------------------------------------------")
    print(f"Prediction: {text}")
    print("------------------------------------------------")
else:
    print("No prediction found.")