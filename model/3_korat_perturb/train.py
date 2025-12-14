import os
import csv

import numpy as np
import librosa

import torch
from espnet2.bin.asr_inference import Speech2Text
from espnet2.layers.create_adapter_fn import create_lora_adapter
import espnetez as ez

# ---------------- CONFIGURATION ----------------
CSV_DIR = f"./transcription"
EXP_DIR = f"./exp/finetune"
STATS_DIR = f"./exp/stats_finetune"

LORA_TARGET = [
    "fc1", "fc2",          # Encoder Feed-Forward
    "w_1", "w_2",          # Decoder Feed-Forward
    "q_proj", "v_proj",    # Encoder Attention (Optional but recommended)
    "ctc_lo"               # CTC Classification Head
]

# Internal paths
config_path = "[path to thai-central]/config.yaml"
model_path = "[path to thai-central]/valid.acc.ave_10best.pth"

# ---------------- 1. GLOBAL MODEL LOADING ----------------
print("Loading model globally (ONCE) to save RAM...")
# Load once and keep it in memory. Do not delete!
global_speech2text = Speech2Text(
    asr_train_config=config_path,
    asr_model_file=model_path,
    device="cuda",
    minlenratio=0.0,
    maxlenratio=0.0,
)

# Extract components needed for config
pretrain_config = vars(global_speech2text.asr_train_args)
tokenizer = global_speech2text.tokenizer
converter = global_speech2text.converter

# ---------------- 2. CONFIG UPDATE ----------------
finetune_config = ez.config.update_finetune_config(
    'asr',
    pretrain_config,
    f"finetune_with_lora.yaml"
)

# FORCE these settings in Python to ensure YAML doesn't override them
finetune_config["normalize"] = None
finetune_config["preprocessor_conf"]["normalize"] = None
finetune_config["batch_size"] = 1
finetune_config["batch_type"] = "sorted" # Use sorted to force batch_size=1 strictly
finetune_config["chunk_max_abs_length"] = 20000
finetune_config["max_input_length"] = 16000 * 15 # Skip files longer than 15s

# ---------------- 3. MODEL BUILDER (FIXED) ----------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model_fn(args):
    # USE THE GLOBAL MODEL (Fixes Double Loading)
    print("Using global model instance...")
    model = global_speech2text.asr_model
    
    # 🔥 CRITICAL FIX: Kill the normalization layer 🔥
    # This prevents the 7000GB memory allocation
    if hasattr(model, "normalize"):
        print("DISABLE: Found normalization layer, setting to None.")
        model.normalize = None
    
    model.train()
    print(f'Trainable parameters (Full): {count_parameters(model)}')
    
    create_lora_adapter(model, target_modules=LORA_TARGET)
    
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False

    print(f'Trainable parameters (LoRA): {count_parameters(model)}')
    return model

# ---------------- 4. DATASET ----------------
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._parse_single_data(self.data[idx])

    def _parse_single_data(self, d):
        text = d['sentence']
        return {
            "audio_path": d["audio"],
            "text": text,
            "speed": d.get("speed", 1.0),
        }

TRAIN_CSV_FILE = os.path.join(CSV_DIR, "train.csv")
VALID_CSV_FILE = os.path.join(CSV_DIR, "dev.csv")

train_data_list = []
valid_data_list = []

print(f"Loading training data from: {TRAIN_CSV_FILE}")
with open(TRAIN_CSV_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # ORIGINAL (1.0x)
        row["speed"] = 1.0
        train_data_list.append(row)
        
        # SLOW (0.9x) - Create a copy with slow speed
        row_slow = row.copy()
        row_slow["speed"] = 0.9
        train_data_list.append(row_slow)
        
        # FAST (1.1x) - Create a copy with fast speed
        row_fast = row.copy()
        row_fast["speed"] = 1.1
        train_data_list.append(row_fast)

print(f"Loading validation data from: {VALID_CSV_FILE}")
with open(VALID_CSV_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row["speed"] = 1.0
        valid_data_list.append(row)

train_dataset = CustomDataset(train_data_list)
valid_dataset = CustomDataset(valid_data_list)

def tokenize(text):
    return np.array(converter.tokens2ids(tokenizer.text2tokens(text)))

data_info = {
    "speech": lambda d: librosa.load(d["audio_path"], sr=16000)[0]
              if d["speed"] == 1.0 
              # If speed != 1.0, we resample. 
              # Logic: "Fast" speed means fewer samples (shorter duration).
              else librosa.resample(
                  librosa.load(d["audio_path"], sr=16000)[0], 
                  orig_sr=16000, 
                  target_sr=int(16000 / d["speed"])
              ),
    "text": lambda d: tokenize(d["text"]),
}

train_dataset = ez.dataset.ESPnetEZDataset(train_dataset, data_info=data_info)
valid_dataset = ez.dataset.ESPnetEZDataset(valid_dataset, data_info=data_info)

# ---------------- 5. TRAINER ----------------
trainer = ez.Trainer(
    task='asr',
    train_config=finetune_config,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    build_model_fn=build_model_fn,
    data_info=data_info,
    output_dir=EXP_DIR,
    stats_dir=STATS_DIR,
    ngpu=1
)

print("Starting training...")
trainer.collect_stats()
trainer.save_adapter_only = True
#trainer.resume = True
#trainer.resume_path = "./exp/finetune/checkpoint.pth"
trainer.train()