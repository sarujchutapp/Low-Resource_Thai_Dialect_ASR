import os
import csv
import torch
import librosa
import numpy as np
import jiwer
from pythainlp import Tokenizer
from espnet2.bin.asr_inference import Speech2Text
from espnet2.layers.create_adapter_fn import create_lora_adapter
from tqdm import tqdm

# ================= CONFIGURATION =================
DEVICE = "cuda"

# Files
CSV_FILE = "./transcription/dev.csv"       
VOCAB_FILE = "word_korat.txt"              
ADAPTER_FILE = "./exp/finetune/[model].pth" 
AUDIO_BASE_PATH = "./" 

# Model Paths
CONFIG_PATH = "[path to thai-central]/config.yaml"
MODEL_PATH = "[path to thai-central]/valid.acc.ave_10best.pth"

# LoRA Targets
LORA_TARGET = ["w_1", "w_2", "merge_proj"]

# ================= 1. SETUP TOKENIZER =================
def get_tokenizer(vocab_path):
    print(f"Loading custom tokenizer from {vocab_path}...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f.readlines() if line.strip()]
    return Tokenizer(set(vocab), engine='newmm')

# ================= 2. MODEL LOADING =================
def load_model():
    print("Loading Base Model...")
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

    if hasattr(model, "normalize"):
        model.normalize = None

    print("-> Attaching LoRA adapters...")
    create_lora_adapter(model, target_modules=LORA_TARGET)

    print(f"-> Loading weights from {ADAPTER_FILE}...")
    state_dict = torch.load(ADAPTER_FILE, map_location=DEVICE)
    if "lora_state" in state_dict: lora_state = state_dict["lora_state"]
    elif "state_dict" in state_dict: lora_state = state_dict["state_dict"]
    else: lora_state = state_dict

    model.load_state_dict(lora_state, strict=False)
    model.eval()
    return speech2text

# ================= 3. METRIC CALCULATION =================
def calculate_metrics(refs, hyps, tokenizer):
    if not refs: return 0.0, 0.0, 0.0, 0.0

    tokenized_refs = []
    tokenized_hyps = []

    # Tokenize
    for r, h in zip(refs, hyps):
        # Strip spaces for fair comparison
        r_clean = r.replace(" ", "")
        h_clean = h.replace(" ", "")
        
        tok_r = tokenizer.word_tokenize(r_clean)
        tok_h = tokenizer.word_tokenize(h_clean)
        
        tokenized_refs.append(" ".join(tok_r))
        tokenized_hyps.append(" ".join(tok_h))

    # MICRO Metrics
    micro_wer = jiwer.wer(tokenized_refs, tokenized_hyps)
    micro_cer = jiwer.cer(refs, hyps)

    # MACRO Metrics
    wer_list = []
    cer_list = []
    for tr, th, r, h in zip(tokenized_refs, tokenized_hyps, refs, hyps):
        wer_list.append(jiwer.wer(tr, th))
        cer_list.append(jiwer.cer(r, h))

    macro_wer = np.mean(wer_list)
    macro_cer = np.mean(cer_list)

    return micro_wer, macro_wer, micro_cer, macro_cer

# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    tokenizer = get_tokenizer(VOCAB_FILE)
    speech2text = load_model()
    
    print(f"Reading data from {CSV_FILE}...")
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # Data containers
    results = {
        "ALL":  {"refs": [], "hyps": []},
        "ECOM": {"refs": [], "hyps": []},
        "SURV": {"refs": [], "hyps": []}
    }

    print(f"Starting inference on {len(data)} files...")
    
    with open("benchmark_results.csv", "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["audio", "ground_truth", "prediction", "dialect_type", "cer", "wer"])
        
        for row in tqdm(data):
            audio_path = os.path.join(AUDIO_BASE_PATH, row["audio"])
            ground_truth = row["text"]
            category = row["dialect_type"] # E.g., "ECOM" or "SURV"
            
            try:
                audio, rate = librosa.load(audio_path, sr=16000)
                with torch.no_grad():
                    res = speech2text(audio)
                    prediction = res[0][0] if res else ""
                
                # Add to ALL
                results["ALL"]["refs"].append(ground_truth)
                results["ALL"]["hyps"].append(prediction)

                # Add to specific category
                if category in results:
                    results[category]["refs"].append(ground_truth)
                    results[category]["hyps"].append(prediction)
                else:
                    # Just in case there's a typo in the CSV or a 3rd category
                    if category not in results: results[category] = {"refs": [], "hyps": []}
                    results[category]["refs"].append(ground_truth)
                    results[category]["hyps"].append(prediction)

                # Save row
                writer.writerow([row["audio"], ground_truth, prediction, category])
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

    # --- FINAL REPORT ---
    print("\n" + "="*80)
    print(f"{'CATEGORY':<10} | {'Count':<6} | {'Mic CER':<10} | {'Mac CER':<10} | {'Mic WER':<10} | {'Mac WER':<10}")
    print("-" * 80)
    
    # Ensure ECOM and SURV are printed even if they weren't hardcoded
    for cat in sorted(results.keys()):
        refs = results[cat]["refs"]
        hyps = results[cat]["hyps"]
        count = len(refs)
        
        if count > 0:
            mic_wer, mac_wer, mic_cer, mac_cer = calculate_metrics(refs, hyps, tokenizer)
            print(f"{cat:<10} | {count:<6} | {mic_cer:.2%}    | {mac_cer:.2%}    | {mic_wer:.2%}    | {mac_wer:.2%}")
        
    print("="*80)