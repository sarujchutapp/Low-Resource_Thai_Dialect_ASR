import csv
import random
from tqdm import tqdm

# Input File (Created by your 'rank_by_text.py' or 'smart_curriculum' script)
INPUT_CSV = "train_smart_curriculum.csv"

# Output Files
PHASE1_CSV = "./transcription/train_phase1.csv"
PHASE2_CSV = "./transcription/train_phase2.csv"

def create_datasets():
    print(f"Reading {INPUT_CSV}...")
    try:
        with open(INPUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {INPUT_CSV}. Did you run the ranking script?")
        exit()

    # 1. Separate by Stage
    stage1 = [row for row in data if row["stage"] == "1"]
    stage2 = [row for row in data if row["stage"] == "2"]
    stage3 = [row for row in data if row["stage"] == "3"]

    print(f"Stats:")
    print(f"   Stage 1 (Easy/ECOM): {len(stage1)}")
    print(f"   Stage 2 (Medium):    {len(stage2)}")
    print(f"   Stage 3 (Hard):      {len(stage3)}")

    # --- CREATE PHASE 1 (Easy Only) ---
    print(f"\nCreating {PHASE1_CSV}...")
    with open(PHASE1_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(stage1)
    print(f"✅ Phase 1 saved ({len(stage1)} samples).")

    # --- CREATE PHASE 2 (Everything + Balancing) ---
   # 1. Balance Stage 1: Downsample ECOM to match SURV count (~6.6k)
    s1_ecom = [r for r in stage1 if r.get("dialect_type") == "ECOM"]
    s1_surv = [r for r in stage1 if r.get("dialect_type") != "ECOM"]
    random.shuffle(s1_ecom) 
    s1_balanced = s1_surv + s1_ecom[:len(s1_surv)] # Result: ~13k mixed samples

    # 2. Keep Stage 2 (It is already ~10k, which is perfect)
    s2_balanced = stage2 

    # 3. Upsample Stage 3 (Multiply by 20x to reach ~10k)
    s3_balanced = stage3 * 20
    
    phase2_data = s1_balanced + s2_balanced + s3_balanced
    random.shuffle(phase2_data) # Shuffle so Hard files aren't all at the end

    print(f"\nCreating {PHASE2_CSV}...")
    with open(PHASE2_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(phase2_data)
        
    print(f"✅ Phase 2 saved ({len(phase2_data)} samples).")
    print(f"   - Stage 3 is now {len(s3_balanced)/len(phase2_data):.1%} of the data (visible!)")

if __name__ == "__main__":
    create_datasets()