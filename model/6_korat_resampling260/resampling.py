import csv
import random
from tqdm import tqdm

# Input File (Created by your 'rank_by_text.py' or 'smart_curriculum' script)
INPUT_CSV = "train_smart_curriculum.csv"

# Output Files
BALANCED_CSV = "./transcription/train_balanced.csv"

def create_datasets():
    print(f"Reading {INPUT_CSV}...")
    try:
        with open(INPUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {INPUT_CSV}. Did you run the ranking script?")
        exit()

    # Separate by Stage
    stage1 = [row for row in data if row["stage"] == "1"]
    stage2 = [row for row in data if row["stage"] == "2"]
    stage3 = [row for row in data if row["stage"] == "3"]

    print(f"Stats:")
    print(f"   Stage 1 (Easy/ECOM): {len(stage1)}")
    print(f"   Stage 2 (Medium):    {len(stage2)}")
    print(f"   Stage 3 (Hard):      {len(stage3)}")

    # --- CREATE BALANCE ---
   # 1. Balance Stage 1: Downsample ECOM to match SURV count (~6.6k)
    s1_ecom = [r for r in stage1 if r.get("dialect_type") == "ECOM"]
    s1_surv = [r for r in stage1 if r.get("dialect_type") != "ECOM"]
    random.shuffle(s1_ecom) 
    s1_balanced = s1_surv + s1_ecom[:len(s1_surv)] # Result: ~13k mixed samples

    # 2. Keep Stage 2 (It is already ~10k, which is perfect)
    s2_balanced = stage2 

    # 3. Upsample Stage 3 (Multiply by 20x to reach ~10k)
    s3_balanced = stage3 * 20
    
    balanced_data = s1_balanced + s2_balanced + s3_balanced
    random.shuffle(balanced_data) # Shuffle so Hard files aren't all at the end

    print(f"\nCreating {BALANCED_CSV}...")
    with open(BALANCED_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(balanced_data)
        
    print(f"✅ Balanced saved ({len(balanced_data)} samples).")
    print(f"   - Stage 3 is now {len(s3_balanced)/len(balanced_data):.1%} of the data (visible!)")

if __name__ == "__main__":
    create_datasets()