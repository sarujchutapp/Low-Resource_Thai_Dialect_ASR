import csv
import random
from tqdm import tqdm

# Input File (Created by your 'rank_by_text.py' or 'smart_curriculum' script)
INPUT_CSV = "train_smart_curriculum.csv"
# Output Files
OUTPUT_CSV = "./transcription/train_balanced.csv"
# Configuration
TARGET_SURV_COUNT = 13000  # Based on your Stage 2 SURV stat
STAGE1_LIMIT = 6000        # Limit Stage 1 to balance total ECOM count

def create_balanced_dataset():
    print(f"Reading {INPUT_CSV}...")
    try:
        with open(INPUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {INPUT_CSV}. Did you run the ranking script?")
        exit()

 # 1. Separate by Stage
    # (Note: Ensure your input CSV was generated with FREQ=290 logic!)
    stage1 = [row for row in data if row["stage"] == "1"]
    stage2 = [row for row in data if row["stage"] == "2"]
    stage3 = [row for row in data if row["stage"] == "3"]

    print(f"\nOriginal Stats (Freq=290):")
    print(f"   Stage 1: {len(stage1)}")
    print(f"   Stage 2: {len(stage2)}")
    print(f"   Stage 3: {len(stage3)}")

    # --- BALANCING LOGIC ---

    # 1. Stage 1: Aggressive Downsample
    # We only need enough Stage 1 ECOMs to complement Stage 2 ECOMs.
    print(f"\nProcessing Stage 1...")
    if len(stage1) > STAGE1_LIMIT:
        random.shuffle(stage1)
        s1_balanced = stage1[:STAGE1_LIMIT]
        print(f"   -> Downsampled from {len(stage1)} to {len(s1_balanced)}")
    else:
        s1_balanced = stage1

    # 2. Stage 2: Keep ALL
    # This contains all the unique SURV data we want.
    print(f"Processing Stage 2...")
    s2_balanced = stage2
    print(f"   -> Kept all {len(s2_balanced)} (Contains ~13k SURV)")

    # 3. Stage 3: Upsample to match SURV count
    print(f"Processing Stage 3...")
    if len(stage3) > 0:
        multiplier = TARGET_SURV_COUNT // len(stage3)
        # Add a little extra to strictly meet the target if floor division is low
        s3_balanced = stage3 * multiplier
        
        # Fine-tuning: If multiplier is small (e.g. 17), we might want exactly 13000
        # Let's just stick to simple multiplication for safety
        print(f"   -> Upsampled {len(stage3)} x {multiplier} = {len(s3_balanced)}")
    else:
        s3_balanced = []

    # 4. Combine and Shuffle
    final_data = s1_balanced + s2_balanced + s3_balanced
    random.shuffle(final_data)

    print(f"\nFinal Phase 2 Dataset:")
    print(f"   Total Size: {len(final_data)}")
    print(f"   - Stage 1 (Easy ECOM): ~{len(s1_balanced)}")
    print(f"   - Stage 2 (Unique + Mid): ~{len(s2_balanced)}")
    print(f"   - Stage 3 (Hard Dialect): ~{len(s3_balanced)}")
    
    # Validation of ECOM/SURV Balance
    # (Approximate based on your stats)
    # Total ECOM ~ 6000 (S1) + 6700 (S2) = 12,700
    # Total SURV ~ 13000 (S2) + 600 (S3) = 13,600
    # Ratio is almost perfect 1:1.

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(final_data)
        
    print(f"✅ Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    create_balanced_dataset()