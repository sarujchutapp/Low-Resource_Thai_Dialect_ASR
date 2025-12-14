import csv
import jiwer
from pythainlp import Tokenizer
from tqdm import tqdm

# ================= CONFIGURATION =================
TRAIN_CSV = "train.csv"
VOCAB_FILE = "word_korat.txt"

# Define your column names exactly as they appear in your CSV
DIALECT_COL = "sentence"          # Ground Truth (Korat)
CENTRAL_COL = "thai_sentence"      # Central Thai translation

# ================= HELPERS =================
def get_tokenizer(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f.readlines() if line.strip()]
    return Tokenizer(set(vocab), engine='newmm')

# ================= MAIN =================
if __name__ == "__main__":
    tokenizer = get_tokenizer(VOCAB_FILE)
    
    print(f"Reading {TRAIN_CSV}...")
    with open(TRAIN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    print(f"Calculating Linguistic Distance for {len(data)} samples...")
    
    scored_data = []
    
    for row in tqdm(data):
        try:
            # 1. Get the two texts
            dialect_text = row[DIALECT_COL].replace(" ", "")
            central_text = row[CENTRAL_COL].replace(" ", "")

            # 2. Tokenize (Critical for fair comparison)
            # We treat them as sequences of words to find the structural difference
            dialect_tok = " ".join(tokenizer.word_tokenize(dialect_text))
            central_tok = " ".join(tokenizer.word_tokenize(central_text))

            # 3. Calculate CER/WER Distance
            # High Score = Big difference (Hard)
            # Low Score = Identical (Easy)
            # We use CER because it captures slight spelling changes (accent) better than WER
            dist = jiwer.cer(dialect_tok, central_tok)
            
            row["difficulty_score"] = dist
            scored_data.append(row)
            
        except KeyError:
            print(f"❌ Error: Column '{CENTRAL_COL}' not found in CSV!")
            exit()

    # 4. Sort (Easy -> Hard)
    print("Sorting data...")
    scored_data.sort(key=lambda x: x["difficulty_score"])

    # 5. Save
    out_file = "train_curriculum_text.csv"
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(scored_data[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_data)
        
    print(f"✅ Saved sorted curriculum to {out_file}")
    print(f"   - Top rows: {scored_data[0][DIALECT_COL]} (Score: {scored_data[0]['difficulty_score']:.2f})")
    print(f"   - Bottom rows: {scored_data[-1][DIALECT_COL]} (Score: {scored_data[-1]['difficulty_score']:.2f})")