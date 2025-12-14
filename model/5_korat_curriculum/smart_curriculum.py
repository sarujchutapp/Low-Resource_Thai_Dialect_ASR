import csv
import jiwer
from pythainlp import Tokenizer
from tqdm import tqdm
from collections import Counter

# ================= CONFIGURATION =================
TRAIN_CSV = f"./transcription/train.csv"
VOCAB_FILE = "word_korat.txt"
DIALECT_COL = "sentence"
CENTRAL_COL = "thai_sentence"

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

    # 1. Count Frequency of every sentence
    # (To identify the "ECOM" repetitions)
    print("Analyzing repetitions...")
    texts = [row[DIALECT_COL] for row in data]
    text_counts = Counter(texts)

    print(f"Calculating Difficulty for {len(data)} samples...")
    scored_data = []
    
    for row in tqdm(data):
        text = row[DIALECT_COL]
        
        # 2. Calculate "Central-ness" (Text Distance)
        try:
            # Tokenize for fair comparison
            dialect_tok = " ".join(tokenizer.word_tokenize(text.replace(" ", "")))
            central_tok = " ".join(tokenizer.word_tokenize(row[CENTRAL_COL].replace(" ", "")))
            
            # CER Distance (0.0 = Identical to Central, 1.0+ = Very Different)
            dist = jiwer.cer(dialect_tok, central_tok)
        except:
            dist = 0.5 # Default if missing central text
        
        # 3. Assign Stage Category
        freq = text_counts[text]
        
        if freq > 1:
            # Stage 1: High Frequency (ECOM Repetitions)
            # Logic: Easy to learn acoustics because the text is predictable
            stage = 1
            score = 0.0 + (dist * 0.01) # Minor sort by distance within this group
        else:
            # Unique Sentences (SURV)
            if dist < 0.4: # Adjustable threshold
                # Stage 2: Unique but similar to Central
                stage = 2
                score = 10.0 + dist
            else:
                # Stage 3: The "Hard" 3% (Unique AND very different)
                stage = 3
                score = 20.0 + dist

        row["stage"] = stage
        row["frequency"] = freq
        row["dist_score"] = dist
        scored_data.append(row)

    # 4. Sort
    # Sort primarily by Stage (1 -> 2 -> 3), secondarily by Distance
    print("Sorting...")
    scored_data.sort(key=lambda x: (x["stage"], x["dist_score"]))

    # 5. Save
    out_file = "train_smart_curriculum.csv"
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(scored_data[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_data)

    # Statistics
    s1 = len([x for x in scored_data if x["stage"] == 1])
    s2 = len([x for x in scored_data if x["stage"] == 2])
    s3 = len([x for x in scored_data if x["stage"] == 3])
    
    print(f"\n✅ Curriculum Created: {out_file}")
    print(f"   Stage 1 (Repetitions/ECOM):  {s1} files (Acoustic Foundation)")
    print(f"   Stage 2 (Similar/SURV):      {s2} files (Grammar Expansion)")
    print(f"   Stage 3 (Hard Dialect):      {s3} files (The 'Expert' Exam)")