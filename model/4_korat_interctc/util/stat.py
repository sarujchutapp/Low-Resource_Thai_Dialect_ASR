import csv
import jiwer
from pythainlp import Tokenizer
from tqdm import tqdm
from collections import Counter

# ================= CONFIGURATION =================
TRAIN_CSV = "train.csv"
VOCAB_FILE = "word_korat.txt"
DIALECT_COL = "sentence"
CENTRAL_COL = "thai_sentence"
TYPE_COL = "dialect_type"

FREQ = 260
DIST = 0.4

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
        dialect_type = row[TYPE_COL]
        
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
        
        if freq > FREQ:
            # Stage 1: High Frequency (ECOM Repetitions)
            # Logic: Easy to learn acoustics because the text is predictable
            stage = 1
            score = 0.0 + (dist * 0.01) # Minor sort by distance within this group
        else:
            # Unique Sentences (SURV)
            if dist < DIST: # Adjustable threshold
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
        row["dialect_type"] = dialect_type
        scored_data.append(row)


    # Statistics
    s1 = len([x for x in scored_data if x["stage"] == 1])
    s1_surv = len([x for x in scored_data if x["stage"] == 1 and x[TYPE_COL] == "SURV"])
    s1_ecom = len([x for x in scored_data if x["stage"] == 1 and x[TYPE_COL] == "ECOM"])
    s2 = len([x for x in scored_data if x["stage"] == 2])
    s2_surv = len([x for x in scored_data if x["stage"] == 2 and x[TYPE_COL] == "SURV"])
    s2_ecom = len([x for x in scored_data if x["stage"] == 2 and x[TYPE_COL] == "ECOM"])
    s3 = len([x for x in scored_data if x["stage"] == 3])
    s3_surv = len([x for x in scored_data if x["stage"] == 3 and x[TYPE_COL] == "SURV"])
    s3_ecom = len([x for x in scored_data if x["stage"] == 3 and x[TYPE_COL] == "ECOM"])
    
    print(f"\n✅ Statistics:")
    print(f"   Stage 1 (Repetitions/ECOM):  {s1} files (frequency > {FREQ}) -> SURV: {s1_surv}, ECOM: {s1_ecom}")
    print(f"   Stage 2 (Similar/SURV):      {s2} files (distance < {DIST}) -> SURV: {s2_surv}, ECOM: {s2_ecom}")
    print(f"   Stage 3 (Hard Dialect):      {s3} files (distance >= {DIST}) -> SURV: {s3_surv}, ECOM: {s3_ecom}")