import csv

# Your training data file (change filename if needed)
INPUT_CSV = "[training transciption].csv" 
OUTPUT_FILE = "word.txt"

def create_vocab_file():
    unique_words = set()
    
    print(f"Reading {INPUT_CSV}...")
    try:
        with open(INPUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Assuming text is in 'sentence' column
                # and words are separated by spaces
                text = row.get('sentence', '') 
                words = text.strip().split()
                unique_words.update(words)
                
        # Sort and write to file
        sorted_words = sorted(list(unique_words))
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            # OPTIONAL: Add special tokens if your decoder crashes without them
            # f.write("<blank>\n")
            # f.write("<unk>\n")
            for word in sorted_words:
                f.write(f"{word}\n")
                
        print(f"✅ Successfully created {OUTPUT_FILE} with {len(sorted_words)} unique words.")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {INPUT_CSV}. Please check the filename.")

if __name__ == "__main__":
    create_vocab_file()