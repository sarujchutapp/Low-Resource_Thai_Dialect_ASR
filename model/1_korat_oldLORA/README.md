## 📂 Project Structure

### Core Scripts & Configs
* **`train.py`**: Main training script using **ESPnet-EZ**. Handles model building, LoRA injection, and the training loop.
* **`benchmark.py`**: Evaluation script. Loads the trained model to calculate WER/CER on the test set.
* **`inference.py`**: Script for transcribing custom audio files.
* **`finetune_with_lora.yaml`**: Master configuration file defining model hyperparameters and training settings.

### Directories
* **`debug/`**: Contains debug files.
* **`dev_audio00/`**: **[Place Audio Files Here]** (See README inside folder for setup instructions).
* **`train_audio00/`**: **[Place Audio Files Here]** (See README inside folder for setup instructions).
* **`transcription/`**: Contains the CSV manifest files (`train.csv`, `dev.csv`) linking audio paths to text.
* **`util/`**: Helper scripts (e.g., vocabulary generation, LoRA target name finder).

### Key Files
* **`word_korat.txt`**: A list of all unique words in the Korat dataset. This is required for token validation during evaluation utilizing PyThaiNLP. You can generate a new list for a different dialect using **`util/create_vocab.py`**.