# Low-Resource Thai Dialect ASR
This project is a part of "2110391 INDIVIDUAL STUDY IN COMPUTER ENGINEERING"

The project explores the feasibility of reproducing state-of-the-art Automatic Speech Recognition (ASR) performance for low-resource Thai dialects under severe hardware constraints (Single GPU, 4GB VRAM).

Our methodology replicates and optimizes **Experiment 2** from Suwanbandit et al. (2023), aiming to achieve competitive Word Error Rates (WER) and Character Error Rates (CER) using **Low-Rank Adaptation (LoRA)** instead of computationally expensive full-parameter fine-tuning.

## 📄 Key Reference
**Base Paper:** [Thai Dialect Corpus and Transfer-based Curriculum Learning Investigation for Dialect Automatic Speech Recognition](https://www.isca-archive.org/interspeech_2023/suwanbandit23_interspeech.html) (Suwanbandit et al., Interspeech 2023)

## 🎯 Project Goals
1.  **Efficiency Benchmark:** Demonstrate that lightweight adaptation (LoRA) can rival full fine-tuning performance using ~10% of trainable parameters.
2.  **Constraint Optimization:** Implement memory-efficient strategies (Gradient Accumulation, Lazy Loading, OTF Augmentation) to train Conformer-CTC models on consumer-grade hardware.
3.  **Technique Validation:** Evaluate the effectiveness of specific advanced methodologies in enhancing model robustness and convergence speed for dialectal speech.

## 🛠️ Implementation Details
This project is built on **[ESPnet-EZ](https://espnet.github.io/espnet/notebook/ESPnetEZ/ASR/ASR_finetune_owsm.html)** (based on the OWSM fine-tuning workflow).

We selected `espnetez` over standard ESPnet recipes to enable **custom Python-based data scheduling**. This allowed us to inject dynamic curriculum logic (Stage 1 vs. Stage 2 switching) and custom upsampling algorithms directly into the training loop, which is not natively supported in standard shell-based recipes.

* **Toolkit:** [ESPnet](https://github.com/espnet/espnet) (End-to-End Speech Processing Toolkit)
* **Model:** Conformer Encoder + Transformer Decoder + CTC
* **Techniques:** LoRA, Curriculum Learning, Speed Perturbation, Intermediate CTC, Class-Balanced Resampling

## 📊 Results & Methodology
For detailed experimental setup and comparative results against the official baseline, please refer to the project report:
👉 **[ASR.pdf](./ASR.pdf)**

## 📂 Dataset
This project utilizes the **Thai Dialect Corpus**, specifically the **Korat** dialect subset.
* **Source:** [SLSCU/thai-dialect-corpus](https://github.com/SLSCU/thai-dialect-corpus)
* **License:** CC-BY-SA 4.0