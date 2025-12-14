# ESPnet InterCTC Compatibility Patch

**⚠️ Issue Description:**
You may encounter runtime errors when using **Intermediate CTC (InterCTC)** with HuBERT or other `s3prl` upstream encoders in ESPnet. This occurs because the standard upstream encoder returns a 2-tuple `(x, ilens)`, whereas InterCTC requires a 3-tuple `(x, ilens, intermediate_outs)`.

**✅ Solution:**
This directory contains the necessary code to wrap the encoder and force compatibility without modifying the core ESPnet source code.

### 🚀 Usage Instructions
Please apply the fixes in the following order:

1.  **hubert_fix.py:**
2.  **fix_hubert_manual_forward**
3.  **fix_hubert.py**
4.  **restore_hubert.py**
5.  **fix_hubert_hook.py**
6.  **Verify Config:** Ensure your YAML config defines `interctc_layer_idx` correctly (e.g., `[6, 12]`) and run the training script.

*(If issues persist, please refer to the project documentation or consult your AI assistant for debugging steps.)*