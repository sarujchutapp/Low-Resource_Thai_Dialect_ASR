import os

TARGET_FILE = "[path to ESPnet]/espnet/espnet2/asr/encoder/hubert_encoder.py"

def fix_hubert_manual_forward():
    if not os.path.exists(TARGET_FILE):
        print(f"❌ Error: Could not find {TARGET_FILE}")
        return

    print(f"🔧 Patching {TARGET_FILE} with Manual Forward Pass...")

    with open(TARGET_FILE, "r") as f:
        content = f.read()

    # We need to completely replace the block where self.encoders() is called.
    # The original code looks like this:
    #    enc_outputs = self.encoders(
    #        xs_pad,
    #        padding_mask=masks,
    #        mask=self.apply_mask and self.training,
    #        features_only=True,
    #        output_layer=None,
    #        layer_results=True,  <-- This caused the error
    #    )

    # We will replace it with a manual call to the internal components.
    
    # 1. Identify the start of the block to replace
    start_marker = "with torch.no_grad() if not ft else contextlib.nullcontext():"
    
    # 2. Identify the end of the block (where we process xs_pad)
    end_marker = 'xs_pad = enc_outputs["x"]'

    # 3. Create the NEW code block
    # This code manually calls feature_extractor -> feature_projection -> encoder
    # This bypasses the restrictive HubertModel.forward() method.
    new_code_block = """        with torch.no_grad() if not ft else contextlib.nullcontext():
            # --- MANUAL FORWARD PASS START (InterCTC Fix) ---
            # 1. Extract Features (CNN)
            if hasattr(self.encoders, "feature_extractor"):
                features = self.encoders.feature_extractor(xs_pad)
            elif hasattr(self.encoders, "wav2vec2"): # Handle wrapper variants
                features = self.encoders.wav2vec2.feature_extractor(xs_pad)
            else:
                # Fallback for weird models
                features = self.encoders.extract_features(xs_pad, padding_mask=masks, mask=False)["x"]

            # 2. Handle Feature Projection (if it exists)
            if hasattr(self.encoders, "feature_projection"):
                features, _ = self.encoders.feature_projection(features)
            
            # 3. Run Encoder with 'return_all_hiddens' (This is what we needed!)
            # Note: masks needs to be inverted for some versions, but Fairseq usually handles it.
            # We use the internal 'encoder' module directly.
            if hasattr(self.encoders, "encoder"):
                encoder_out = self.encoders.encoder(
                    features,
                    padding_mask=masks,
                    return_all_hiddens=True 
                )
            elif hasattr(self.encoders, "wav2vec2"):
                encoder_out = self.encoders.wav2vec2.encoder(
                    features,
                    padding_mask=masks,
                    return_all_hiddens=True
                )
            else:
                raise RuntimeError("Could not find internal encoder in Fairseq model!")

            # 4. Pack results to match ESPnet expectations
            enc_outputs = {
                "x": encoder_out["encoder_out"],
                "padding_mask": encoder_out["encoder_padding_mask"],
                "layer_results": encoder_out["encoder_states"]
            }
            # --- MANUAL FORWARD PASS END ---
"""

    # We need to find the chunk to replace. 
    # Since the file might be slightly different, we use regex or string splitting.
    if start_marker in content and end_marker in content:
        pre_part = content.split(start_marker)[0]
        post_part = content.split(end_marker)[1]
        
        # Check if we already patched it to avoid duplication
        if "MANUAL FORWARD PASS START" in content:
            print("⚠️  File already appears to be patched.")
            return

        # Construct the new file content
        new_content = pre_part + new_code_block + "\n        " + end_marker + post_part
        
        with open(TARGET_FILE, "w") as f:
            f.write(new_content)
        print("✅ Successfully patched hubert_encoder.py!")
        
    else:
        print("❌ Could not locate the specific code block to patch.")
        print("Please check if the file content matches the expected structure.")

if __name__ == "__main__":
    fix_hubert_manual_forward()