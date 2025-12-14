import os

# Path to your specific file
TARGET_FILE = "/mnt/c/users/Saruj/espnet/espnet2/asr/encoder/hubert_encoder.py"

def fix_hubert_manual_forward():
    if not os.path.exists(TARGET_FILE):
        print(f"❌ Error: Could not find {TARGET_FILE}")
        return

    print(f"🔧 Patching {TARGET_FILE} with Manual Forward Pass...")

    with open(TARGET_FILE, "r") as f:
        content = f.read()

    # We need to identify the block where self.encoders() is called and replace it.
    # We look for the specific 'with torch.no_grad()...' block.
    
    # The original broken code usually looks like this:
    # with torch.no_grad() if not ft else contextlib.nullcontext():
    #    enc_outputs = self.encoders(
    #        ...
    #    )
    
    search_pattern = "with torch.no_grad() if not ft else contextlib.nullcontext():"
    
    if search_pattern not in content:
        print("❌ Could not locate the forward pass block. The file might be different than expected.")
        return

    # We will construct the NEW code block that manually extracts features
    new_code_block = """        with torch.no_grad() if not ft else contextlib.nullcontext():
            # --- PATCH: Manual Forward to get Intermediate Layers (InterCTC) ---
            # 1. Extract Features (CNN)
            if hasattr(self.encoders, "feature_extractor"):
                features = self.encoders.feature_extractor(xs_pad)
            elif hasattr(self.encoders, "wav2vec2"): 
                features = self.encoders.wav2vec2.feature_extractor(xs_pad)
            else:
                # Fallback 
                features = self.encoders.extract_features(xs_pad, padding_mask=masks, mask=False)["x"]

            # 2. Handle Feature Projection (Linear)
            if hasattr(self.encoders, "feature_projection"):
                features, _ = self.encoders.feature_projection(features)
            elif hasattr(self.encoders, "wav2vec2"):
                features, _ = self.encoders.wav2vec2.feature_projection(features)
            
            # 3. Run Transformer Encoder with 'return_all_hiddens=True'
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
            # We convert the list of layers into the format ESPnet wants
            layer_results = [x.transpose(0, 1) for x in encoder_out["encoder_states"]]
            
            enc_outputs = {
                "x": encoder_out["encoder_out"],
                "padding_mask": encoder_out["encoder_padding_mask"],
                "layer_results": layer_results
            }
            # -----------------------------------------------------------
"""

    # Logic to replace the old block
    parts = content.split(search_pattern)
    pre_part = parts[0]
    
    # Find where the old block ends (usually at 'xs_pad = enc_outputs["x"]')
    remaining = parts[1]
    end_marker = 'xs_pad = enc_outputs["x"]'
    
    if end_marker in remaining:
        post_part = remaining.split(end_marker, 1)[1]
        
        # Combine everything
        new_content = pre_part + new_code_block + "        " + end_marker + post_part
        
        with open(TARGET_FILE, "w") as f:
            f.write(new_content)
        print("✅ Successfully patched hubert_encoder.py!")
    else:
        print("❌ Could not find the end of the code block. Patch failed.")

if __name__ == "__main__":
    fix_hubert_manual_forward()