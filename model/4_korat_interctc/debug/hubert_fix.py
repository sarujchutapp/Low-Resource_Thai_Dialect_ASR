import os
import re

# Path to your specific file
TARGET_FILE = "/mnt/c/users/Saruj/espnet/espnet2/asr/encoder/hubert_encoder.py"

def patch_hubert_interctc():
    if not os.path.exists(TARGET_FILE):
        print(f"❌ Error: Could not find {TARGET_FILE}")
        return

    print(f"🔧 Patching {TARGET_FILE} to enable InterCTC...")

    with open(TARGET_FILE, "r") as f:
        content = f.read()

    # 1. Modify the forward call to request 'layer_results'
    # We look for the call to self.encoders(...) inside the forward method
    # Original: output_layer=None,
    # New:      output_layer=None, layer_results=True,
    
    if "layer_results=True" in content:
        print("⚠️  File seems to be already patched for layer_results.")
    else:
        content = content.replace(
            "output_layer=None,", 
            "output_layer=None, layer_results=True,"
        )
        print("   -> Enabled layer_results=True in forward pass")

    # 2. Modify the return statement to return the layers
    # Original: return xs_pad, olens, None
    # New:      return xs_pad, olens, intermediate_outs
    
    # We need to capture the layer results from enc_outputs before we return
    # Find the line where enc_outputs is deleted or used
    
    extraction_logic = """
        # Extract intermediate outputs for InterCTC
        intermediate_outs = None
        if "layer_results" in enc_outputs:
            # enc_outputs["layer_results"] is a list of (x, attn) tuples
            # We want just 'x', and we permute to (B, T, D)
            intermediate_outs = [l[0].transpose(0, 1) for l in enc_outputs["layer_results"]]
    """
    
    # Insert extraction logic before "del enc_outputs"
    if "intermediate_outs =" not in content:
        content = content.replace('del enc_outputs', extraction_logic + '\n        del enc_outputs')
        print("   -> Added logic to extract intermediate layers")

    # Change the return statement
    if "return xs_pad, olens, None" in content:
        content = content.replace(
            "return xs_pad, olens, None", 
            "return xs_pad, olens, intermediate_outs"
        )
        print("   -> Updated return statement to pass intermediate_outs")
    else:
        print("⚠️  Could not find standard return statement. Please check manually.")

    # Write back
    with open(TARGET_FILE, "w") as f:
        f.write(content)
    
    print("✅ hubert_encoder.py patched! You can now use InterCTC.")

if __name__ == "__main__":
    patch_hubert_interctc()