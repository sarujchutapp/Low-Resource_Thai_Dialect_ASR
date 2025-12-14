import os

# 1. Hardcoded path based on your error logs
# This is the file causing the 13GB memory crash
TARGET_FILE = "[path to ESPnet]/espnet/espnet2/asr/encoder/hubert_encoder.py"

def fix_hubert_mask():
    if not os.path.exists(TARGET_FILE):
        print(f"❌ Error: Could not find {TARGET_FILE}")
        return

    print(f"🔧 Patching file: {TARGET_FILE}")

    with open(TARGET_FILE, "r") as f:
        content = f.read()

    # 2. Define the inefficient line and the optimized block
    # The line causing the 13GB crash (O(N^2) memory):
    target_line = "masks = make_pad_mask(ilens).to(xs_pad.device)"
    
    # The optimized replacement (O(N) linear memory):
    # This manually calculates the mask indices without creating the massive matrix first
    replacement_code = """# Optimized mask creation to avoid O(N^2) memory crash on raw audio
        # masks = make_pad_mask(ilens).to(xs_pad.device)
        B_dim = xs_pad.size(0)
        T_dim = xs_pad.size(1)
        masks = torch.arange(T_dim, device=xs_pad.device).view(1, -1).expand(B_dim, -1) >= ilens.view(-1, 1)"""

    if target_line in content:
        new_content = content.replace(target_line, replacement_code)
        
        with open(TARGET_FILE, "w") as f:
            f.write(new_content)
        print("✅ Successfully patched hubert_encoder.py to use Linear Memory masking!")
    elif "torch.arange(T_dim" in content:
        print("⚠️ File is already patched.")
    else:
        print("❌ Could not find the specific line to patch. The file might have changed.")
        print("   Please look for 'masks = make_pad_mask(ilens).to(xs_pad.device)' manually.")

if __name__ == "__main__":
    fix_hubert_mask()
