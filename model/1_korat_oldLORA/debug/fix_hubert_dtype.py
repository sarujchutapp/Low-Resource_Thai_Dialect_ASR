import os
import site

def fix_hubert_dtype():
    # 1. Locate the file
    site_packages = site.getsitepackages()
    target_file = None
    
    # We look for fairseq/models/hubert/hubert.py
    for path in site_packages:
        potential_path = os.path.join(path, "fairseq", "models", "hubert", "hubert.py")
        if os.path.exists(potential_path):
            target_file = potential_path
            break
            
    if not target_file:
        print("❌ Could not find fairseq/models/hubert/hubert.py")
        return

    print(f"🔧 Patching file: {target_file}")

    with open(target_file, "r") as f:
        content = f.read()

    # 2. Define the broken line and the fixed line
    # Broken: x[mask_indices] = self.mask_emb
    # Fixed:  x[mask_indices] = self.mask_emb.to(x.dtype)
    
    broken_code = "x[mask_indices] = self.mask_emb"
    fixed_code = "x[mask_indices] = self.mask_emb.to(x.dtype)"

    if broken_code in content:
        new_content = content.replace(broken_code, fixed_code)
        
        with open(target_file, "w") as f:
            f.write(new_content)
        print("✅ Successfully patched hubert.py to support BFloat16/Mixed Precision!")
    elif fixed_code in content:
        print("⚠️ File is already patched.")
    else:
        print("❌ Could not find the specific line to patch. The file version might be different.")

if __name__ == "__main__":
    fix_hubert_dtype()
