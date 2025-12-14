import os
import re

# The specific file causing the crash
TARGET_FILE = "[path to ESPnet]/espnet/tools/miniconda/envs/espnet/lib/python3.11/site-packages/fairseq/models/transformer/transformer_config.py"

def fix_transformer_config():
    if not os.path.exists(TARGET_FILE):
        print(f"❌ Could not find: {TARGET_FILE}")
        return

    print(f"🔧 Patching {TARGET_FILE}...")

    with open(TARGET_FILE, "r") as f:
        content = f.read()

    # 1. Add 'field' to imports if missing
    if "from dataclasses import dataclass" in content and "from dataclasses import dataclass, field" not in content:
        content = content.replace(
            "from dataclasses import dataclass", 
            "from dataclasses import dataclass, field"
        )
        print("   -> Added 'field' import")

    # 2. Fix the specific 'quant_noise' error (and others in this file)
    # Matches:  any_var: AnyType = SomeClass()
    pattern = r"(\s+)(\w+):\s*(.+?)\s*=\s*(\w+)\(\)"
    
    def replacement(match):
        indent, var_name, type_hint, class_name = match.groups()
        # Only replace if it looks like a Class instantiation (starts with Uppercase)
        if class_name[0].isupper():
            return f"{indent}{var_name}: {type_hint} = field(default_factory={class_name})"
        return match.group(0)

    new_content, count = re.subn(pattern, replacement, content)
    
    if count > 0:
        with open(TARGET_FILE, "w") as f:
            f.write(new_content)
        print(f"✅ Fixed {count} mutable defaults in transformer_config.py!")
    else:
        print("⚠️ No patterns found. The file might already be fixed or format is unexpected.")

if __name__ == "__main__":
    fix_transformer_config()
