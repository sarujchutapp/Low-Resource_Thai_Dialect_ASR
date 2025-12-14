import os
import re
import sys

# Hardcoded path from your traceback
TARGET_DIR = "[path to ESPnet]/espnet/tools/miniconda/envs/espnet/lib/python3.11/site-packages/fairseq"

def fix_files():
    if not os.path.exists(TARGET_DIR):
        print(f"❌ Error: Could not find directory: {TARGET_DIR}")
        return

    print(f"🚀 Scanning {TARGET_DIR} for Python 3.11 issues...")
    
    # Regex Breakdown:
    # 1. (\s+)      -> Capture indentation
    # 2. (\w+)      -> Capture variable name (e.g., 'quant_noise')
    # 3. :\s* -> Colon and spaces
    # 4. (.+?)      -> Capture Type hint (lazy match, e.g., 'QuantNoiseConfig' or 'Optional[int]')
    # 5. \s*=\s* -> Equals sign
    # 6. (\w+)\(\)  -> Capture Class instantiation (e.g., 'QuantNoiseConfig()')
    #
    # We look for:  var: Type = Class()
    pattern = re.compile(r"^(\s+)(\w+):\s*(.+?)\s*=\s*(\w+)\(\)\s*(?:#.*)?$")

    files_fixed = 0

    for root, _, files in os.walk(TARGET_DIR):
        for file in files:
            if not file.endswith(".py"):
                continue
            
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            new_lines = []
            file_modified = False
            has_dataclass = False

            for line in lines:
                # Check if file uses dataclasses
                if "@dataclass" in line:
                    has_dataclass = True

                match = pattern.match(line)
                if match and has_dataclass:
                    indent, var_name, type_hint, class_name = match.groups()
                    
                    # Logic: If we see 'var: Type = Class()', it's illegal in Py3.11.
                    # We change it to: 'var: Type = field(default_factory=Class)'
                    
                    # Only apply if the type and class look related (heuristic) 
                    # or if the Class is clearly a configuration object (Capitalized).
                    if class_name[0].isupper():
                        new_line = f"{indent}{var_name}: {type_hint} = field(default_factory={class_name})\n"
                        new_lines.append(new_line)
                        file_modified = True
                        continue

                new_lines.append(line)

            if file_modified:
                # Ensure 'field' is imported
                header_check = "".join(new_lines[:50])
                if "from dataclasses import" in header_check and "field" not in header_check:
                    # Find the import line and add 'field'
                    for i, line in enumerate(new_lines):
                        if "from dataclasses import" in line:
                            # Replace "from dataclasses import dataclass" 
                            # with "from dataclasses import dataclass, field"
                            new_lines[i] = line.strip() + ", field\n"
                            break
                elif "from dataclasses import" not in header_check:
                     new_lines.insert(0, "from dataclasses import field\n")

                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                print(f"✅ Fixed: {file}")
                files_fixed += 1

    print(f"\n🎉 Done! Fixed {files_fixed} files.")

if __name__ == "__main__":
    fix_files()
