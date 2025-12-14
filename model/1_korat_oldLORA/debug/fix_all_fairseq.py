import os
import re
import site

def fix_all_fairseq_files():
    # 1. Locate FairSeq
    site_packages = site.getsitepackages()
    fairseq_path = None
    for path in site_packages:
        p = os.path.join(path, "fairseq")
        if os.path.exists(p):
            fairseq_path = p
            break
            
    if not fairseq_path:
        print("❌ Could not find fairseq installation.")
        return

    print(f"🔍 Scanning {fairseq_path} for Python 3.11 dataclass issues...")
    
    # Pattern:  indentation + variable + : + Type + = + Type + ()
    # Example:  "    encoder: EncDecBaseConfig = EncDecBaseConfig()"
    # We use \3 to ensure the Type name on both sides is identical.
    pattern = re.compile(r"(\s+)(\w+):\s*([a-zA-Z0-9_]+)\s*=\s*(\3)\(\)")

    files_fixed = 0
    
    for root, _, files in os.walk(fairseq_path):
        for file in files:
            if not file.endswith(".py"):
                continue
                
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Check if file has the issue
            if not pattern.search(content):
                continue
                
            # If found, process line by line to modify safely
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            
            new_lines = []
            modified = False
            
            for line in lines:
                match = pattern.search(line)
                if match:
                    # Construct the fix: "var: Type = field(default_factory=Type)"
                    indent, var_name, type_name, _ = match.groups()
                    # Keep comments if they exist? regex ignores them, simple replace is safer
                    # Note: We strip the newline from regex match, so we add it back
                    new_line = f"{indent}{var_name}: {type_name} = field(default_factory={type_name})\n"
                    new_lines.append(new_line)
                    modified = True
                else:
                    new_lines.append(line)
            
            if modified:
                # Ensure 'field' is imported
                header_content = "".join(new_lines[:30]) # Check top 30 lines
                if "from dataclasses import" in header_content:
                    if "field" not in header_content:
                        # Find the dataclasses import and append field
                        for i, line in enumerate(new_lines):
                            if "from dataclasses import" in line:
                                new_lines[i] = line.strip() + ", field\n"
                                break
                else:
                    # Add import if missing entirely
                    new_lines.insert(0, "from dataclasses import field\n")
                
                # Write back
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                print(f"✅ Fixed: {file}")
                files_fixed += 1

    print(f"\n🎉 Finished! Patched {files_fixed} files.")

if __name__ == "__main__":
    fix_all_fairseq_files()
