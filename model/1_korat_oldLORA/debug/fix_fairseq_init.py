import os
import site

def fix_fairseq_init():
    # 1. Locate the file
    site_packages = site.getsitepackages()
    target_file = None
    
    for path in site_packages:
        potential_path = os.path.join(path, "fairseq", "dataclass", "initialize.py")
        if os.path.exists(potential_path):
            target_file = potential_path
            break
            
    if not target_file:
        print("❌ Could not find fairseq/dataclass/initialize.py")
        return

    print(f"🔧 Patching file: {target_file}")

    with open(target_file, "r") as f:
        lines = f.readlines()

    new_lines = []
    import_added = False
    
    # We need to replace the loop that handles registration
    # The original code usually looks like:
    #    if v.default is MISSING:
    #        continue
    #    cs.store(name=k, node=v.default)
    
    skip_mode = False
    
    for line in lines:
        # 2. Add 'import dataclasses' at the top
        if not import_added and ("import" in line or "from" in line):
            new_lines.append("import dataclasses\n")
            import_added = True
            
        # 3. Detect the loop logic we need to replace
        if "if v.default is MISSING:" in line:
            # We found the block! We will replace the next few lines with our custom logic.
            # We skip the original lines until we hit the 'cs.store' line.
            
            # Insert our new logic
            indent = line.split("if")[0] # Capture indentation
            new_lines.append(f"{indent}if v.default_factory is not dataclasses.MISSING:\n")
            new_lines.append(f"{indent}    node = v.default_factory()\n")
            new_lines.append(f"{indent}elif v.default is not MISSING:\n")
            new_lines.append(f"{indent}    node = v.default\n")
            new_lines.append(f"{indent}else:\n")
            new_lines.append(f"{indent}    continue\n")
            new_lines.append(f"{indent}cs.store(name=k, node=node)\n")
            
            skip_mode = True # Start skipping original lines
            continue
            
        if skip_mode:
            # We skip lines until we see 'cs.store', which is the last line of the block we are replacing
            if "cs.store(name=k, node=v.default)" in line:
                skip_mode = False # Stop skipping after this line
            continue
            
        new_lines.append(line)

    with open(target_file, "w") as f:
        f.writelines(new_lines)
    
    print("✅ Successfully patched initialize.py to handle default_factory!")

if __name__ == "__main__":
    fix_fairseq_init()
