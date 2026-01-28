
import argparse
import re
import os

SETUP_PY = "setup.py"

def get_version():
    if not os.path.exists(SETUP_PY):
        return "0.0.0"
    with open(SETUP_PY, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    
    match = re.search(r"version\s*=\s*'([^']+)'", content)
    if match:
        return match.group(1)
        
    return "0.0.0"

def update_version(new_version):
    if not os.path.exists(SETUP_PY):
        print(f"Error: {SETUP_PY} not found.")
        return

    with open(SETUP_PY, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to find version="..." or version='...'
    # We replace the whole line or just the value? Replacing value is safer if formatted well.
    # But setup.py is python code.
    
    # Try double quotes
    updated_content = re.sub(r'(version\s*=\s*)"([^"]+)"', f'\\1"{new_version}"', content)
    
    # Check if changed (if not, try single quotes)
    if updated_content == content:
        updated_content = re.sub(r"(version\s*=\s*)'([^']+)'", f"\\1'{new_version}'", content)

    if updated_content != content:
        with open(SETUP_PY, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f"Updated setup.py version to {new_version}")
    else:
        print("Could not find version string to update in setup.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update_version", type=str, help="New version string to set")
    args = parser.parse_args()

    if args.update_version:
        update_version(args.update_version)
    else:
        print(get_version())
