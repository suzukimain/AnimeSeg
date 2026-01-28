import argparse
import re

import packaging.version


REPLACE_PATTERNS = {
    "init": (
        re.compile(r'^__version__\s*=\s*"([^"]+)"\s*$', re.MULTILINE),
        '__version__ = "VERSION"\n',
    ),
    "setup": (
        re.compile(r'^(\s*)version\s*=\s*"[^"]+",', re.MULTILINE),
        r'\1version="VERSION",',
    ),
}
REPLACE_FILES = {
    "init": "src/anime_seg/__init__.py",
    "setup": "setup.py",
}
README_FILE = "README.md"


def update_version_in_file(fname, version, pattern):
    """Update the version in one file using a specific pattern."""
    with open(fname, "r", encoding="utf-8", newline="\n") as f:
        code = f.read()

    re_pattern, replace = REPLACE_PATTERNS[pattern]
    replace_filled = replace.replace("VERSION", version)

    # Try a direct substitution first
    new_code, n_subs = re_pattern.subn(replace_filled, code)
    if n_subs == 0:
        # Fallbacks for common cases
        if pattern == "init":
            # Try a looser pattern to find __version__ assignment
            loose = re.compile(r'__version__\s*=\s*["\']([^"\']+)["\']')
            if loose.search(code):
                new_code = loose.sub(f'__version__ = "{version}"', code)
                n_subs = 1
        if n_subs == 0:
            raise RuntimeError(f"Could not find version pattern '{pattern}' in {fname}")

    with open(fname, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_code)


def global_version_update(version):
    """Update the version in all needed files."""
    for pattern, fname in REPLACE_FILES.items():
        print(f"Updating {fname} ({pattern}) -> {version}")
        update_version_in_file(fname, version, pattern)


def get_version():
    """Reads the current version in the __init__."""
    with open(REPLACE_FILES["init"], "r", encoding="utf-8") as f:
        code = f.read()

    m = REPLACE_PATTERNS["init"][0].search(code)
    if m:
        v = m.groups()[0]
        return packaging.version.parse(v)

    # Fallback: looser search
    loose = re.compile(r'__version__\s*=\s*["\']([^"\']+)["\']')
    m2 = loose.search(code)
    if m2:
        return packaging.version.parse(m2.group(1))

    raise RuntimeError(f"Could not determine version from {REPLACE_FILES['init']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update_version", help="Update version number.")
    parser.add_argument("--show_version", action="store_true", help="Print current package version")
    parser.add_argument("--publish", action="store_true", help="Build and upload to PyPI via twine (requires twine)")
    args = parser.parse_args()

    if args.show_version:
        print(get_version())

    if args.update_version:
        global_version_update(version=args.update_version)
        print("Version files updated.")

    if args.publish:
        # Build and upload
        import subprocess

        # Ensure version argument was provided or current version is used
        try:
            version_to_publish = args.update_version or str(get_version())
        except Exception as e:
            raise RuntimeError("Specify --update_version or ensure package __init__ contains a valid version") from e

        print(f"Building distributions for version {version_to_publish}...")
        subprocess.run(["python", "-m", "build"], check=True)
        print("Uploading to PyPI via twine...")
        subprocess.run(["python", "-m", "twine", "upload", "dist/*"], check=True)