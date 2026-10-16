"""Check every packaged artifact against its recorded SHA-256, without ML dependencies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def verify_bundle():
    manifest = json.loads((ROOT/'checksums.json').read_text())
    for name, expected in manifest.items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f'Invalid bundle path: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Bundle integrity mismatch: {name}')
    print(f'Verified {len(manifest)} bundled file hashes.')
    return len(manifest)

if __name__ == '__main__':
    verify_bundle()
