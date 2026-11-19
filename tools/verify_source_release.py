"""CPU-only source release verification; does not import or execute model code."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'provenance/source_release_manifest.json').read_text(encoding='utf-8'))
failures = []
for rel, expected in manifest['files'].items():
    path = root / rel
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(rel)
parsed = 0
for folder in ('code', 'src', 'tools'):
    for path in (root / folder).rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        parsed += 1
print(json.dumps({'source_files': len(manifest['files']), 'python_files_parsed': parsed,
                  'hash_mismatches': failures, 'status': 'PASS' if not failures else 'FAIL'}, indent=2))
raise SystemExit(bool(failures))
