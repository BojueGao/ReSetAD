"""Small, dependency-free evidence primitives. Never collect credentials."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_recorded(argv, cwd, evidence_dir, name, *, timeout=None, input_text=None):
    """No shell interpolation. Keep failed commands and their output."""
    out = Path(evidence_dir)
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / (name + '.json')
    if record_path.exists():
        raise FileExistsError(record_path)
    started = utc_now()
    t0 = time.monotonic()
    result = subprocess.run(argv, cwd=cwd, input=input_text, capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=timeout)
    log = out / (name + '.log')
    log.write_text(result.stdout + '\n[stderr]\n' + result.stderr, encoding='utf-8')
    record = dict(argv=argv, cwd=str(Path(cwd).resolve()), started_at=started,
                  ended_at=utc_now(), elapsed_seconds=time.monotonic()-t0,
                  exit_code=result.returncode, log=str(log), log_sha256=sha256(log))
    write_json(record_path, record)
    return result
