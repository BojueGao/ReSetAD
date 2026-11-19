"""Run an argv command and retain its real exit status and output."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evidence import run_recorded

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--name',required=True)
p.add_argument('--evidence-dir',default='outputs/evidence')
p.add_argument('--cwd',default='.')
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args()
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
if not cmd: p.error('A command is required')
r=run_recorded(cmd,a.cwd,a.evidence_dir,a.name)
print(r.stdout,end='');print(r.stderr,end='',file=sys.stderr)
sys.exit(r.returncode)
