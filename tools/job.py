"""Local Linux process jobs; PID/start ticks prevent duplicate or stale ownership."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evidence import write_json,utc_now,sha256


def ticks(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[21]
    except FileNotFoundError:return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['launch','status','worker'])
    p.add_argument('--name',required=True);p.add_argument('--dry-run',action='store_true')
    argv=sys.argv[1:];split=argv.index('--') if '--' in argv else len(argv)
    a=p.parse_args(argv[:split]);command=argv[split+1:]
    root=Path('outputs/jobs');root.mkdir(parents=True,exist_ok=True);path=root/(a.name+'.json')
    if a.action=='status':
        state=json.loads(path.read_text());state['alive']=ticks(state.get('pid',0))==state.get('start_ticks') and state.get('start_ticks') is not None
        print(json.dumps(state,indent=2));return
    if not command:p.error('Command required')
    if a.dry_run:print(json.dumps(dict(cwd=str(Path.cwd()),command=command,name=a.name)));return
    if a.action=='launch':
        if path.exists():raise FileExistsError('Job name already exists; inspect status and resume explicitly')
        # Refuse simultaneous project jobs, even if a new name was supplied.
        for old in root.glob('*.json'):
            s=json.loads(old.read_text())
            if s.get('start_ticks') is not None and ticks(s.get('pid',0))==s['start_ticks'] and s.get('ended_at') is None:
                raise RuntimeError(f'Active project job: {old}')
        with open(root/(a.name+'.launcher.log'),'xb') as log:
            child=subprocess.Popen([sys.executable,__file__,'worker','--name',a.name,'--',*command],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':child.pid,'job':a.name,'record':str(path)}));return
    record=dict(name=a.name,cwd=str(Path.cwd()),argv=command,pid=os.getpid(),start_ticks=ticks(os.getpid()),started_at=utc_now(),ended_at=None,exit_code=None)
    write_json(path,record);log_path=root/(a.name+'.log')
    with open(log_path,'xb',buffering=0) as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        record['child_pid']=child.pid;write_json(path,record);code=child.wait()
    record.update(ended_at=utc_now(),exit_code=code,log=str(log_path),log_sha256=sha256(log_path));write_json(path,record)
    sys.exit(code)


if __name__=='__main__':main()
