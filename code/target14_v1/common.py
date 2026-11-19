import json, hashlib, datetime, csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/home/ubuntu/MyDisk/data')
OTHER=Path('/home/ubuntu/MyDisk/VSI-AD-2/research_autorun/direct_20261017_v1')
RESULT=ROOT/'result'
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def csvwrite(path,rows,fields):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def read(path):return json.loads(Path(path).read_text())
