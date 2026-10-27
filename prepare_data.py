"""Verify release shards and reconstruct the exact prepared SFT rows."""
import gzip,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent

def main():
 manifest=json.loads((R/'data/manifest.json').read_text());tasks=json.loads((R/'data/tasks.json').read_text())
 assert len(tasks)==manifest['task_count']==2500
 by_id={t['id']:t for t in tasks};assert len(by_id)==len(tasks)
 prompts=[' '.join(t['prompt'].lower().split()) for t in tasks];assert len(set(prompts))==2500
 seen=set();counts={}
 for split in ['train','validation']:
  count=0
  with (R/'training'/f'{split}.jsonl').open('w') as out:
   for item in manifest['files']:
    if not item['path'].startswith(split+'-'):continue
    p=R/'data'/item['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],p
    with gzip.open(p,'rt') as f:
     for line in f:
      row=json.loads(line);assert row['id'] not in seen;seen.add(row['id'])
      assert by_id[row['task_id']]['split']==split
      assert row['messages'] and isinstance(row['completion'],str)
      out.write(line);count+=1
  counts[split]=count
 assert counts['train']==sum(v for k,v in manifest['rows'].items() if k.endswith('_train'))
 assert counts['validation']==sum(v for k,v in manifest['rows'].items() if k.endswith('_validation'))
 (R/'pipeline/tasks.json').write_text(json.dumps(tasks,ensure_ascii=False,indent=2))
 print(json.dumps(dict(tasks=len(tasks),rows=counts,checks='SHA256, task IDs, normalized prompts, split isolation and row IDs passed'),indent=2))
if __name__=='__main__':main()
