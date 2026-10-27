"""One target assistant turn per row, native Qwen tools, previous targets masked."""
from common import *
import statistics,collections,sys

def main():
 from transformers import AutoTokenizer
 source=Path(sys.argv[1]) if len(sys.argv)>1 else R.parent
 preflight='--preflight' in sys.argv
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 task_map={t['id']:t for t in read(R.parent/'pipeline/tasks.json')}
 if preflight:
  task_map={p.parent.name:{'split':'validation' if i%5==0 else 'train','category':'preflight'} for i,p in enumerate(sorted((source/'results').glob('*/task.json')))}
 dest=R/'preflight' if preflight else R;dest.mkdir(exist_ok=True)
 files={s:(dest/(s+'.jsonl.tmp')).open('w') for s in ['train','validation']}
 totals=collections.Counter();task_counts=collections.Counter();lengths=[];dropped=[];source_errors=[];invalid_targets=[];serialization_excluded=[]
 for task_index,(task_id,task) in enumerate(task_map.items(),1):
  d=source/'results'/task_id
  if not (d/'result.json').exists() or read(d/'result.json').get('status')!='complete':source_errors.append(task_id);continue
  task_counts[task['split']]+=1
  for accepted in sorted((d/'calls').glob('*/accepted.json')):
   history=None
   for f in sorted(accepted.parent.glob('turn_*/response.json')):
    req=read(f.parent/'request.json');b=read(f);m=b['choices'][0]['message']
    if (f.parent/'error.json').exists():
     invalid_targets.append(str(f));continue
    assert b['choices'][0]['finish_reason'] in ('stop','tool_calls')
    rel=f.parent.relative_to(source/'results');c=source/'compression'/rel;metric=read(c/'metrics.json');original=m.get('reasoning_content') or ''
    assert metric['ok'] and metric['source_sha256']==hashlib.sha256(original.encode()).hexdigest()
    compressed=(c/'compressed.txt').read_text();sha=hashlib.sha256(compressed.encode()).hexdigest()
    # Use exact actual request history. Replace earlier assistant reasoning with its compressed version.
    try:
     messages=normalize_messages(req['messages']);target=normalize_messages([m])[0]
    except (json.JSONDecodeError,ValueError,TypeError) as e:
     serialization_excluded.append(dict(id=task_id+'/'+accepted.parent.name+'/'+f.parent.name,error=str(e)));continue
    prior=iter(sorted(accepted.parent.glob('turn_*/response.json')))
    for a in messages:
     if a['role']=='assistant':
      pf=next(prior);pd=source/'compression'/pf.parent.relative_to(source/'results');a['reasoning_content']=(pd/'compressed.txt').read_text()
    target['reasoning_content']=compressed
    prefix=tok.apply_chat_template(messages,tools=req.get('tools',[]),tokenize=False,add_generation_prompt=True,enable_thinking=True)
    full=tok.apply_chat_template(messages+[target],tools=req.get('tools',[]),tokenize=False,add_generation_prompt=False,enable_thinking=True)
    assert full.startswith(prefix),(task_id,str(f),'Chat template mismatch')
    completion=full[len(prefix):];assert completion.endswith('<|im_end|>\n')
    row=dict(id=task_id+'/'+accepted.parent.name+'/'+f.parent.name,task_id=task_id,category=task['category'],stage=accepted.parent.name,messages=messages,tools=req.get('tools',[]),completion=completion,provenance=dict(response_path=str(f),original_reasoning_sha256=metric['source_sha256'],compressed_reasoning_sha256=sha,assistant_content_sha256=hashlib.sha256((m.get('content') or '').encode()).hexdigest(),tool_target=bool(m.get('tool_calls'))))
    n=len(tok.encode(prefix,add_special_tokens=False))+len(tok.encode(completion,add_special_tokens=False))
    if n>read(R/'config.json')['max_sequence_tokens']:
     dropped.append(dict(id=row['id'],tokens=n,reason='over_context_no_truncation'));continue
    enc=encode(tok,row);row['token_counts']={k:enc[k] for k in ['input_tokens','target_tokens']};lengths.append(n);totals[task['split']]+=1;totals['tool_rows' if m.get('tool_calls') else 'final_rows']+=1
    files[task['split']].write(json.dumps(row,ensure_ascii=False)+'\n')
  if task_index%25==0:
   save(dest/'prepare_state.json',dict(tasks_processed=task_index,tasks_total=len(task_map),rows=dict(totals),serialization_excluded=len(serialization_excluded),updated=__import__('time').time()))
   print('PREPARED_TASKS',task_index,'/',len(task_map),flush=True)
 for s,f in files.items():f.close();(dest/(s+'.jsonl.tmp')).replace(dest/(s+'.jsonl'))
 assert not source_errors,('Incomplete teacher tasks',source_errors[:10])
 assert totals['train'] and totals['validation'] and totals['tool_rows']
 report=dict(tasks=len(task_map),task_split=dict(task_counts),rows=dict(totals),max_sequence_tokens=max(lengths),mean_sequence_tokens=statistics.mean(lengths),overlength_excluded=dropped,invalid_format_targets_excluded=invalid_targets,invalid_tool_serialization_excluded=serialization_excluded,truncated=0,target='compressed reasoning + visible content/tool_calls; only current assistant turn unmasked',tool_serialization='Qwen native chat template; function.arguments parsed into mappings',generation_effort='low',compression_effort='low',base_init=read(R/'config.json')['initialization'])
 save(dest/'DATA_READY.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
