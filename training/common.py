from pathlib import Path
import json,os,copy,hashlib,unicodedata
R=Path(__file__).resolve().parent
MODEL=os.environ.get('MODEL_PATH','Qwen/Qwen3.5-9B')
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def normalize_messages(messages):
 messages=copy.deepcopy(messages)
 for m in messages:
  for tc in m.get('tool_calls') or []:
   args=tc['function'].get('arguments',{})
   if isinstance(args,str):tc['function']['arguments']=json.loads(args)
   parsed=tc['function'].get('arguments',{})
   if not isinstance(parsed,dict) or not isinstance(parsed.get('code'),str):raise ValueError('Tool arguments require a code string mapping')
 return messages

def encode(tok,row):
 prefix=tok.apply_chat_template(row['messages'],tools=row['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
 assert prefix.endswith('<think>\n'),repr(prefix[-100:])
 pre=tok.encode(prefix,add_special_tokens=False)
 # Keep the inference generation prefix intact: BPE can merge its final newline
 # with an empty-reasoning target's leading newline if tokenized jointly.
 target=tok.encode(row['completion'],add_special_tokens=False)
 ids=pre+target
 # Qwen normalizes canonically equivalent Unicode, e.g. y + combining diaeresis -> ÿ.
 decoded=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
 assert unicodedata.normalize('NFC',decoded)==unicodedata.normalize('NFC',prefix+row['completion']),('Decoded text mismatch',row['id'])
 assert len(ids)<=read(R/'config.json')['max_sequence_tokens'],('No truncation allowed',row['id'],len(ids))
 return dict(id=row['id'],ids=ids,labels=[-100]*len(pre)+ids[len(pre):],input_tokens=len(pre),target_tokens=len(ids)-len(pre))
