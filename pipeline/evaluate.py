"""Held-out tasks, identical tool loop, native Qwen tool serialization."""
import os,sys,json,asyncio,re,shutil,uuid,time
from pathlib import Path
R=Path(__file__).resolve().parent

def normalize(messages):
 messages=json.loads(json.dumps(messages))
 for m in messages:
  for tc in m.get('tool_calls') or []:
   if isinstance(tc['function'].get('arguments'),str):tc['function']['arguments']=json.loads(tc['function']['arguments'])
 return messages
async def main():
 import torch
 from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
 from peft import PeftModel
 import generate
 shard=int(sys.argv[1]);world=int(sys.argv[2]);dest=R/json.loads((R/'config.json').read_text()).get('evaluation_dir','evaluation3')/f'shard_{shard}';dest.mkdir(parents=True,exist_ok=True)
 tasks=json.loads((R/'evaluation_tasks.json').read_text())[shard::world]
 if not tasks:return
 for n in ['style.txt','draw_system.txt','plan_system.txt','agent_system.txt']:shutil.copy2(R/n,dest/n)
 (dest/'tasks.json').write_text(json.dumps(tasks));generate.R=dest
 tok=AutoTokenizer.from_pretrained(os.environ.get('MODEL_PATH','Qwen/Qwen3.5-9B'),local_files_only=True)
 model=Qwen3_5ForConditionalGeneration.from_pretrained(os.environ.get('MODEL_PATH','Qwen/Qwen3.5-9B'),dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',local_files_only=True)
 model=PeftModel.from_pretrained(model,Path(os.environ.get('ADAPTER_PATH',str(R.parent/'training/training/adapter'))));model.eval()
 async def local_post(payload):
  text=tok.apply_chat_template(normalize(payload['messages']),tools=payload.get('tools',[]),add_generation_prompt=True,enable_thinking=True,tokenize=False)
  x=tok(text,return_tensors='pt',add_special_tokens=False).to(model.device);budget=min(16000,32768-x.input_ids.shape[1]);assert budget>128,'Context exhausted'
  with torch.inference_mode():out=model.generate(**x,max_new_tokens=budget,do_sample=True,temperature=.6,top_p=.95,pad_token_id=tok.pad_token_id or tok.eos_token_id)
  ids=out[0,x.input_ids.shape[1]:];raw=tok.decode(ids,skip_special_tokens=False).replace('<|im_end|>','').strip()
  reasoning,content=raw.split('</think>',1) if '</think>' in raw else (raw,'')
  content=content.strip();calls=[]
  for block in re.findall(r'<tool_call>(.*?)</tool_call>',content,re.S):
   fn=re.search(r'<function=([^>]+)>(.*?)</function>',block,re.S);assert fn,'Bad Qwen tool function'
   args={k:v.strip() for k,v in re.findall(r'<parameter=([^>]+)>(.*?)</parameter>',fn.group(2),re.S)}
   calls.append(dict(id='call_'+uuid.uuid4().hex[:16],type='function',function=dict(name=fn.group(1),arguments=json.dumps(args))))
  visible=re.sub(r'<tool_call>.*?</tool_call>','',content,flags=re.S).strip()
  msg=dict(role='assistant',content=visible,reasoning_content=reasoning.strip())
  if calls:msg['tool_calls']=calls
  finish='length' if len(ids)>=budget else ('tool_calls' if calls else 'stop')
  return dict(model='Qwen3.5-9B-python1500-continuation-epoch1',choices=[dict(message=msg,finish_reason=finish)],usage=dict(prompt_tokens=x.input_ids.shape[1],completion_tokens=len(ids),completion_tokens_details=dict(reasoning_tokens=len(tok.encode(reasoning,add_special_tokens=False)))))
 generate.post=local_post
 # Sequential per GPU: one model and one generation at a time.
 # Override only task scheduling by using one task at a time, without changing parallel tool execution.
 for task in tasks:
  (dest/'tasks.json').write_text(json.dumps([task]));await generate.main()
 (dest/'tasks.json').write_text(json.dumps(tasks))
if __name__=='__main__':asyncio.run(main())
