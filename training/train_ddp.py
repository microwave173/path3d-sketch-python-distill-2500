"""Three-rank BF16 LoRA. Exact global target-token normalization; resumable checkpoints."""
from common import *
import argparse,math,random,time,traceback,shutil,contextlib
from datetime import timedelta

def main():
 import torch,torch.distributed as dist
 from torch.nn.parallel import DistributedDataParallel as DDP
 from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
 from peft import LoraConfig,get_peft_model,PeftModel
 from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--smoke-dir',default='smoke');args=ap.parse_args()
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
 torch.cuda.set_device(local);dist.init_process_group('nccl',device_id=torch.device('cuda',local),timeout=timedelta(minutes=90))
 cfg=read(R/'config.json');dest=R/(args.smoke_dir if args.smoke else 'training');dest.mkdir(exist_ok=True)
 if rank==0:
  import fcntl
  run_lock=(dest/'run.lock').open('a');fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 dist.barrier()
 def write(name,obj):
  if rank==0:save(dest/name,obj)
 state={'status':'loading','pid':os.getpid(),'world_size':world,'smoke':args.smoke}
 if rank==0:
  if not (dest/'started.json').exists():save(dest/'started.json',dict(started_epoch=time.time(),model=MODEL,epochs=cfg['epochs'],world_size=world))
  state['started_epoch']=read(dest/'started.json')['started_epoch']
 write('state.json',state);write('config_used.json',cfg)
 torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 source={s:[json.loads(l) for l in (R/(s+'.jsonl')).read_text().splitlines()] for s in ['train','validation']}
 if args.smoke:source={s:sorted(v,key=lambda r:sum(r['token_counts'].values()),reverse=True)[:cfg.get('smoke_train_rows',192) if s=='train' else cfg.get('smoke_validation_rows',12)] for s,v in source.items()}
 data={s:[encode(tok,r) for r in rows] for s,rows in source.items()};del source
 write('tokenization.json',{'counts':{s:len(v) for s,v in data.items()},'max_sequence':max(len(r['ids']) for v in data.values() for r in v),'train_mean_tokens':sum(len(r['ids']) for r in data['train'])/len(data['train']),'prompt_masked':True,'truncated':0})
 model=Qwen3_5ForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={'':local},attn_implementation='sdpa',local_files_only=True)
 checkpoints=dest/'checkpoints';valid=[]
 for c in checkpoints.glob('step_*'):
  if (c/'progress.json').exists() and (c/'optimizer.pt').exists() and (c/'adapter/adapter_model.safetensors').exists():valid.append((read(c/'progress.json')['step'],c))
 ck=max(valid,key=lambda x:x[0])[1] if valid else None;resume=read(ck/'progress.json') if ck else None
 if not args.smoke and cfg.get('require_resume_checkpoint'):assert ck is not None,'Required migration checkpoint missing; refusing to restart training from scratch'
 if resume:model=PeftModel.from_pretrained(model,ck/'adapter',is_trainable=True)
 elif cfg.get('initial_adapter'):
  source=Path(cfg['initial_adapter']);assert (source/'adapter_model.safetensors').exists(),'Predecessor final adapter missing'
  adapter_config=read(source/'adapter_config.json');assert adapter_config['r']==cfg['lora_rank'] and adapter_config['lora_alpha']==cfg['lora_alpha']
  model=PeftModel.from_pretrained(model,source,is_trainable=True)
  write('initialization.json',dict(source='previous_sft_adapter',adapter=str(source),base_model=MODEL,lora_rank=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],preexisting_adapter=True,source_config_sha256=hashlib.sha256((source/'adapter_config.json').read_bytes()).hexdigest()))
 else:
  targets=[n for n,m in model.named_modules() if n.startswith('model.language_model.') and isinstance(m,torch.nn.Linear)]
  write('initialization.json',{'source':'base_model_fresh_lora','base_model':MODEL,'lora_rank':cfg['lora_rank'],'lora_alpha':cfg['lora_alpha'],'preexisting_adapter':False})
  model=get_peft_model(model,LoraConfig(r=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],lora_dropout=0,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
 model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
 params=[p for p in model.parameters() if p.requires_grad];base=model.get_base_model();pad=tok.pad_token_id or tok.eos_token_id
 class LossModule(torch.nn.Module):
  def __init__(self):
   super().__init__();self.peft=model;self.ce=LigerFusedLinearCrossEntropyLoss(reduction='mean')
  def forward(self,rows):
   width=math.ceil(max(len(r['ids']) for r in rows)/cfg.get('pad_to_multiple_of',256))*cfg.get('pad_to_multiple_of',256);device=torch.device('cuda',local)
   ids=torch.tensor([r['ids']+[pad]*(width-len(r['ids'])) for r in rows],device=device)
   mask=torch.tensor([[1]*len(r['ids'])+[0]*(width-len(r['ids'])) for r in rows],device=device)
   labels=torch.tensor([r['labels']+[-100]*(width-len(r['ids'])) for r in rows],device=device)[:,1:].contiguous().view(-1)
   hidden=base.model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state[:,:-1,:].contiguous().view(-1,base.config.text_config.hidden_size)
   return self.ce(base.lm_head.weight,hidden,labels)
 wrapped=DDP(LossModule(),device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
 opt=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=.01)
 optimizer_offset=0
 if resume:
  opt.load_state_dict(torch.load(ck/'optimizer.pt',map_location=torch.device('cuda',local),weights_only=True))
  optimizer_steps={int(v['step'].item()) if hasattr(v['step'],'item') else int(v['step']) for v in opt.state.values() if 'step' in v}
  assert optimizer_steps=={resume.get('optimizer_step',resume['step'])},('Optimizer step mismatch',optimizer_steps,resume['step'])
  optimizer_offset=resume.get('optimizer_step',resume['step'])-resume['step']
  state.update(resume_from_step=resume['step'],resume_from_checkpoint=str(ck),optimizer_restored=True)
  write('resume.json',dict(step=resume['step'],checkpoint=str(ck),optimizer_steps=sorted(optimizer_steps),previous_world_size=resume.get('world_size'),world_size=world))
  if rank==0:print('RESUME_OPTIMIZER',resume['step'],'WORLD',world,flush=True)
 elif cfg.get('initial_checkpoint_manifest'):
  previous=read(cfg['initial_checkpoint_manifest']);assert previous['epoch']==1
  opt.load_state_dict(torch.load(previous['optimizer'],map_location=torch.device('cuda',local),weights_only=True))
  steps={int(v['step'].item()) if hasattr(v['step'],'item') else int(v['step']) for v in opt.state.values() if 'step' in v}
  assert steps=={previous['step']},('Predecessor optimizer mismatch',steps,previous['step'])
  optimizer_offset=previous['step'];state.update(optimizer_restored=True,predecessor_optimizer_step=optimizer_offset)
  write('predecessor.json',dict(adapter=cfg['initial_adapter'],checkpoint_manifest=cfg['initial_checkpoint_manifest'],optimizer=previous['optimizer'],optimizer_step=optimizer_offset,new_data_start_step=0,learning_rate_schedule='new-stage warmup/cosine; optimizer moments preserved'))
 micro=cfg['micro_batch'];effective=micro*cfg['gradient_accumulation']*world
 if args.smoke:effective=cfg.get('smoke_effective_batch',effective)
 def chunks(rows):
  offset=0
  while offset<len(rows):
   size=min(micro,len(rows)-offset)
   while size>1 and size*math.ceil(max(len(r['ids']) for r in rows[offset:offset+size])/cfg.get('pad_to_multiple_of',256))*cfg.get('pad_to_multiple_of',256)>cfg['max_micro_tokens']:size-=1
   yield rows[offset:offset+size];offset+=size
 schedule=[]
 for ep in range(cfg['epochs']):
  rr=list(data['train']);rng=random.Random(cfg['seed']+ep);rng.shuffle(rr);ordered=[]
  for i in range(0,len(rr),256):ordered+=sorted(rr[i:i+256],key=lambda r:len(r['ids']))
  batches=[ordered[i:i+effective] for i in range(0,len(ordered),effective)]
  if len(batches)>1 and len(batches[-1])<world:batches[-2].extend(batches.pop())
  rng.shuffle(batches);schedule.extend((ep+1,x) for x in batches)
 if args.smoke:schedule=schedule[:3]
 total=len(schedule);start_step=resume['step'] if resume else 0
 def validate(full=False):
  wrapped.eval();rows=(data['validation'] if full else data['validation'][:cfg['validation_monitor_rows']])[rank::world];stats=torch.zeros(2,device=local,dtype=torch.float64)
  with torch.no_grad():
   for mb in chunks(rows):
    nt=sum(r['target_tokens'] for r in mb);stats[0]+=wrapped.module(mb).double()*nt;stats[1]+=nt
  dist.all_reduce(stats);wrapped.train();return {'target_token_ce':float(stats[0]/stats[1]),'target_tokens':int(stats[1]),'n':len(data['validation']) if full else min(cfg['validation_monitor_rows'],len(data['validation']))}
 state.update(status='validation_before',step=start_step,total_steps=total,trainable_parameters=sum(p.numel() for p in params),effective_batch=effective);write('state.json',state)
 need_before=torch.tensor(int(not (dest/'validation_before.json').exists()) if rank==0 else 0,device=local);dist.broadcast(need_before,src=0)
 if int(need_before):write('validation_before.json',validate())
 start=time.monotonic();previous=resume.get('elapsed_seconds',0) if resume else 0;torch.cuda.reset_peak_memory_stats();wrapped.train()
 for ix,(ep,batch) in enumerate(schedule):
  if ix<start_step:continue
  tick=time.monotonic();opt.zero_grad(set_to_none=True);nt=sum(r['target_tokens'] for r in batch);sub=list(chunks(batch[rank::world]));assert sub
  lr=cfg['learning_rate']*min(1,(ix+1)/max(4,int(total*.03)))*(.2+.8*.5*(1+math.cos(math.pi*ix/total)))
  for group in opt.param_groups:group['lr']=lr
  value=torch.zeros((),device=local)
  for j,mb in enumerate(sub):
   weight=sum(r['target_tokens'] for r in mb)/nt
   with (wrapped.no_sync() if j+1<len(sub) else contextlib.nullcontext()):
    v=wrapped(mb);assert torch.isfinite(v);(v*weight*world).backward();value+=v.detach()*weight
  norm=float(torch.nn.utils.clip_grad_norm_(params,1));assert math.isfinite(norm) and norm>0
  opt.step()
  if ix==start_step:
   sample=params[-1].detach().flatten()[:64].float().contiguous();copies=[torch.zeros_like(sample) for _ in range(world)];dist.all_gather(copies,sample)
   delta=max(float((v-copies[0]).abs().max()) for v in copies);assert delta<1e-6,('DDP adapter weights diverged',delta)
   state['weight_sync_max_delta']=delta
  dist.all_reduce(value);elapsed=previous+time.monotonic()-start
  mem=torch.tensor(torch.cuda.max_memory_allocated()/2**30,device=local);dist.all_reduce(mem,op=dist.ReduceOp.MAX)
  record={'step':ix+1,'optimizer_step':optimizer_offset+ix+1,'total_steps':total,'epoch':ep,'world_size':world,'effective_batch':effective,'loss':float(value),'grad_norm':norm,'lr':lr,'seconds':time.monotonic()-tick,'elapsed_seconds':elapsed,'peak_gpu_GiB':float(mem),'eta_seconds':(time.monotonic()-start)/max(1,ix+1-start_step)*(total-ix-1)}
  state.update(record,status='training');write('state.json',state)
  if rank==0:
   with (dest/'train_log.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   print('STEP',json.dumps(record),flush=True)
  epoch_end=(ix+1==total or schedule[ix+1][0]!=ep)
  pinned_epoch=(not args.smoke and epoch_end and ep in cfg.get('keep_epoch_checkpoints',[]))
  if not args.smoke and ((ix+1)%cfg['checkpoint_every_steps']==0 or ix+1==total or pinned_epoch):
   if rank==0:
    checkpoints.mkdir(exist_ok=True);temp=checkpoints/f'.step_{ix+1:06d}.tmp';final=checkpoints/f'step_{ix+1:06d}'
    if temp.exists():shutil.rmtree(temp)
    temp.mkdir();model.save_pretrained(temp/'adapter');torch.save(opt.state_dict(),temp/'optimizer.pt');save(temp/'progress.json',record);temp.rename(final)
    if pinned_epoch:
     (final/'KEEP').write_text(f'User requested full epoch {ep} checkpoint; do not prune.\n')
     save(dest/f'epoch_{ep}_checkpoint.json',dict(epoch=ep,step=ix+1,path=str(final),adapter=str(final/'adapter'),optimizer=str(final/'optimizer.pt'),progress=str(final/'progress.json'),pinned=True))
    rolling=[p for p in sorted(checkpoints.glob('step_*')) if not (p/'KEEP').exists()]
    for p in (rolling if pinned_epoch else rolling[:-1]):shutil.rmtree(p)
   dist.barrier()
  if ix+1==total or schedule[ix+1][0]!=ep:
   write(f'validation_epoch_{ep}.json',validate())
   if rank==0 and not args.smoke:
    alias=dest/f'epoch_{ep}_adapter'
    if not alias.exists():alias.symlink_to(checkpoints/f'step_{ix+1:06d}'/'adapter',target_is_directory=True)
   dist.barrier()
   if not args.smoke and ep in cfg.get('epoch_probe_epochs',[]):
    state.update(status='epoch_probe',probe_epoch=ep);write('state.json',state)
    try:
     from epoch_probe import run_epoch_probe
     run_epoch_probe(model,tok,rank,world,ep,R)
    except Exception as exc:
     write(f'epoch_{ep}_probe_failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
    dist.barrier();state.update(status='training');write('state.json',state)
 if rank==0:
  model.save_pretrained(dest/'adapter')
  from safetensors import safe_open
  n_params=n_tensors=0
  with safe_open(dest/'adapter/adapter_model.safetensors',framework='pt',device='cpu') as handle:
   for key in handle.keys():
    tensor=handle.get_tensor(key);assert torch.isfinite(tensor).all(),key;n_params+=tensor.numel();n_tensors+=1
  save(dest/'adapter_check.json',{'all_finite':True,'parameters':n_params,'tensors':n_tensors,'lora_rank':cfg['lora_rank']})
 dist.barrier();write('validation_after.json',validate(full=True));state.update(status='complete',training_seconds=previous+time.monotonic()-start,adapter=str(dest/'adapter'),completed_epoch=time.time())
 if rank==0:state['total_wall_seconds']=state['completed_epoch']-state['started_epoch']
 write('state.json',state);dist.destroy_process_group()
if __name__=='__main__':
 try:main()
 except Exception as e:
  if os.environ.get('RANK','0')=='0':save(R/'training_failure.json',{'error':str(e),'traceback':traceback.format_exc()})
  raise
