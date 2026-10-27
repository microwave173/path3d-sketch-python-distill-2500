import asyncio,json,time,sys,os,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
CODE=Path(os.getenv('PATH3D_CODE',str(R)))
sys.path.insert(0,str(CODE))
import httpx
from transport import post
CONFIG=json.loads((R/"config.json").read_text())
from geometry import parse_response,normalize_frame,render_frame
from PIL import Image,ImageDraw
from calculator import calculate
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def read(p):return json.loads(p.read_text())
async def main():
 style=(R/'style.txt').read_text();drawsys=style+'\n'+(R/'draw_system.txt').read_text()
 addon=(R/'agent_system.txt').read_text()
 drawsys+='\n'+addon
 states={};last_state_write=[0.0];save(R/'state.json',dict(status='running',tasks=states))
 async with httpx.AsyncClient(timeout=900,trust_env=False) as client:
  async def call(d,stage,system,user,thinking,validate):
   out=d/'calls'/stage;out.mkdir(parents=True,exist_ok=True)
   if (out/'accepted.json').exists() and (out/'main_output.json').exists():
    return validate((out/'main_output.json').read_text())
   recovery=d.name in CONFIG.get('recovery_task_ids',[]) or (d/'recovery_enabled.json').exists()
   budget=CONFIG.get('recovery_tool_budget',6) if recovery else 3
   max_turns=CONFIG.get('recovery_max_turns',16) if recovery else 8
   if recovery:system+=f'\nRecovery override: this stage allows up to {budget} native Python tool calls. Then finalize the requested JSON using observed results. Never emit DSML or tool markup in the answer content.'
   messages=[dict(role='system',content=system),dict(role='user',content=json.dumps(user,ensure_ascii=False))]
   schema=dict(type='function',function=dict(name='python_calculate',description='Batch numerical geometry calculations using Python math/json/statistics/re; map/filter/format and ordinary underscore variables are supported; print labeled results.',parameters=dict(type='object',properties=dict(code=dict(type='string')),required=['code'],additionalProperties=False)))
   tool_count=0;start=time.time();usages=[];budget_notice=False
   for turn in range(max_turns):
    if tool_count>=budget and not budget_notice:
     messages.append(dict(role='user',content='Python calls are now unavailable. Use the results already observed and return the final requested JSON only. Do not output DSML, XML, code blocks, or further tool requests.'))
     budget_notice=True
    payload=dict(model='deepseek-flash',thinking={'type':'enabled'},reasoning_effort='low',max_tokens=24000,messages=messages,tools=[schema],tool_choice='auto' if tool_count<budget else 'none')
    if tool_count>=budget:
     payload.pop('tools');payload.pop('tool_choice')
    a=out/f'turn_{turn:02d}';a.mkdir(exist_ok=True)
    if (a/'request.json').exists():assert read(a/'request.json')==payload,'Cache mismatch'
    save(a/'request.json',payload)
    if (a/'response.json').exists():b=read(a/'response.json')
    else:
     for retry in range(3):
      try:
       b=await post(payload);break
      except Exception:
       if retry==2:raise
       await asyncio.sleep(3*(retry+1))
     save(a/'response.json',b)
    choice=b['choices'][0];m=choice['message'];usages.append(b.get('usage',{}))
    (a/'content.txt').write_text(m.get('content') or '');(a/'reasoning.txt').write_text(m.get('reasoning_content') or '')
    
    if choice['finish_reason'] not in ('stop','tool_calls'):
     (a/'response.json').rename(a/f'truncated_{int(time.time())}.json')
     raise RuntimeError('Truncated response; response archived for fresh resampling')
    # Preserve native reasoning in assistant tool history for DeepSeek thinking tool use.
    assistant={k:m[k] for k in ['role','content','reasoning_content','tool_calls'] if k in m};messages.append(assistant)
    calls=m.get('tool_calls') or []
    if calls:
     async def execute(tc):
      nonlocal tool_count
      tool_count+=1
      cached=a/(tc['id']+'_result.json')
      if cached.exists():
       return dict(role='tool',tool_call_id=tc['id'],content=json.dumps(read(cached)))
      if tool_count>budget:result={'ok':False,'error':'Tool budget exhausted; finalize using existing results'}
      else:
       try:
        assert tc['function']['name']=='python_calculate'
        code=json.loads(tc['function']['arguments'])['code'];(a/(tc['id']+'.py')).write_text(code)
        result=await calculate(code)
       except Exception as e:result={'ok':False,'error':str(e)}
      save(a/(tc['id']+'_result.json'),result)
      return dict(role='tool',tool_call_id=tc['id'],content=json.dumps(result))
     messages.extend(await asyncio.gather(*(execute(tc) for tc in calls)))
     continue
    try:
     obj=validate(m.get('content') or '')
     (out/'main_output.json').write_text(m.get('content') or '')
     save(out/'accepted.json',dict(turns=turn+1,tool_calls=tool_count,usage=usages,seconds=round(time.time()-start,2),thinking=True,tool_budget=budget,max_turns=max_turns,recovery=recovery))
     return obj
    except Exception as e:
     save(a/'error.json',dict(error=str(e)))
     messages.append(dict(role='user',content='The final JSON failed validation: '+str(e)+'. Return the requested frame JSON, not a Python script or DSML/XML tool markup. C3 needs exactly 9 numbers, Q3 needs 6.'))
   
   for cached in sorted(out.glob('turn_*/response.json')):
    cached.rename(cached.with_name(f'rejected_{int(time.time())}.json'))
   raise RuntimeError(f'Stage exceeded {max_turns} model turns; rejected replies archived')
  def plan_parse(text):
   obj,_=parse_response(text);n=obj['total_frames'];assert type(n)==int and 5<=n<=9
   assert isinstance(obj['total_plan'],dict);return obj
  def frame_plan_parse(text):
   assert all(h in text for h in ['Design and rationale','Mathematical questions','Calculations and results','Checks and remaining uncertainties']);assert not text.lstrip().startswith('{');return text
  async def task(task):
   d=R/'results'/task['id'];d.mkdir(parents=True,exist_ok=True);save(d/'task.json',task);start=time.time()
   if (d/'result.json').exists() and read(d/'result.json').get('status')=='complete':
    states[task['id']]=read(d/'result.json');return
   def state(status,**kw):
    states[task['id']]=dict(status=status,**kw);
    if time.time()-last_state_write[0]>2:
     save(R/'state.json',dict(status='running',updated=time.time(),tasks=states));last_state_write[0]=time.time()
    print(task['id'],status,kw,flush=True)
   try:
    state('total_plan')
    total=await call(d,'00_total_plan',style+'\n'+(R/'plan_system.txt').read_text()+'\n'+addon,dict(animation_prompt=task['prompt']),True,plan_parse)
    save(d/'total_plan.json',total);n=total['total_frames'];frames=[]
    for i in range(1,n+1):
     # Only main total_plan and previous frame persist; explanations/reasoning never enter later calls.
     ctx=dict(animation_prompt=task['prompt'],total_frames=n,duration_seconds=2.5,total_plan=total['total_plan'],stage='FIRST' if i==1 else 'NEXT',current_frame_index=i,target_normalized_time=(i-1)/(n-1),target_time_seconds=2.5*(i-1)/(n-1),previous_frame=frames[-1] if frames else None)
     state('drawing',frame=i,total_frames=n)
     def parse_frame(text):
      obj,compat=parse_response(text);frame,notes=normalize_frame(obj,i);save(d/'frames'/f'{i:02d}_notes.json',dict(compat=compat,notes=notes));return frame
     frame=await call(d,f'{i:02d}_draw',drawsys,ctx,True,parse_frame)
     frames.append(frame);save(d/'frames'/f'{i:02d}.json',frame);await asyncio.to_thread(render_frame,frame,d)
    images=[Image.open(d/'frames'/f'{i:02d}.png').convert('RGB') for i in range(1,n+1)]
    images[0].save(d/'animation.gif',save_all=True,append_images=images[1:],duration=round(2500/n),loop=0)
    sheet=Image.new('RGB',(320*n,345),'white');dr=ImageDraw.Draw(sheet)
    for j,im in enumerate(images):sheet.paste(im.resize((320,320)),(320*j,25));dr.text((320*j+8,5),f'F{j+1}',fill='black')
    sheet.save(d/'contact_sheet.png');save(d/'result.json',dict(status='complete',frames=n,seconds=round(time.time()-start,1)));state('complete',frames=n)
   except Exception as e:save(d/'result.json',dict(status='failed',error=str(e)));state('failed',error=str(e))
  await asyncio.gather(*(task(t) for t in read(R/'tasks.json')))
 save(R/'state.json',dict(status='finished',updated=time.time(),tasks=states))
if __name__=='__main__':asyncio.run(main())
