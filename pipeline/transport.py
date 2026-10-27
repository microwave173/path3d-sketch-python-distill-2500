"""One process-wide gate shared by teacher generation and compression."""
import asyncio,json,time,random,resource,os
from pathlib import Path
import httpx
R=Path(__file__).resolve().parent
CONFIG=json.loads((R/'config.json').read_text())
CAP=CONFIG['api_concurrency']
LIMIT=asyncio.Semaphore(CAP)
CLIENT=None
try:stats=json.loads((R/'api_state.json').read_text())
except (OSError,json.JSONDecodeError):stats={}
for key in ['peak_inflight','requests','responses','errors']:stats.setdefault(key,0)
stats.update(inflight=0,limit=CAP,session_peak=0,session_started_at=time.time())
stats.setdefault('error_counts',{})
def persist():
 p=R/'api_state.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(stats,updated=time.time())));t.replace(p)
async def post(payload):
 global CLIENT
 if CLIENT is None:
  soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(16384,hard),hard))
  CLIENT=httpx.AsyncClient(timeout=httpx.Timeout(900,connect=30,pool=900),limits=httpx.Limits(max_connections=CAP,max_keepalive_connections=CAP,keepalive_expiry=60),trust_env=False)
 for retry in range(30):
  try:
   async with LIMIT:
    stats['inflight']+=1;stats['session_peak']=max(stats['session_peak'],stats['inflight']);stats['peak_inflight']=max(stats['peak_inflight'],stats['inflight']);stats['requests']+=1
    if stats['requests']%50==0:persist()
    try:
     key=os.environ['DEEPSEEK_API_KEY'].strip()
     res=await CLIENT.post(os.getenv('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+key},json=payload)
     res.raise_for_status();b=res.json();assert b.get('choices'),'No choices'
     stats['responses']+=1;return b
    finally:stats['inflight']-=1
  except Exception as e:
   stats['errors']+=1
   status=e.response.status_code if isinstance(e,httpx.HTTPStatusError) else None
   kind=str(status) if status else type(e).__name__
   stats['error_counts'][kind]=stats['error_counts'].get(kind,0)+1
   stats['last_error']=dict(kind=kind,at=time.time())
   persist()
   if status in (400,404,422):raise
   # Billing/authorization can recover after replenishment or key replacement.
   if status in (401,402,403):delay=300
   else:delay=min(120,3*2**min(retry,5))+random.random()*5
   if retry==29:raise
   await asyncio.sleep(delay)
async def close():
 if CLIENT is not None:await CLIENT.aclose()
 persist()
