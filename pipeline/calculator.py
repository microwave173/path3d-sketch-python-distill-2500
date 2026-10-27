import ast, asyncio, json, sys
from pathlib import Path
WORKER = r'''
import ast, json, math, statistics, re, resource, sys
resource.setrlimit(resource.RLIMIT_CPU,(8,8))
if sys.platform == 'linux': resource.setrlimit(resource.RLIMIT_AS,(256*1024*1024,256*1024*1024))
code=sys.stdin.read()
tree=ast.parse(code)
allowed={'math':math,'json':json,'statistics':statistics,'re':re}
for n in ast.walk(tree):
 if isinstance(n,(ast.Import,ast.ImportFrom)):
  names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module]
  if any(x not in allowed for x in names): raise ValueError('Only math/json/statistics/re imports allowed')
 if isinstance(n,ast.Attribute) and n.attr.startswith('_'): raise ValueError('Private attributes disabled')
 if isinstance(n,ast.Name) and n.id.startswith('__'): raise ValueError('Private names disabled')
def limited_import(name,*args,**kwargs):return allowed[name]
import builtins
safe={k:getattr(builtins,k) for k in ['print','range','len','float','int','str','list','dict','tuple','set','abs','min','max','sum','round','enumerate','zip','sorted','reversed','all','any','bool','map','filter','format','isinstance','type','Exception','ValueError']}
safe['__import__']=limited_import
count=[0]
def bounded_print(*args,**kw):
 s=' '.join(str(a) for a in args)+'\n';count[0]+=len(s)
 if count[0]>24000:raise ValueError('Output exceeds 24000 characters; summarize results')
 sys.stdout.write(s)
safe['print']=bounded_print
exec(compile(tree,'calculation','exec'),{'__builtins__':safe,**allowed})
'''
async def calculate(code):
 if len(code)>20000:return {'ok':False,'error':'Code too long'}
 p=await asyncio.create_subprocess_exec(sys.executable,'-I','-S','-c',WORKER,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,env={},cwd='/tmp')
 try:
  out,err=await asyncio.wait_for(p.communicate(code.encode()),15)
  return {'ok':p.returncode==0,'stdout':out.decode()[:24000],'stderr':err.decode()[-3000:]}
 except asyncio.TimeoutError:
  p.kill();await p.wait();return {'ok':False,'error':'Calculation timeout'}
