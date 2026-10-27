"""Base Qwen3.5-9B: shared plan, then previous-frame-only Path3D generation."""
from pathlib import Path
import argparse, fcntl, hashlib, json, math, os, subprocess, sys, tarfile, time, traceback
from PIL import Image, ImageDraw
from path3d import Path3DScene, parse_path3d
from path3d.renderer import DEFAULT_CAMERAS, render_view, view_metrics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
MODEL=os.environ.get('MODEL_PATH','Qwen/Qwen3.5-9B')

def read(path):return json.loads(path.read_text())
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)

def parse_response(text):
    """Format compatibility only: extract JSON or close unmatched delimiters."""
    original=text.strip();start=original.find('{')
    if start<0:raise ValueError('No JSON object in model answer')
    text=original[start:]
    try:
        obj,end=json.JSONDecoder().raw_decode(text)
        return obj, [] if original==text[:end] else ['Extracted JSON object; trailing/prefix text retained in raw answer']
    except json.JSONDecodeError:
        pass
    # Only append missing final braces/brackets; do not rewrite numbers, strings or geometry.
    candidate=text.strip()
    if candidate.endswith('```'):candidate=candidate[:-3].rstrip()
    stack=[];quoted=False;escaped=False
    for ch in candidate:
        if quoted:
            if escaped:escaped=False
            elif ch=='\\':escaped=True
            elif ch=='"':quoted=False
        elif ch=='"':quoted=True
        elif ch in '{[':stack.append('}' if ch=='{' else ']')
        elif ch in '}]':
            if not stack or stack.pop()!=ch:raise ValueError('Mismatched JSON delimiters')
    if quoted or not stack:raise ValueError('Malformed JSON beyond final delimiter compatibility')
    suffix=''.join(reversed(stack))
    return json.loads(candidate+suffix),['Appended missing final JSON delimiters: '+suffix]

def frame_payload(task,plan,index,previous):
    assert (previous is None)==(index==1)
    if previous is not None:assert previous['i']==index-1
    return dict(animation_prompt=task['prompt'],total_frames=5,duration_seconds=2,
        total_plan=plan,stage='FIRST' if index==1 else 'NEXT',current_frame_index=index,
        target_normalized_time=(index-1)/4,target_time_seconds=(index-1)/2,
        previous_frame=previous)

def normalize_frame(obj,index):
    assert isinstance(obj.get('frames'),list) and len(obj['frames'])==1,'Expected exactly one frame'
    frame=obj['frames'][0];notes=[]
    if frame.get('i')!=index:notes.append(dict(kind='index_metadata',original=frame.get('i'),requested=index))
    strokes=[];ids=set()
    for n,source in enumerate(frame.get('strokes',[])):
        path=source.get('path');assert isinstance(path,str) and path.strip(),'Missing 3D path'
        if ';' in path:
            path=path.replace(';',' ')
            notes.append(dict(kind='command_separator_compatibility',stroke=n,change='semicolon to whitespace; coordinates unchanged'))
        commands=parse_path3d(path)
        assert all(math.isfinite(v) for c in commands for v in c.values),'Nonfinite 3D coordinate'
        if any(abs(v)>1 for c in commands for v in c.values):notes.append(dict(kind='coordinate_outside_requested_range',stroke=n))
        sid=str(source.get('id') or f'stroke_{n+1}')
        if sid in ids:
            original=sid;sid=f'{sid}__{n+1}';notes.append(dict(kind='duplicate_id_metadata',original=original,assigned=sid))
        ids.add(sid)
        strokes.append(dict(id=sid,path=path,description=str(source.get('description') or sid)))
    assert strokes,'No visible 3D strokes'
    normalized=dict(i=index,strokes=strokes)
    Path3DScene.from_dict(normalized)
    return normalized,notes

def render_frame(frame,directory):
    scene=Path3DScene.from_dict(frame)
    path=directory/'frames'/f"{frame['i']:02d}.png"
    camera=next(c for c in DEFAULT_CAMERAS if c.name=='perspective')
    render_view(scene,camera,path,width=640,height=640,normalize=False)
    return view_metrics(scene,camera,width=640,height=640,normalize=False)

def previews(directory,frames):
    ordered=sorted(frames)
    images=[Image.open(directory/'frames'/f'{i:02d}.png').convert('RGB') for i in ordered]
    if len(images)==6:
        images[0].save(directory/'animation.gif',save_all=True,append_images=images[1:],duration=500,loop=0,disposal=2)
    if images:
        sheet=Image.new('RGB',(320*len(images),350),'white');draw=ImageDraw.Draw(sheet)
        for n,(i,image) in enumerate(zip(ordered,images)):
            sheet.paste(image.resize((320,320),Image.Resampling.LANCZOS),(320*n,28));draw.text((320*n+8,8),f'F{i}',fill='black')
        sheet.save(directory/'contact_sheet.png')
        # The same model coordinates are reprojected, never reconstructed from a perspective image.
        diagnostics=Image.new('RGB',(3*256,2*280),'white');draw=ImageDraw.Draw(diagnostics)
        for row,i in enumerate([ordered[0],ordered[-1]]):
            scene=Path3DScene.from_dict(frames[i])
            for col,camera in enumerate(DEFAULT_CAMERAS[:3]):
                path=directory/f'{camera.name}_F{i:02d}.png'
                render_view(scene,camera,path,width=256,height=256,normalize=False)
                with Image.open(path) as im:diagnostics.paste(im,(col*256,row*280+24))
                draw.text((col*256+5,row*280+5),f'F{i}: {camera.name}',fill='black')
        diagnostics.save(directory/'spatial_check.png')

