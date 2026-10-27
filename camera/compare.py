"""Personal throw comparison in the AI4AnimationPy renderer."""
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from dartistic import load_throw, player_id, distance
from replay import Timeline, EDGES


def motion_anchor(data):
    """Peak of a short averaged wrist-speed trace; a motion cue, not dart release."""
    frames = data['frames']
    s,w = (12,16) if data['hand']=='right' else (11,15)
    speeds=[]
    for a,b in zip(frames,frames[1:]):
        dt=b['t']-a['t']
        if not a.get('world') or not b.get('world') or not 0<dt<=.15:
            continue
        if min(a['visibility'][s],a['visibility'][w],b['visibility'][s],b['visibility'][w])<.65:
            continue
        pa=[a['world'][w][i]-a['world'][s][i] for i in range(3)]
        pb=[b['world'][w][i]-b['world'][s][i] for i in range(3)]
        speeds.append(((a['t']+b['t'])/2-frames[0]['t'],distance(pa,pb)/dt))
    if not speeds: raise ValueError('No reliable wrist motion to align.')
    smoothed=[(t,statistics.mean(v for u,v in speeds if abs(u-t)<=.065)) for t,_ in speeds]
    return max(smoothed,key=lambda x:x[1])[0]


def shared_interval(left, right, offset=0):
    # right anchor + offset is paired with left anchor.
    return max(-left['anchor'],-right['anchor']-offset), min(
        left['timeline'].duration-left['anchor'],right['timeline'].duration-right['anchor']-offset)


def launch(args):
    player_id(args.player)
    records=[]; skipped=0
    for path in sorted((Path('profiles')/args.player/'recordings').glob('*.json')):
        try:
            data,metrics=load_throw(path)
            if data.get('player')!=args.player or data.get('excluded'): continue
            records.append(dict(path=path,data=data,metrics=metrics,timeline=Timeline(data['frames']),anchor=motion_anchor(data)))
        except (ValueError,KeyError): skipped+=1
    if len(records)<2: raise ValueError('Need two usable recordings for this player.')
    sys.path.insert(0,os.environ.get('AI4ANIMATION_PATH',r'C:\Users\shuji\OneDrive\Desktop\dartistic\ai4animationpy'))
    from ai4animation import AI4Animation, Tensor
    import raylib as rl

    class Compare:
        def Start(self):
            # Scene.GUI runs after Program.GUI and overlays the review with editor panels.
            # Hide inspectors only on this viewer's scene instance; keep rendering/update intact.
            AI4Animation.Scene.GUI = lambda: None
            rl.SetWindowTitle(b'Dartistic - Personal Throw Review')
            rl.SetWindowSize(1400,900); AI4Animation.Standalone.SetTargetFPS(60)
            self.indices=[next((i for i,r in enumerate(records) if r['data'].get('outcome',{}).get('hit') is True),0),
                          next((i for i,r in enumerate(records) if r['data'].get('outcome',{}).get('hit') is False),1)]
            self.offset=0.; self.position=0.; self.paused=False; self.speed=.5
            self.last=self.started=time.monotonic()
            print(f'COMPARE_READY: {len(records)} throws; {skipped} invalid recordings skipped',flush=True)

        def selected(self): return [records[i] for i in self.indices]

        def cycle(self,side,delta):
            self.indices[side]=(self.indices[side]+delta)%len(records)
            self.offset=0.; self.position=0.

        def Update(self):
            now=time.monotonic(); dt=now-self.last; self.last=now
            if args.smoke_seconds and now-self.started>args.smoke_seconds: AI4Animation.Standalone.Exit()
            if rl.IsKeyPressed(rl.KEY_Q): self.cycle(0,-1)
            if rl.IsKeyPressed(rl.KEY_W): self.cycle(0,1)
            if rl.IsKeyPressed(rl.KEY_O): self.cycle(1,-1)
            if rl.IsKeyPressed(rl.KEY_P): self.cycle(1,1)
            if rl.IsKeyPressed(rl.KEY_SPACE): self.paused=not self.paused
            if rl.IsKeyPressed(rl.KEY_ONE): self.speed=.25
            if rl.IsKeyPressed(rl.KEY_TWO): self.speed=.5
            if rl.IsKeyPressed(rl.KEY_THREE): self.speed=1.
            if rl.IsKeyPressed(rl.KEY_J): self.offset-=1/30
            if rl.IsKeyPressed(rl.KEY_L): self.offset+=1/30
            if rl.IsKeyPressed(rl.KEY_R): self.offset=0.; self.position=0.
            left,right=self.selected()
            self.offset=max(-right['anchor'],min(right['timeline'].duration-right['anchor'],self.offset))
            lo,hi=shared_interval(left,right,self.offset)
            if rl.IsKeyPressed(rl.KEY_LEFT): self.position-=1/30; self.paused=True
            if rl.IsKeyPressed(rl.KEY_RIGHT): self.position+=1/30; self.paused=True
            if not self.paused: self.position+=dt*self.speed
            if self.position>hi: self.position=lo
            self.position=max(lo,min(hi,self.position))
            camera=AI4Animation.Standalone.Camera
            camera.Mode=1; camera.Target=None
            camera.Camera.position=(0,1.8,5.2); camera.Camera.target=(0,.85,0)

        def Draw(self):
            for side,r in enumerate(self.selected()):
                t=self.position+r['anchor']+(self.offset if side else 0)
                points,visibility=r['timeline'].pose(t)
                if points is None: continue
                first=next(f['world'] for f in r['data']['frames'] if f.get('world'))
                height=max(p[1] for p in first)
                joints=Tensor.Create([[p[0]+(-.85 if side==0 else .85),height-p[1],-p[2]] for p in points])
                arm={12,14,16,18,20} if r['data']['hand']=='right' else {11,13,15,17,19}
                for a,b in EDGES:
                    if min(visibility[a],visibility[b])<.65: continue
                    color=(240,135,35,255) if a in arm and b in arm else (35,155,185,255)
                    AI4Animation.Draw.Cylinder(joints[a],joints[b],.014,.014,color=color)
                for i in set(v for e in EDGES for v in e):
                    if visibility[i]>=.65: AI4Animation.Draw.Sphere(joints[i],size=.023,color=(35,60,80,255))

        def text(self,s,x,y,size=18): rl.DrawText(s.encode('ascii','replace'),x,y,size,(25,40,55,255))

        def GUI(self):
            width,height=rl.GetScreenWidth(),rl.GetScreenHeight()
            rl.DrawRectangle(0,0,width,180,(245,248,250,245))
            self.text(f'{args.player} | Personal throw review | {len(records)} recordings',20,15,23)
            self.text('Q/W: previous/next LEFT throw     O/P: previous/next RIGHT throw',20,48)
            self.text('SPACE pause | arrows: step 1/30s | 1/2/3: .25/.5/1x | J/L: adjust right alignment | R reset | ESC exit',20,74)
            self.text(f'{"Paused" if self.paused else "Playing"} {self.speed:g}x | relative time {self.position:+.2f}s | right adjustment {self.offset:+.2f}s',20,100)
            self.text('Aligned to peak averaged wrist speed, NOT detected dart release. Adjust if the wrong motion was selected.',20,126)
            self.text('Camera-estimated skeletons. Differences describe these throws; they do not establish causes of hits/misses.',20,151)
            for side,r in enumerate(self.selected()):
                x=20+side*(width//2)
                d=r['data']; o=d.get('outcome',{})
                result='HIT' if o.get('hit') is True else ('MISS' if o.get('hit') is False else 'UNLABELED')
                rl.DrawRectangle(x-5,190,width//2-20,58,(245,248,250,240))
                self.text(f'{"LEFT" if side==0 else "RIGHT"} {self.indices[side]+1}/{len(records)} | {result} | {o.get("target", "?")} | {o.get("board", "?")}',x,195)
                self.text(r['path'].name,x,222,16)
            left,right=self.selected()
            rl.DrawRectangle(0,height-170,width,170,(245,248,250,245))
            self.text('Full-recording measurements                    LEFT       RIGHT       RIGHT - LEFT',20,height-160)
            keys=[('elbow_max_deg','Max elbow angle (degrees)'),('elbow_range_deg','Elbow angle range (degrees)'),
                  ('elbow_drift_body_lengths','Elbow drift (torso lengths)'),('wrist_peak_body_lengths_s','Peak wrist speed (torso lengths/s)')]
            for i,(key,label) in enumerate(keys):
                a,b=left['metrics'][key],right['metrics'][key]
                self.text(f'{label:<43} {a:8.2f}    {b:8.2f}    {b-a:+8.2f}',20,height-130+i*26)
            a,b=left['data'],right['data']
            if a['hand']!=b['hand'] or a.get('outcome',{}).get('target')!=b.get('outcome',{}).get('target') or a.get('outcome',{}).get('board')!=b.get('outcome',{}).get('board'):
                self.text('Different hand/target/board: interpret this comparison cautiously.',20,height-22,16)

    AI4Animation(Compare(),mode=AI4Animation.Mode.STANDALONE)
