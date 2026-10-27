"""Direct full-rate NatNet feed into AI4AnimationPy; global Motive hierarchy preserved."""
import argparse
import math
import os
from pathlib import Path
import sys
import threading
import time
from motive import Pipe,run,canonical,DEFAULT_SDK


def segments(description,frame):
    """Use Motive's parent IDs, without fabricating joints or changing coordinates."""
    bones={b['id'] & 0xffff:b for b in description['bones']}
    joints=frame['joints']; lines=[]
    for bid,b in bones.items():
        parent=bones.get(b['parent_id'] & 0xffff)
        if parent is None or parent is b: continue
        a=joints.get(canonical(parent['name'])); c=joints.get(canonical(b['name']))
        if a and c and a['tracked'] and c['tracked']:
            lines.append((a['position'],c['position'],b['name']))
    return lines


def launch(args):
    pipe=Pipe(args)
    def network():
        try: run(args,pipe)
        except Exception as e: pipe.error=str(e)
        finally: pipe.stop.set()
    thread=threading.Thread(target=network,daemon=True); thread.start()
    try:
        deadline=time.monotonic()+12
        while pipe.latest is None:
            if pipe.error: raise ValueError(pipe.error)
            if time.monotonic()>deadline: raise ValueError('No skeleton frame received within 12 seconds.')
            time.sleep(.05)
        sys.path.insert(0,os.environ.get('AI4ANIMATION_PATH',str(Path(__file__).resolve().parents[1]/'ai4animationpy')))
        from ai4animation import AI4Animation,Tensor
        import raylib as rl

        class Live:
            def Start(self):
                AI4Animation.Scene.GUI=lambda:None
                rl.SetWindowTitle(b'Dartistic - Live Motive Skeleton')
                rl.SetWindowSize(1280,800); AI4Animation.Standalone.SetTargetFPS(60)
                self.angle=.4; self.distance=3.; self.paused=False; self.axes=False
                self.frame=None; self.started=time.monotonic(); self.target=[0,1,0]
                self.screenshot_done=False
                print('LIVE_VIEW_READY',flush=True)

            def Update(self):
                now=time.monotonic()
                if args.smoke_seconds and now-self.started>args.smoke_seconds: AI4Animation.Standalone.Exit()
                if rl.IsKeyPressed(rl.KEY_SPACE): self.paused=not self.paused
                if rl.IsKeyPressed(rl.KEY_T): self.axes=not self.axes
                if not self.paused:
                    with pipe.lock: self.frame=pipe.latest
                dt=rl.GetFrameTime()
                if rl.IsKeyDown(rl.KEY_A): self.angle-=dt
                if rl.IsKeyDown(rl.KEY_D): self.angle+=dt
                if rl.IsKeyDown(rl.KEY_W): self.distance=max(1,self.distance-dt)
                if rl.IsKeyDown(rl.KEY_S): self.distance=min(8,self.distance+dt)
                points=[j['position'] for j in self.frame['joints'].values() if j['tracked']] if self.frame else []
                if points:
                    self.target=[sum(p[i] for p in points)/len(points) for i in range(3)]
                camera=AI4Animation.Standalone.Camera
                camera.Mode=1; camera.Target=None
                x,y,z=self.target
                camera.Camera.target=(x,y,z)
                camera.Camera.position=(x+self.distance*math.sin(self.angle),y+.5,z+self.distance*math.cos(self.angle))

            def Draw(self):
                # Hide stale motion rather than showing a frozen pose as live.
                if not self.frame or (not self.paused and (pipe.error or time.monotonic()-pipe.last_received>1)): return
                for a,b,n in segments(pipe.selected,self.frame):
                    color=(245,135,35,255) if '_R' in n else (30,150,190,255)
                    AI4Animation.Draw.Cylinder(Tensor.Create(a),Tensor.Create(b),.012,.012,color=color)
                for j in self.frame['joints'].values():
                    if not j['tracked']: continue
                    p=Tensor.Create(j['position'])
                    AI4Animation.Draw.Sphere(p,size=.022,color=(35,55,70,255))
                    if self.axes:
                        # Quaternion rotates local unit axes into Motive global coordinates.
                        x,y,z,w=j['rotation_xyzw']
                        axes=[(1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)),
                              (2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),
                              (2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y))]
                        for axis,color in zip(axes,[(220,40,40,255),(40,180,60,255),(40,70,220,255)]):
                            AI4Animation.Draw.Line(p,p+Tensor.Create(axis)*.08,color)

            def GUI(self):
                age=time.monotonic()-pipe.last_received if pipe.last_received else 999
                state='FROZEN' if self.paused else ('LIVE' if age<1 and not pipe.error else 'STREAM LOST')
                count=sum(j['tracked'] for j in self.frame['joints'].values()) if self.frame else 0
                width=rl.GetScreenWidth()
                rl.DrawRectangle(0,0,width,130,(245,248,250,245))
                lines=[f"{state} | {args.skeleton} | {count}/{len(pipe.selected['bones'])} tracked bones | frame {self.frame['frame_id'] if self.frame else '-'}",
                       'Motive global Y-up coordinates and parent hierarchy | orange: right-side bones',
                       'SPACE freeze/resume | A/D orbit | W/S zoom | T bone axes | ESC close',
                       pipe.error or 'Verify: move right arm, bend elbow, rotate wrist. Missing bones are hidden.']
                for i,line in enumerate(lines): rl.DrawText(line.encode('ascii','replace'),18,15+i*28,18,(25,40,55,255))
                if args.screenshot and not self.screenshot_done and time.monotonic()-self.started>1:
                    # Capture the previous completed frame for visual verification.
                    rl.TakeScreenshot(str(Path(args.screenshot).resolve()).encode()); self.screenshot_done=True
        AI4Animation(Live(),mode=AI4Animation.Mode.STANDALONE)
    finally:
        pipe.stop.set(); thread.join(timeout=6)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',default=DEFAULT_SDK);p.add_argument('--skeleton',default='Skeleton 002')
    p.add_argument('--server',default='127.0.0.1');p.add_argument('--client',default='127.0.0.1')
    p.add_argument('--multicast',action='store_true');p.add_argument('--player',default='player-1')
    p.add_argument('--hand',default='right');p.add_argument('--output')
    p.add_argument('--status',default='work/viewer-motive-status.json');p.add_argument('--latest',default='work/viewer-latest-frame.json')
    p.add_argument('--smoke-seconds',type=float,default=0);p.add_argument('--screenshot')
    args=p.parse_args();args.seconds=0
    try: launch(args)
    except (ValueError,OSError,ImportError) as e: p.exit(1,f'Error: {e}\n')


if __name__=='__main__':main()
