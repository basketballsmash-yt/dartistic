"""Timestamp-driven recording playback inside AI4AnimationPy's standalone renderer."""
import bisect
import json
import math
import os
from pathlib import Path
import sys
import time

EDGES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
         (23,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),
         (27,31),(28,30),(30,32),(28,32),(15,17),(15,19),(16,18),(16,20)]


class Timeline:
    def __init__(self, frames):
        if len(frames) < 2:
            raise ValueError('Replay needs at least two frames.')
        self.frames = frames
        self.times = [f['t']-frames[0]['t'] for f in frames]
        if not all(math.isfinite(t) for t in self.times) or any(b<=a for a,b in zip(self.times,self.times[1:])):
            raise ValueError('Replay timestamps must be finite and increasing.')
        self.duration = self.times[-1]

    def index(self, seconds):
        return max(0, min(len(self.frames)-1, bisect.bisect_right(self.times, seconds)-1))

    def pose(self, seconds):
        i = self.index(seconds)
        a = self.frames[i]
        if not a.get('world'):
            return None, []
        if i == len(self.frames)-1:
            return a['world'], a['visibility']
        b = self.frames[i+1]
        gap = self.times[i+1]-self.times[i]
        if not b.get('world') or gap > .15:
            return (a['world'], a['visibility']) if seconds-self.times[i] < .05 else (None, [])
        blend = max(0,min(1,(seconds-self.times[i])/gap))
        points = [[x+(y-x)*blend for x,y in zip(p,q)] for p,q in zip(a['world'],b['world'])]
        return points, [min(x,y) for x,y in zip(a['visibility'],b['visibility'])]


def launch(args):
    local = os.environ.get('AI4ANIMATION_PATH', r'C:\Users\shuji\OneDrive\Desktop\dartistic\ai4animationpy')
    sys.path.insert(0, local)
    from ai4animation import AI4Animation, Tensor
    import raylib as rl
    if Path(args.recording).suffix=='.jsonl':
        from stream_io import load_stream, playback_projection
        header,frames,events,summary=load_stream(args.recording)
        data=playback_projection(header,frames)
        print(f"Schema 2: {header['source']}, {len(frames)} frames, {len(events)} release events; {summary}",flush=True)
    else:
        data = json.loads(Path(args.recording).read_text())
    if data.get('schema') != 1:
        raise ValueError('Unsupported recording schema.')
    timeline = Timeline(data['frames'])
    if not any(f.get('world') for f in timeline.frames):
        raise ValueError('Recording has no tracked poses.')

    class Replay:
        def Start(self):
            # The stock scene inspectors otherwise draw over our playback controls.
            AI4Animation.Scene.GUI = lambda: None
            rl.SetWindowTitle(b'Dartistic - AI4AnimationPy Replay')
            rl.SetWindowSize(1280, 800)
            AI4Animation.Standalone.SetTargetFPS(60)
            self.position, self.speed, self.paused = 0., 1., False
            self.last = time.monotonic()
            self.angle = .45
            self.started = self.last
            first = next(f['world'] for f in timeline.frames if f.get('world'))
            self.height = max(p[1] for p in first)
            print('REPLAY_READY', flush=True)

        def Update(self):
            now = time.monotonic(); dt = now-self.last; self.last = now
            if args.smoke_seconds and now-self.started >= args.smoke_seconds:
                AI4Animation.Standalone.Exit()
            if rl.IsKeyPressed(rl.KEY_SPACE): self.paused = not self.paused
            if rl.IsKeyPressed(rl.KEY_R): self.position = 0.
            if rl.IsKeyPressed(rl.KEY_ONE): self.speed = .25
            if rl.IsKeyPressed(rl.KEY_TWO): self.speed = .5
            if rl.IsKeyPressed(rl.KEY_THREE): self.speed = 1.
            if rl.IsKeyPressed(rl.KEY_LEFT) or rl.IsKeyPressed(rl.KEY_RIGHT):
                step = -1 if rl.IsKeyPressed(rl.KEY_LEFT) else 1
                self.position = timeline.times[max(0,min(len(timeline.times)-1,timeline.index(self.position)+step))]
                self.paused = True
            if not self.paused:
                self.position = (self.position+dt*self.speed) % timeline.duration
            if rl.IsKeyDown(rl.KEY_A): self.angle -= dt
            if rl.IsKeyDown(rl.KEY_D): self.angle += dt
            camera = AI4Animation.Standalone.Camera
            camera.Mode = 1; camera.Target = None
            camera.Camera.position = (3*math.sin(self.angle), 1.5, 3*math.cos(self.angle))
            camera.Camera.target = (0, .9, 0)

        def Draw(self):
            points, visibility = timeline.pose(self.position)
            if points is None: return
            # MediaPipe: y down, z toward camera negative. Rotate 180 degrees around X.
            joints = Tensor.Create([[p[0], self.height-p[1], -p[2]] for p in points])
            arm = {12,14,16,18,20} if data['hand']=='right' else {11,13,15,17,19}
            for a,b in EDGES:
                if min(visibility[a],visibility[b]) < .65: continue
                color = (240,130,35,255) if a in arm and b in arm else (30,150,190,255)
                AI4Animation.Draw.Cylinder(joints[a],joints[b],.014,.014,color=color)
            for i in set(v for edge in EDGES for v in edge):
                if visibility[i]>=.65:
                    AI4Animation.Draw.Sphere(joints[i],size=.025,color=(35,60,85,255))

        def GUI(self):
            outcome = data.get('outcome') or {}
            lines = [f"Player: {data.get('player','unassigned')} | {data['hand']} hand | {data.get('source','camera')} | {outcome.get('board','')} | {outcome.get('target','')}",
                     f"{'Paused' if self.paused else 'Playing'}  {self.position:.2f} / {timeline.duration:.2f}s  {self.speed:g}x",
                     'SPACE pause | LEFT/RIGHT frame | R restart | 1/2/3 speed: .25/.5/1x | A/D rotate | ESC exit',
                     'Skeleton replay; orange = throwing arm. Synthetic sources are software tests only. Playback loops.']
            for i,line in enumerate(lines):
                rl.DrawRectangle(15,15+i*27,1210,26,(245,248,250,235))
                rl.DrawText(line.encode(),25,20+i*27,18,(25,40,55,255))

    AI4Animation(Replay(),mode=AI4Animation.Mode.STANDALONE)
