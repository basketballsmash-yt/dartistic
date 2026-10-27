"""Repeated throw capture with durable per-throw saves and keyboard labeling."""
import json
from pathlib import Path
import time
import uuid
from dartistic import outcome_label, player_id, summarize


class PracticeSession:
    def __init__(self, root, player, hand, board, target):
        player_id(player)
        outcome_label('hit', target, board)
        self.folder = Path(root) / 'profiles' / player / 'recordings'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(schema=1, source='mediapipe', player=player, hand=hand,
                             session_id=time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8],
                             intended_target=target.upper(), board=board)
        self.state = 'ready'
        self.frames = []
        self.saved = self.hits = self.misses = 0
        self.message = 'Ready for your first throw.'
        self.path = self.data = None

    def start(self):
        if self.state != 'ready': return
        self.frames = []
        self.state = 'recording'

    def add(self, frame):
        if self.state == 'recording': self.frames.append(frame)

    def stop(self):
        if self.state != 'recording': return
        if not self.frames:
            self.state = 'ready'; return
        self.data = dict(self.metadata, frames=self.frames)
        try:
            self.data['metrics'] = summarize(self.frames, self.metadata['hand'])
            self.data['quality'] = {'accepted':True}
        except ValueError as error:
            self.data['quality'] = {'accepted':False, 'reason':str(error)}
        self.path = self.folder / (time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]+'.json')
        with self.path.open('x') as f: json.dump(self.data,f)
        self.state = 'label'
        self.message = ('Throw saved. Label its result.' if self.data['quality']['accepted'] else
                        'Saved, but tracking failed: '+self.data['quality']['reason'])

    def finish(self, result=None, distance=None):
        if self.state not in ('label','distance'): return
        if result:
            self.data['outcome'] = outcome_label(result,self.metadata['intended_target'],self.metadata['board'],distance)
        else:
            self.data['excluded'] = True
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.data)); temporary.replace(self.path)
        self.saved += 1
        self.hits += int(result == 'hit'); self.misses += int(result == 'miss')
        self.message = f'Saved {self.path.name}. Ready for next throw.'
        self.state = 'ready'


def launch(args):
    import cv2
    import mediapipe as mp
    session = PracticeSession('.',args.player,args.hand,args.board,args.target)
    model = Path(args.pose_model)
    if not model.is_file():
        raise ValueError('Pose model missing. Run capture once to download it.')
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        raise ValueError('Cannot open camera. Close other camera apps or try --camera 1.')
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=1)
    start, last, entry = time.monotonic(), -1, ''
    window = 'Dartistic Practice'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window,1100,850)
    edges = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
             (23,24),(23,25),(25,27),(24,26),(26,28)]
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as tracker:
            print('PRACTICE_READY',flush=True)
            while True:
                ok, frame = cap.read()
                if not ok:
                    session.message = 'Camera disconnected. Any pending recording was saved.'
                    break
                stamp = max(last+1,round((time.monotonic()-start)*1000)); last=stamp
                result = tracker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),stamp)
                row = {'t':stamp/1000,'world':None,'visibility':[]}
                if result.pose_world_landmarks:
                    row['world'] = [[p.x,p.y,p.z] for p in result.pose_world_landmarks[0]]
                    points = result.pose_landmarks[0]
                    row['visibility'] = [p.visibility for p in points]
                    h,w = frame.shape[:2]
                    for a,b in edges:
                        if min(points[a].visibility,points[b].visibility)>=.65:
                            cv2.line(frame,(int(points[a].x*w),int(points[a].y*h)),
                                     (int(points[b].x*w),int(points[b].y*h)),(80,230,100),3)
                session.add(row)
                panel = cv2.copyMakeBorder(frame,180,0,0,max(0,1050-frame.shape[1]),cv2.BORDER_CONSTANT,value=(25,30,38))
                unit = 'cm' if args.board=='steel' else 'bits (board holes)'
                prompt = {'ready':'SPACE: start one throw | Q: end session',
                          'recording':'RECORDING - SPACE: stop after follow-through | Q: save and exit',
                          'label':'H: hit | M: miss | X: exclude throw | Q: exit (saved without label)',
                          'distance':f'Miss distance in {unit}: {entry}_ | ENTER: save (blank = unknown)'}[session.state]
                lines = [f'{args.player} | {args.hand} | {args.board} | Target: {args.target.upper()}',
                         f'Labeled hits: {session.hits} | misses: {session.misses} | state: {session.state}',
                         prompt,session.message[:115],
                         'Record ONE throw, stop, then label. Distance is from the target region, not its center.']
                for i,line in enumerate(lines):
                    cv2.putText(panel,line,(15,27+i*32),cv2.FONT_HERSHEY_SIMPLEX,.58,(235,240,245),1,cv2.LINE_AA)
                cv2.imshow(window,panel)
                key = cv2.waitKey(1)&255
                if key in (ord('q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1: break
                if session.state=='ready' and key==32: session.start()
                elif session.state=='recording' and key==32: session.stop()
                elif session.state=='label':
                    if key==ord('h'): session.finish('hit')
                    elif key==ord('x'): session.finish()
                    elif key==ord('m'):
                        # Persist the miss immediately, even if the session closes before distance entry.
                        session.data['outcome'] = outcome_label('miss',args.target,args.board)
                        temporary = session.path.with_suffix('.json.tmp')
                        temporary.write_text(json.dumps(session.data)); temporary.replace(session.path)
                        session.state='distance'; entry=''
                elif session.state=='distance':
                    if key in (10,13):
                        try: session.finish('miss',float(entry) if entry else None)
                        except ValueError as error: session.message=str(error)
                    elif key in (8,127): entry=entry[:-1]
                    elif chr(key) in '0123456789.' and len(entry)<10: entry+=chr(key)
    finally:
        cap.release(); cv2.destroyAllWindows()
        if session.state=='recording': session.stop()
        print(session.message)
        print('Session recordings:',session.folder.resolve())
