"""Versioned motion records and a decoded-NatNet adapter. No NatNet wire decoder."""
import argparse
import json
import math
from pathlib import Path
import time
from dartistic import player_id

JOINTS = {11:'left_shoulder',12:'right_shoulder',13:'left_elbow',14:'right_elbow',
          15:'left_wrist',16:'right_wrist',23:'left_hip',24:'right_hip',
          25:'left_knee',26:'right_knee',27:'left_ankle',28:'right_ankle'}


def vector(value, size):
    if len(value)!=size or not all(math.isfinite(v) for v in value):
        raise ValueError('Invalid vector.')
    return list(value)


def header(player, source, session, hand='right'):
    player_id(player)
    if hand not in ('right','left'): raise ValueError('Invalid throwing hand.')
    return dict(type='session',schema=2,player=player,hand=hand,source=source,session_id=session,
                units='meters',coordinate_system='right-handed-y-up',
                clock='source-seconds',events=[],
                release_definition='interval between last finger contact and first confirmed separation')


class DecodedNatNetAdapter:
    """SDK bridge must supply global, Y-up, right-handed poses in meters, xyzw rotations.

    mapping keys are (skeleton ID, bone ID) from model descriptions. Semantic joint
    placement must be calibrated; a bone origin is not automatically a joint center.
    """
    def __init__(self, mapping):
        if not mapping or len(set(mapping.values()))!=len(mapping):
            raise ValueError('Mapping must have unique joint names.')
        self.mapping=mapping
        self.last_id=None; self.last_time=None; self.dropped=0

    def convert(self, frame_id, timestamp_s, bodies, received_s=None):
        if not isinstance(frame_id,int) or frame_id<0 or not math.isfinite(timestamp_s):
            raise ValueError('Invalid source frame ID/timestamp.')
        if self.last_id is not None and (frame_id<=self.last_id or timestamp_s<=self.last_time):
            raise ValueError('Duplicate/out-of-order frame or source clock reset; start a new session.')
        joints={name:dict(position=None,rotation_xyzw=None,tracked=False,confidence=None)
                for name in self.mapping.values()}
        seen=set()
        for body in bodies:
            key=(body['skeleton_id'],body['bone_id'])
            if key not in self.mapping: continue
            if key in seen: raise ValueError('Duplicate bone in frame.')
            seen.add(key)
            if not body['tracked']: continue
            p=vector(body['position'],3); q=vector(body['rotation_xyzw'],4)
            norm=math.sqrt(sum(v*v for v in q))
            if norm<1e-8: raise ValueError('Invalid zero quaternion.')
            joints[self.mapping[key]]=dict(position=p,rotation_xyzw=[v/norm for v in q],
                                           tracked=True,confidence=None)
        gap=0 if self.last_id is None else frame_id-self.last_id-1
        self.dropped+=gap; self.last_id=frame_id; self.last_time=timestamp_s
        return dict(type='frame',frame_id=frame_id,timestamp_s=timestamp_s,
                    received_monotonic_s=time.monotonic() if received_s is None else received_s,
                    dropped_before=gap,joints=joints)


def release_event(last_contact_s, first_separated_s, method='manual-video',sync_uncertainty_s=None):
    if not all(math.isfinite(t) for t in (last_contact_s,first_separated_s)) or first_separated_s<last_contact_s:
        raise ValueError('Invalid release interval.')
    if sync_uncertainty_s is not None and (not math.isfinite(sync_uncertainty_s) or sync_uncertainty_s<0):
        raise ValueError('Invalid synchronization uncertainty.')
    return dict(type='event',event='release',last_contact_s=last_contact_s,
                first_separated_s=first_separated_s,method=method,sync_uncertainty_s=sync_uncertainty_s)


def simulate(args):
    from stream_io import QueuedRecorder
    if not 0<args.fps<=1000 or not 0<args.seconds<=3600:
        raise ValueError('Use fps 1-1000 and duration up to 3600 seconds.')
    mapping={(1,i):name for i,name in enumerate(JOINTS.values(),1)}
    adapter=DecodedNatNetAdapter(mapping)
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    recorder=QueuedRecorder(path,header(args.player,'simulated-optitrack','simulation'))
    try:
        for i in range(round(args.fps*args.seconds)):
            if args.drop_every and i and i%args.drop_every==0: continue
            t=i/args.fps
            if args.realtime: time.sleep(max(0,start+t-time.monotonic()))
            bodies=[]
            for (sid,bid),name in mapping.items():
                x=-.2 if name.startswith('left') else .2
                segment=name.split('_')[1]
                y={'shoulder':1.5,'elbow':1.25,'wrist':1.4,'hip':.95,'knee':.5,'ankle':.05}[segment]
                if segment=='elbow': x*=1.6
                if segment=='wrist': x*=2
                z=.25*math.sin(t*5) if segment=='wrist' else 0
                bodies.append(dict(skeleton_id=sid,bone_id=bid,position=[x,y,z],
                                   rotation_xyzw=[0,0,0,1],tracked=True))
            recorder.submit(adapter.convert(i,t,bodies))
    finally: recorder.close()
    print(f'Saved {path}; simulated missing frames: {adapter.dropped}. Synthetic data, not a Motive capture.')


def convert_camera(args):
    data=json.loads(Path(args.recording).read_text())
    if data.get('schema')!=1: raise ValueError('Expected schema 1 camera recording.')
    h=header(data['player'],'mediapipe',data.get('session_id') or 'legacy-camera',data['hand'])
    h['position_reference']='estimated hip-centered; not calibrated room coordinates'
    h['outcome']=data.get('outcome'); h['legacy_file']=str(args.recording)
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(h)+'\n')
        for i,row in enumerate(data['frames']):
            joints={}
            for index,name in JOINTS.items():
                valid=row.get('world') is not None
                p=vector(row['world'][index],3) if valid else None
                confidence=row['visibility'][index] if valid else None
                joints[name]=dict(position=[p[0],-p[1],-p[2]] if p else None,
                    rotation_xyzw=None,tracked=bool(valid and confidence>=.65),confidence=confidence)
            f.write(json.dumps(dict(type='frame',frame_id=i,timestamp_s=row['t'],
                                   received_monotonic_s=None,dropped_before=None,joints=joints))+'\n')
    print('Converted camera recording; original preserved:',path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(required=True)
    p=sub.add_parser('simulate'); p.add_argument('--player',default='test-player')
    p.add_argument('--output',required=True); p.add_argument('--fps',type=float,default=120)
    p.add_argument('--seconds',type=float,default=3); p.add_argument('--realtime',action='store_true')
    p.add_argument('--drop-every',type=int,default=0); p.set_defaults(run=simulate)
    p=sub.add_parser('convert-camera'); p.add_argument('recording'); p.add_argument('--output',required=True)
    p.set_defaults(run=convert_camera)
    args=parser.parse_args()
    try: args.run(args)
    except (ValueError,OSError) as e: parser.exit(1,f'Error: {e}\n')


if __name__=='__main__': main()
