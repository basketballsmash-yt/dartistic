"""Validated schema-2 recording I/O and a bounded callback-to-recorder queue."""
from copy import deepcopy
import json
import math
from pathlib import Path
import queue
import threading
from motion_stream import vector, release_event


def validate_header(value):
    if (value.get('type'),value.get('schema'),value.get('units'),value.get('coordinate_system')) != (
            'session',2,'meters','right-handed-y-up'):
        raise ValueError('Expected schema 2, meter units, and right-handed Y-up coordinates.')
    if not value.get('player') or value.get('hand') not in ('left','right'):
        raise ValueError('Missing player or invalid throwing hand.')


def validate_frame(frame):
    if frame.get('type')!='frame' or not isinstance(frame.get('frame_id'),int) or frame['frame_id']<0:
        raise ValueError('Invalid frame ID.')
    if not math.isfinite(frame['timestamp_s']): raise ValueError('Invalid frame timestamp.')
    if not isinstance(frame.get('joints'),dict): raise ValueError('Invalid joints.')
    for joint in frame['joints'].values():
        if type(joint.get('tracked')) is not bool: raise ValueError('Invalid tracking flag.')
        if joint.get('position') is not None: vector(joint['position'],3)
        elif joint['tracked']: raise ValueError('Tracked joint has no position.')
        if joint.get('rotation_xyzw') is not None:
            q=vector(joint['rotation_xyzw'],4)
            if abs(sum(v*v for v in q)-1)>.01: raise ValueError('Quaternion must be normalized.')
        c=joint.get('confidence')
        if c is not None and (not math.isfinite(c) or not 0<=c<=1): raise ValueError('Invalid confidence.')


def load_stream(path):
    records=[]
    with Path(path).open() as f:
        for i,line in enumerate(f,1):
            try: records.append(json.loads(line))
            except json.JSONDecodeError as e: raise ValueError(f'Invalid/incomplete JSON at line {i}') from e
    if not records: raise ValueError('Empty recording.')
    validate_header(records[0]); frames=[]; events=[]; summary=None
    for record in records[1:]:
        if summary is not None: raise ValueError('Data after stream summary.')
        if record.get('type')=='frame':
            validate_frame(record)
            if frames and (record['timestamp_s']<=frames[-1]['timestamp_s'] or record['frame_id']<=frames[-1]['frame_id']):
                raise ValueError('Frame timestamps and IDs must increase.')
            frames.append(record)
        elif record.get('type')=='event' and record.get('event')=='release':
            release_event(record['last_contact_s'],record['first_separated_s'],record.get('method'),record.get('sync_uncertainty_s'))
            events.append(record)
        elif record.get('type')=='summary': summary=record
        else: raise ValueError('Unknown stream record type.')
    return records[0],frames,events,summary


class QueuedRecorder:
    """SDK callback submits complete frames; disk writes occur on a dedicated worker.

    On overflow the new frame is rejected, and counted. close() drains accepted frames.
    Errors propagate to the producer/closer. SDK callbacks must be serialized.
    """
    def __init__(self,path,header,capacity=240):
        validate_header(header)
        if capacity<1: raise ValueError('Queue capacity must be positive.')
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        self.file=path.open('x'); self.file.write(json.dumps(header)+'\n'); self.file.flush()
        self.queue=queue.Queue(capacity); self.closing=threading.Event()
        self.error=None; self.accepted=0; self.overflow=0; self.written=0
        self.last=None
        self.worker=threading.Thread(target=self._write,daemon=True); self.worker.start()

    def submit(self,frame):
        if self.closing.is_set(): raise ValueError('Recorder is closed.')
        if self.error: raise RuntimeError('Recorder failed') from self.error
        validate_frame(frame)
        current=(frame['frame_id'],frame['timestamp_s'])
        if self.last and (current[0]<=self.last[0] or current[1]<=self.last[1]):
            raise ValueError('Duplicate/out-of-order source frame.')
        self.last=current
        try: self.queue.put_nowait(deepcopy(frame))
        except queue.Full: self.overflow+=1; return False
        self.accepted+=1; return True

    def _write(self):
        try:
            while not self.closing.is_set() or not self.queue.empty():
                try: frame=self.queue.get(timeout=.05)
                except queue.Empty: continue
                self.file.write(json.dumps(frame)+'\n'); self.file.flush(); self.written+=1
        except Exception as e: self.error=e

    def close(self):
        if self.closing.is_set(): return
        self.closing.set(); self.worker.join()
        try:
            if self.error: raise RuntimeError('Recorder failed') from self.error
            self.file.write(json.dumps(dict(type='summary',frames_written=self.written,queue_overflow=self.overflow))+'\n')
        finally: self.file.close()


def playback_projection(header,frames):
    """Display-only projection for the existing viewer. Never used as AI training data."""
    from motion_stream import JOINTS
    validate_header(header)
    result=[]
    for frame in frames:
        validate_frame(frame)
        points=[[0.,0.,0.] for _ in range(33)]; visibility=[0.]*33
        for index,name in JOINTS.items():
            joint=frame['joints'].get(name)
            if joint and joint['tracked'] and joint['position'] is not None:
                x,y,z=joint['position']; points[index]=[x,-y,-z]; visibility[index]=1.
        result.append(dict(t=frame['timestamp_s'],world=points if any(visibility) else None,visibility=visibility))
    return dict(schema=1,source=header['source'],player=header['player'],hand=header['hand'],
                outcome=header.get('outcome'),frames=result,display_only=True)
