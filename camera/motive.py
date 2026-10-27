"""Live Motive/NatNet connection using the installed official Python SDK."""
import argparse
import builtins
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time
from motion_stream import DecodedNatNetAdapter, header
from stream_io import QueuedRecorder

DEFAULT_SDK = r'C:\Users\shuji\Downloads\NatNet_SDK_4.4_windows\NatNetSDK\Samples\PythonClient'


def name(value):
    return value.decode('utf-8',errors='replace') if isinstance(value,bytes) else str(value)


def describe(descriptions):
    return [dict(name=name(s.name),id=s.id_num,bones=[dict(name=name(b.sz_name),
                id=b.id_num,parent_id=b.parent_id,offset=list(b.pos))
                for b in s.rigid_body_description_list]) for s in descriptions.skeleton_list]


def canonical(bone_name):
    # Motive bone origins: upper arm = shoulder, forearm = elbow, hand = wrist.
    suffix=bone_name.split('_')[-1]
    return {'LArm':'left_shoulder','RArm':'right_shoulder',
            'LUArm':'left_shoulder','RUArm':'right_shoulder',
            'LFArm':'left_elbow','RFArm':'right_elbow',
            'LForeArm':'left_elbow','RForeArm':'right_elbow',
            'LHand':'left_wrist','RHand':'right_wrist',
            'LThigh':'left_hip','RThigh':'right_hip',
            'LShin':'left_knee','RShin':'right_knee',
            'LFoot':'left_ankle','RFoot':'right_ankle'}.get(suffix, 'bone:'+bone_name)


class Pipe:
    def __init__(self,args):
        self.args=args; self.descriptions=None; self.skeletons=[]
        self.frames=0; self.last=None; self.error=None; self.recorder=None; self.adapter=None
        self.lock=threading.Lock(); self.changed=False; self.tracked=0
        self.selected=None
        self.latest=None; self.last_received=None
        self.stop=threading.Event()

    def on_descriptions(self,value):
        with self.lock:
            self.descriptions=value; self.skeletons=describe(value)

    def on_frame(self,packet):
        try:
            with self.lock:
                self.frames+=1; self.last=packet['timestamp']
                self.last_received=time.monotonic()
                mocap=packet['mocap_data']
                self.tracked=sum(b.tracking_valid for s in mocap.skeleton_data.skeleton_list for b in s.rigid_body_list)
                if packet['tracked_models_changed'] and self.adapter:
                    raise ValueError('Motive asset definitions changed; restart the connection for a new session.')
                if not self.adapter: return
                skeleton=next((s for s in mocap.skeleton_data.skeleton_list if s.id_num==self.selected['id']),None)
                bodies=[]
                if skeleton:
                    for b in skeleton.rigid_body_list:
                        bodies.append(dict(skeleton_id=skeleton.id_num,bone_id=b.id_num & 0xffff,
                            position=list(b.pos),rotation_xyzw=list(b.rot),tracked=b.tracking_valid))
                frame=self.adapter.convert(packet['frame_number'],packet['timestamp'],bodies)
                self.latest=frame
                if self.recorder: self.recorder.submit(frame)
        except Exception as e:
            self.error=str(e)

    def start_recording(self):
        with self.lock:
            candidates=[s for s in self.skeletons if not self.args.skeleton or s['name']==self.args.skeleton]
            if len(candidates)!=1:
                raise ValueError('Select one skeleton with --skeleton. Available: '+', '.join(s['name'] for s in self.skeletons))
            self.selected=candidates[0]
            mapping={(self.selected['id'],b['id'] & 0xffff):canonical(b['name']) for b in self.selected['bones']}
            self.adapter=DecodedNatNetAdapter(mapping)
            h=header(self.args.player,'optitrack-natnet',datetime.now().strftime('%Y%m%d-%H%M%S'),self.args.hand)
            h.update(skeleton=self.selected,position_reference='Motive global bone origins',
                     network=dict(server=self.args.server,client=self.args.client,multicast=self.args.multicast),
                     calibration_note='Bone-origin anatomical mapping requires verification before AI training.')
            if self.args.output: self.recorder=QueuedRecorder(self.args.output,h)


def run(args,pipe=None):
    sdk=Path(args.sdk)
    if not (sdk/'NatNetClient.py').is_file(): raise ValueError('Official NatNet Python SDK not found: '+str(sdk))
    sys.path.insert(0,str(sdk))
    import NatNetClient as sdk_module
    # SDK 4.4 prints every frame even at print_level=0. Suppress that line only.
    sdk_module.print=lambda *v,**kw: None if v and str(v[0]).startswith('MoCap Frame:') else builtins.print(*v,**kw)
    NatNetClient=sdk_module.NatNetClient
    pipe=pipe or Pipe(args)
    class Client(NatNetClient):
        def _NatNetClient__unpack_data_descriptions(self,*values):
            offset, descriptions=super()._NatNetClient__unpack_data_descriptions(*values)
            pipe.on_descriptions(descriptions)
            return offset,descriptions
    client=Client(); client.set_server_address(args.server); client.set_client_address(args.client)
    client.set_use_multicast(args.multicast); client.set_print_level(0)
    client.new_frame_with_data_listener=pipe.on_frame
    start=time.monotonic(); connected=False; requested=False
    report_path=Path(args.status); report_path.parent.mkdir(parents=True,exist_ok=True)
    try:
        if not client.run('d'): raise ValueError('NatNet sockets could not open.')
        while not pipe.stop.is_set() and (args.seconds<=0 or time.monotonic()-start<args.seconds):
            if pipe.error: raise ValueError(pipe.error)
            connected=client.connected()
            if connected and not requested:
                print('CONNECTED:',client.get_application_name(),client.get_server_version(),flush=True)
                client.send_request(client.command_socket,client.NAT_REQUEST_MODELDEF,'',(args.server,client.command_port))
                requested=True
            if pipe.descriptions and not pipe.adapter: pipe.start_recording()
            if pipe.last_received and time.monotonic()-pipe.last_received>3:
                raise ValueError('No frames received for three seconds; Motive stream stopped.')
            status=dict(connected=connected,frames_received=pipe.frames,source_timestamp_s=pipe.last,
                        tracked_bones=pipe.tracked,skeletons=pipe.skeletons,
                        recording=str(args.output) if pipe.recorder else None,error=pipe.error,
                        selected_skeleton=pipe.selected['name'] if pipe.selected else None,
                        server_version=client.get_server_version(),natnet_version=client.get_nat_net_version_server())
            temp=report_path.with_suffix('.tmp'); temp.write_text(json.dumps(status,indent=2)); temp.replace(report_path)
            if pipe.latest:
                latest_path=Path(args.latest); latest_path.parent.mkdir(parents=True,exist_ok=True)
                temp=latest_path.with_suffix('.tmp'); temp.write_text(json.dumps(pipe.latest)); temp.replace(latest_path)
            if not connected and time.monotonic()-start>8: raise ValueError('No NatNet server response. Check Motive Streaming Enable/interface/transmission type.')
            time.sleep(.5)
        if not pipe.frames: raise ValueError('Server responded but no frame packets arrived.')
        print(f'RECEIVED {pipe.frames} frames, {pipe.tracked} tracked bones. Status: {report_path}',flush=True)
    finally:
        client.stop_threads=True
        for sock in (client.command_socket,client.data_socket):
            if sock: sock.close()
        for thread in (client.command_thread,client.data_thread):
            if thread and thread.is_alive(): thread.join(timeout=2)
        if pipe.recorder: pipe.recorder.close()
        if report_path.exists():
            status=json.loads(report_path.read_text())
            status.update(connected=False,error=pipe.error,stopped=True,frames_received=pipe.frames)
            temp=report_path.with_suffix('.tmp'); temp.write_text(json.dumps(status,indent=2)); temp.replace(report_path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',default=DEFAULT_SDK); p.add_argument('--server',default='127.0.0.1')
    p.add_argument('--client',default='127.0.0.1'); p.add_argument('--multicast',action='store_true')
    p.add_argument('--seconds',type=float,default=0); p.add_argument('--skeleton')
    p.add_argument('--player',default='player-1'); p.add_argument('--hand',choices=['left','right'],default='right')
    p.add_argument('--output'); p.add_argument('--status',default='work/motive-status.json')
    p.add_argument('--latest',default='work/motive-latest-frame.json')
    args=p.parse_args()
    try: run(args)
    except KeyboardInterrupt: print('Motive connection stopped.')
    except (ValueError,OSError) as e: p.exit(1,f'Error: {e}\n')


if __name__=='__main__': main()
