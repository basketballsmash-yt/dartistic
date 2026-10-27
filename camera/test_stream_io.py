import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import queue
from motion_stream import header, DecodedNatNetAdapter
from stream_io import QueuedRecorder, load_stream, playback_projection


class StreamIOTests(unittest.TestCase):
    def test_record_drain_read_and_display(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'stream.jsonl'
            h=header('test-player','simulated-optitrack','test')
            r=QueuedRecorder(path,h)
            a=DecodedNatNetAdapter({(1,1):'right_wrist'})
            body=dict(skeleton_id=1,bone_id=1,position=[1,2,3],rotation_xyzw=[0,0,0,1],tracked=True)
            for i in range(10): r.submit(a.convert(i,i/120,[body]))
            r.close()
            h,frames,events,summary=load_stream(path)
            self.assertEqual(len(frames),10)
            self.assertEqual(summary['frames_written'],10)
            data=playback_projection(h,frames)
            self.assertEqual(data['frames'][0]['world'][16],[1,-2,-3])
            self.assertEqual(data['frames'][0]['visibility'][12],0)
            self.assertTrue(data['display_only'])
            with self.assertRaises(ValueError): r.submit(frames[-1])

    def test_overflow_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'stream.jsonl'
            r=QueuedRecorder(path,header('test-player','simulated-optitrack','test'))
            frame=DecodedNatNetAdapter({(1,1):'wrist'}).convert(0,0,[])
            with patch.object(r.queue,'put_nowait',side_effect=queue.Full):
                self.assertFalse(r.submit(frame))
            r.close()
            self.assertEqual(load_stream(path)[3]['queue_overflow'],1)
