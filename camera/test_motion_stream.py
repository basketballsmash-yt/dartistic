import unittest
from motion_stream import DecodedNatNetAdapter, release_event


class StreamTests(unittest.TestCase):
    def test_identity_quality_and_dropped_frames(self):
        a=DecodedNatNetAdapter({(1,1):'right_wrist',(2,1):'left_wrist'})
        body=dict(skeleton_id=1,bone_id=1,tracked=True,position=[1,2,3],rotation_xyzw=[0,0,0,2])
        frame=a.convert(10,1,[body])
        self.assertEqual(frame['joints']['right_wrist']['rotation_xyzw'],[0,0,0,1])
        self.assertFalse(frame['joints']['left_wrist']['tracked'])
        self.assertIsNone(frame['joints']['right_wrist']['confidence'])
        self.assertEqual(a.convert(13,1.1,[])['dropped_before'],2)
        with self.assertRaises(ValueError): a.convert(12,1.2,[])

    def test_release_interval(self):
        e=release_event(1,1.002)
        self.assertIsNone(e['sync_uncertainty_s'])
        with self.assertRaises(ValueError): release_event(2,1)

    def test_invalid_frame_does_not_advance(self):
        a=DecodedNatNetAdapter({(1,1):'wrist'})
        with self.assertRaises(ValueError):
            a.convert(1,1,[dict(skeleton_id=1,bone_id=1,tracked=True,position=[0,0,0],rotation_xyzw=[0,0,0,0])])
        self.assertEqual(a.convert(1,1,[])['frame_id'],1)
