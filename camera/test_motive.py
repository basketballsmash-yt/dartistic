from types import SimpleNamespace as S
import unittest
from motive import Pipe, canonical, describe


class MotiveTests(unittest.TestCase):
    def test_actual_motive_bone_names(self):
        self.assertEqual(canonical('Skeleton 002_RUArm'),'right_shoulder')
        self.assertEqual(canonical('Skeleton 002_RFArm'),'right_elbow')
        self.assertEqual(canonical('Skeleton 002_RHand'),'right_wrist')
        self.assertEqual(canonical('Skeleton 002_RThumb1'),'bone:Skeleton 002_RThumb1')

    def test_description_ids_preserved(self):
        d=S(skeleton_list=[S(name=b'Player',id_num=3,rigid_body_description_list=[
            S(sz_name=b'Player_RHand',id_num=13,parent_id=12,pos=[0,0,0])])])
        self.assertEqual(describe(d)[0]['bones'][0]['id'],13)

    def test_packed_skeleton_bone_id(self):
        from motion_stream import DecodedNatNetAdapter
        p=Pipe(S()); p.selected={'id':3}; p.adapter=DecodedNatNetAdapter({(3,13):'right_wrist'})
        b=S(id_num=(3<<16)|13,pos=[1,2,3],rot=[0,0,0,1],tracking_valid=True)
        m=S(skeleton_data=S(skeleton_list=[S(id_num=3,rigid_body_list=[b])]))
        p.on_frame(dict(frame_number=1,timestamp=1.,mocap_data=m,tracked_models_changed=False))
        self.assertIsNone(p.error)
        self.assertTrue(p.latest['joints']['right_wrist']['tracked'])
