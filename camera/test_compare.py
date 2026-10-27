import unittest
from compare import motion_anchor, shared_interval
from replay import Timeline


class CompareTests(unittest.TestCase):
    def test_anchor_tracks_motion_not_recording_start(self):
        frames=[]
        for i in range(15):
            p=[[0.,0.,0.] for _ in range(33)]
            p[16]=[max(0,min(3,i-5)),0,0]
            frames.append({'t':10+i*.1,'world':p,'visibility':[1.]*33})
        anchor=motion_anchor({'frames':frames,'hand':'right'})
        self.assertGreater(anchor,.5)
        self.assertLess(anchor,.85)

    def test_common_interval_avoids_out_of_range_playback(self):
        timeline=Timeline([{'t':0},{'t':3}])
        a={'anchor':1.,'timeline':timeline}
        b={'anchor':2.,'timeline':timeline}
        self.assertEqual(shared_interval(a,b),(-1.,1.))
        self.assertEqual(shared_interval(a,b,.5),(-1.,.5))
