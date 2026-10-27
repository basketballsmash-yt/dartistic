import unittest
from replay import Timeline


class ReplayTests(unittest.TestCase):
    def frames(self):
        return [{'t':10.,'world':[[0,0,0]],'visibility':[1.]},
                {'t':10.1,'world':[[2,0,0]],'visibility':[.8]}]

    def test_interpolates_recorded_time(self):
        timeline = Timeline(self.frames())
        pose, visibility = timeline.pose(.05)
        self.assertAlmostEqual(pose[0][0],1.)
        self.assertEqual(visibility,[.8])
        self.assertAlmostEqual(timeline.duration,.1)

    def test_missing_pose_is_not_invented(self):
        frames = self.frames(); frames[0]['world'] = None
        self.assertIsNone(Timeline(frames).pose(.05)[0])

    def test_duplicate_timestamps_rejected(self):
        frames = self.frames(); frames[1]['t'] = 10.
        with self.assertRaises(ValueError): Timeline(frames)

    def test_does_not_bridge_long_gap(self):
        frames = self.frames(); frames[1]['t'] = 11.
        self.assertIsNone(Timeline(frames).pose(.5)[0])
