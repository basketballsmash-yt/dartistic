import math
import unittest
from dartistic import summarize, outcome_label, training_value, player_id, require_player


class PlayerTests(unittest.TestCase):
    def test_matching_player(self):
        require_player({'player':'alice'}, 'alice')

    def test_wrong_or_missing_player_rejected(self):
        for data in ({'player':'bob'}, {}):
            with self.assertRaises(ValueError):
                require_player(data, 'alice')

    def test_profile_path_validation(self):
        self.assertEqual(player_id('player-1'), 'player-1')
        for value in ('../alice', 'a/b', '', 'Alice', 'a\\b'):
            with self.assertRaises(ValueError):
                player_id(value)


class OutcomeTests(unittest.TestCase):
    def test_hit_is_region_success(self):
        data = {'outcome': outcome_label('hit','T20','steel')}
        self.assertEqual(training_value(data,'hit'),1)
        self.assertEqual(training_value(data,'distance-cm'),0)
        self.assertIsNone(training_value(data,'distance-bits'))

    def test_unknown_distance_is_not_zero(self):
        data = {'outcome': outcome_label('miss','T20','steel')}
        self.assertEqual(training_value(data,'hit'),0)
        self.assertIsNone(training_value(data,'distance-cm'))

    def test_hole_counts_stay_separate(self):
        data = {'outcome': outcome_label('miss','T20','soft',3)}
        self.assertEqual(training_value(data,'distance-bits'),3)
        self.assertIsNone(training_value(data,'distance-cm'))

    def test_invalid_distances(self):
        for result,board,value in [('hit','steel',2),('miss','soft',1.5),
                                   ('miss','steel',-1),('miss','steel',float('nan')),
                                   ('miss','unspecified',2)]:
            with self.assertRaises(ValueError):
                outcome_label(result,'T20',board,value)


def recording():
    frames = []
    for i in range(30):
        points = [[0.,0.,0.] for _ in range(33)]
        points[12] = [0.,0.,0.]
        points[24] = [0.,.5,0.]
        points[14] = [.3,0.,0.]
        angle = .3+i*.04
        points[16] = [.3+.3*math.cos(angle), .3*math.sin(angle),0.]
        frames.append({'t':i/30, 'world':points, 'visibility':[1.]*33})
    return frames


class MotionTests(unittest.TestCase):
    def test_known_angle_and_stationary_elbow(self):
        result = summarize(recording(), 'right')
        self.assertAlmostEqual(result['elbow_max_deg'],180-math.degrees(.3))
        self.assertAlmostEqual(result['elbow_drift_body_lengths'],0)
        self.assertGreater(result['wrist_peak_body_lengths_s'],0)

    def test_translation_invariant(self):
        frames = recording()
        original = summarize(frames,'right')
        for f in frames:
            f['world'] = [[x+10,y-4,z+2] for x,y,z in f['world']]
        for k,v in original.items():
            self.assertAlmostEqual(v,summarize(frames,'right')[k])

    def test_tracking_loss_rejected(self):
        frames = recording()
        for f in frames[:10]:
            f['world'] = None
        with self.assertRaises(ValueError):
            summarize(frames,'right')

    def test_duplicate_time_rejected(self):
        frames = recording(); frames[2]['t'] = frames[1]['t']
        with self.assertRaises(ValueError):
            summarize(frames,'right')


if __name__ == '__main__':
    unittest.main()
