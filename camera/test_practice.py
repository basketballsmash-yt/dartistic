import json
import tempfile
import unittest
from practice import PracticeSession
from test_dartistic import recording


class PracticeTests(unittest.TestCase):
    def test_repeated_throws_saved_separately(self):
        with tempfile.TemporaryDirectory() as root:
            s=PracticeSession(root,'player-1','right','steel','T20')
            for result,distance in [('hit',None),('miss',1.5)]:
                s.start()
                for f in recording(): s.add(f)
                s.stop()
                self.assertTrue(s.path.exists())
                self.assertNotIn('outcome',json.loads(s.path.read_text()))
                s.finish(result,distance)
                self.assertEqual(s.state,'ready')
            self.assertEqual(len(list(s.folder.glob('*.json'))),2)
            self.assertEqual((s.hits,s.misses),(1,1))

    def test_bad_tracking_and_exclusion_retained(self):
        with tempfile.TemporaryDirectory() as root:
            s=PracticeSession(root,'player-1','right','steel','T20')
            s.start(); s.add({'t':0,'world':None,'visibility':[]}); s.stop()
            self.assertFalse(s.data['quality']['accepted'])
            s.finish()
            self.assertTrue(json.loads(s.path.read_text())['excluded'])

    def test_invalid_distance_preserves_pending_throw(self):
        with tempfile.TemporaryDirectory() as root:
            s=PracticeSession(root,'player-1','right','soft','T20')
            s.start()
            for f in recording(): s.add(f)
            s.stop()
            with self.assertRaises(ValueError): s.finish('miss',1.5)
            self.assertEqual(s.state,'label')
            self.assertNotIn('outcome',json.loads(s.path.read_text()))
