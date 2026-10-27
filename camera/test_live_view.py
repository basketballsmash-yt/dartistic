import unittest
from live_view import segments


class HierarchyTests(unittest.TestCase):
    def test_parent_mapping_and_missing_bone(self):
        d={'bones':[{'id':11,'parent_id':0,'name':'S_RUArm'},
                    {'id':12,'parent_id':11,'name':'S_RFArm'},
                    {'id':13,'parent_id':12,'name':'S_RHand'}]}
        f={'joints':{'right_shoulder':{'position':[1,2,3],'tracked':True},
                     'right_elbow':{'position':[2,2,3],'tracked':True},
                     'right_wrist':{'position':None,'tracked':False}}}
        lines=segments(d,f)
        self.assertEqual(lines,[([1,2,3],[2,2,3],'S_RFArm')])

    def test_global_coordinates_remain_unchanged(self):
        d={'bones':[{'id':1,'parent_id':0,'name':'S_Hip'},
                    {'id':2,'parent_id':1,'name':'S_Ab'}]}
        f={'joints':{'bone:S_Hip':{'position':[-3,1,8],'tracked':True},
                     'bone:S_Ab':{'position':[-3,1.3,8],'tracked':True}}}
        self.assertEqual(segments(d,f)[0][0],[-3,1,8])
