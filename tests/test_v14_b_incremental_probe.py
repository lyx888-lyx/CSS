"""V14 synthetic CPU regression tests. Does not load IEMOCAP or TEST."""
import unittest

import numpy as np

from research.v14_b_incremental_probe import (
    SESSIONS, FEATURE_DIMS, make_projection, build_aba_examples,
    base_and_b_features, select_partition, shuffle_b_donors,
    evaluate_fold, subgroup_metrics, _project_utterances
)


class FakeOriginalTrain:
    """Every dialogue is A,B,A,B,A,B,A,B,A with known previous turns."""
    def __init__(self):
        self.keys=[]
        self.videoText={}
        self.videoAudio={}
        self.videoVisual={}
        self.videoSpeakers={}
        self.videoLabels={}
        for sess in range(1,5):
            for d in range(4):
                key=f"Ses0{sess}M_impro{d:02d}"
                self.keys.append(key)
                n=9
                turn=np.arange(n,dtype=np.float32)
                text=np.zeros((n,1024),dtype=np.float32)
                audio=np.zeros((n,1582),dtype=np.float32)
                visual=np.zeros((n,342),dtype=np.float32)
                text[:,0]=turn+sess*.25+d
                text[:,1]=(turn%3).astype(np.float32)
                audio[:,0]=np.sin(turn+sess+d)
                audio[:,1]=(turn%2).astype(np.float32)
                visual[:,0]=np.cos(turn+d)
                self.videoText[key]=text
                self.videoAudio[key]=audio
                self.videoVisual[key]=visual
                self.videoSpeakers[key]=[
                    "M" if i%2==0 else "F" for i in range(n)
                ]
                self.videoLabels[key]=[
                    (i+d+sess)%6 for i in range(n)
                ]


class TestV14Diagnostic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds=FakeOriginalTrain()
        cls.proj=make_projection(seed=23)
        cls.ex=build_aba_examples(
            cls.ds,strict=False,projection=cls.proj
        )

    def test_fixed_projection_dimension_and_determinism(self):
        a=make_projection(seed=23)
        b=make_projection(seed=23)
        for key,(raw,hidden) in FEATURE_DIMS.items():
            self.assertEqual(a[key].shape,(raw,hidden))
            np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(self.ex["current"].shape,(112,224))

    def test_aba_previous_and_intervening_other_are_exact(self):
        self.assertEqual(self.ex["eligible_examples"],112)
        self.assertEqual(self.ex["train_utterances"],144)
        self.assertEqual(self.ex["session_dialogue_counts"]["Ses01"],4)
        key=self.ds.keys[0]
        projected=_project_utterances(self.ds,key,self.proj)
        first=0
        self.assertEqual(self.ex["dialogue"][first],key)
        self.assertEqual(int(self.ex["turn_index"][first]),2)
        self.assertEqual(int(self.ex["previous_index"][first]),0)
        self.assertEqual(self.ex["b_role"][first],"F")
        np.testing.assert_allclose(
            self.ex["current"][first],projected[2],atol=1e-6
        )
        np.testing.assert_allclose(
            self.ex["previous"][first],projected[0],atol=1e-6
        )
        np.testing.assert_allclose(
            self.ex["other"][first],projected[1],atol=1e-6
        )

    def test_true_emotion_label_change_is_never_a_model_input(self):
        ds=FakeOriginalTrain()
        before=build_aba_examples(ds,strict=False,projection=self.proj)
        # Change ONLY previous A's gold label. The input design
        # must stay unchanged because labels are not model inputs.
        key=ds.keys[0]
        ds.videoLabels[key][0]=(ds.videoLabels[key][0]+1)%6
        after=build_aba_examples(ds,strict=False,projection=self.proj)
        for name in ("current","previous","other"):
            np.testing.assert_array_equal(before[name],after[name])
        a0,b0=base_and_b_features(before,np.arange(10))
        a1,b1=base_and_b_features(after,np.arange(10))
        np.testing.assert_array_equal(a0,a1)
        np.testing.assert_array_equal(b0,b1)

    def test_shuffle_preserves_partition_session_and_role(self):
        ex=self.ex
        tr=select_partition(ex,("Ses01","Ses02","Ses03"))
        va=select_partition(ex,("Ses04",))
        self.assertEqual(len(set(ex["dialogue"][tr])
                             &set(ex["dialogue"][va])),0)
        for ids,seed in ((tr,31),(va,47)):
            donors=shuffle_b_donors(ex,ids,seed)
            self.assertEqual(len(donors),len(ids))
            self.assertTrue(set(donors.tolist()).issubset(set(ids.tolist())))
            np.testing.assert_array_equal(
                ex["session"][ids],ex["session"][donors]
            )
            np.testing.assert_array_equal(
                ex["b_role"][ids],ex["b_role"][donors]
            )
            self.assertTrue(
                np.all(ex["dialogue"][ids]!=ex["dialogue"][donors])
            )
            np.testing.assert_array_equal(
                donors,shuffle_b_donors(ex,ids,seed)
            )

    def test_future_features_do_not_contaminate_earlier_aba(self):
        ds=FakeOriginalTrain()
        before=build_aba_examples(ds,strict=False,projection=self.proj)
        # Future last turn cannot change a target from turn 2.
        ds.videoText[ds.keys[0]][8,0]+=900
        ds.videoAudio[ds.keys[0]][8,1]-=400
        after=build_aba_examples(ds,strict=False,projection=self.proj)
        for name in ("current","previous","other"):
            np.testing.assert_allclose(
                before[name][0],after[name][0],atol=1e-6
            )
        self.assertGreater(
            float(np.linalg.norm(
                before["current"][6]-after["current"][6])),0.0
        )

    def test_smoke_one_fold_all_arms_aligned_and_finite(self):
        ex=self.ex
        tr=select_partition(ex,("Ses01","Ses02","Ses03"))
        va=select_partition(ex,("Ses04",))
        preds,checks=evaluate_fold(ex,tr,va,shuffle_seeds=(31,47,73))
        self.assertEqual(len(preds),8)
        self.assertEqual(len(checks),3)
        for name,p in preds.items():
            self.assertEqual(p.shape,(len(va),),name)
            self.assertTrue(np.all((p>=0)&(p<=5)),name)
            metrics=subgroup_metrics(
                ex["label"][va],p,ex["changed"][va]
            )
            self.assertEqual(metrics["all"]["n"],len(va))
            self.assertTrue(np.isfinite(metrics["all"]["weighted_f1"]))
        self.assertTrue(all(x["valid_different_dialogue"] for x in checks))
        self.assertTrue(all(x["valid_same_session"] for x in checks))
        self.assertTrue(all(x["valid_same_b_role"] for x in checks))

    def test_data_leakage_guards(self):
        ex=self.ex
        tr=select_partition(ex,("Ses01","Ses02","Ses03"))
        va=select_partition(ex,("Ses04",))
        with self.assertRaises(ValueError):
            evaluate_fold(ex,np.concatenate((tr,va[:1])),va)
        with self.assertRaises(ValueError):
            shuffle_b_donors(ex,np.array([],dtype=np.int64),31)

    def test_metrics_only_posthoc_changed(self):
        y=np.array([0,0,1,2])
        pred=np.array([0,1,1,0])
        changed=np.array([False,True,True,False])
        metrics=subgroup_metrics(y,pred,changed)
        self.assertEqual(metrics["all"]["n"],4)
        self.assertEqual(metrics["changed"]["n"],2)
        self.assertEqual(metrics["stable"]["n"],2)
        self.assertAlmostEqual(metrics["all"]["accuracy"],50.0)


if __name__=="__main__":
    unittest.main()
