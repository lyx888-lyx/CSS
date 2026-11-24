"""Synthetic tests; no dataset, model checkpoint or TEST access."""
import unittest
import numpy as np

from research.history_intervention_utility import (
    ARMS, FixedLogistic, feature_matrix, fit_selectors, predict_selectors, metrics,
)


def row(y, prior_true, b, m, prior_pred):
    def prob(c):
        p = np.full(6, .02)
        p[c] = .90
        return p.tolist()
    return {"true": y, "previous_own_true": prior_true,
            "baseline_pred": b, "previous_own_pred": m,
            "baseline_prob": prob(b), "previous_own_prob": prob(m),
            "prior_pred": prior_pred, "index": 3, "previous_own_index": 1}


class TestHistoryUtility(unittest.TestCase):
    def setUp(self):
        self.rows = [row(3,5,5,3,5), row(5,5,5,3,5),
                     row(2,5,5,3,5), row(0,0,0,0,0),
                     row(4,0,0,4,0), row(1,1,1,4,1)]

    def test_fit_utility_and_gold_free_inference(self):
        models, diag = fit_selectors(self.rows)
        self.assertEqual((diag["fix"],diag["harm"],diag["neither_on_disagreement"]),
                         (2,2,1))
        without_gold = [{k:v for k,v in r.items() if k not in
                         ("true","previous_own_true")} for r in self.rows]
        full = predict_selectors(models,self.rows)
        stripped = predict_selectors(models,without_gold)
        self.assertEqual(set(full),set(ARMS))
        b = np.array([r["baseline_pred"] for r in self.rows])
        m = np.array([r["previous_own_pred"] for r in self.rows])
        for a in ARMS:
            np.testing.assert_array_equal(full[a],stripped[a])
            self.assertTrue(np.all((full[a]==b)|(full[a]==m)))
        np.testing.assert_array_equal(full["baseline"],b)
        np.testing.assert_array_equal(full["always_mask"],m)

    def test_only_predicted_history_used(self):
        altered = [dict(r,true=(r["true"]+1)%6,
                        previous_own_true=(r["previous_own_true"]+2)%6)
                   for r in self.rows]
        np.testing.assert_array_equal(feature_matrix(self.rows,True),
                                      feature_matrix(altered,True))

    def test_probability_integrity(self):
        bad = [dict(self.rows[0], baseline_prob=[.5]*6)]
        with self.assertRaises(ValueError):
            feature_matrix(bad)
        with self.assertRaises(ValueError):
            feature_matrix([dict(self.rows[0], prior_pred=6)],True)

    def test_constant_train_class(self):
        x = np.zeros((3,2))
        model = FixedLogistic().fit(x,[0,0,0])
        np.testing.assert_array_equal(model.probability(x,0),np.ones(3))
        np.testing.assert_array_equal(model.probability(x,1),np.zeros(3))

    def test_paired_correctness(self):
        m = metrics([0,1,2],[1,1,2],[0,0,2])
        self.assertEqual((m["fixed"],m["broken"],m["net_more_correct"]),(1,1,0))
        self.assertEqual(m["accuracy_delta_pp"],0.)


if __name__ == "__main__":
    unittest.main()
