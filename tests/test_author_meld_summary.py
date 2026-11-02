"""Tests for summarizing original-author-protocol MELD log lines."""
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_author_meld import parse_log, summarize


class TestSummarizeAuthorMeld(unittest.TestCase):
    def test_correctly_summarizes_two_completed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "checkpoints").mkdir()
            for name, f1, acc, epoch in [
                ("css", 66.95, 68.01, 57),
                ("v0", 67.12, 68.20, 62),
            ]:
                ck = d / "checkpoints" / (name + "_best.pt")
                ck.touch()
                (d / f"{name}.log").write_text(
                    f"Test performance..\nF-Score: {f1}\n"
                    f"AUTHOR_PROTOCOL_BEST epoch={epoch} test_acc={acc} "
                    f"test_weighted_f1={f1} checkpoint={ck}\n"
                    "AUTHOR_PROTOCOL_NOTE test_set_used_for_epoch_selection=true\n",
                    encoding="utf-8"
                )
            summary = summarize(d)
            self.assertIn("66.95%", summary)
            self.assertIn("67.12%", summary)
            self.assertIn("+0.17 percentage points", summary)
            self.assertTrue((d / "summary.json").is_file())
            self.assertEqual(parse_log(d / "v0.log")["selected_epoch"], 62)

    def test_incomplete_run_does_not_create_misleading_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "css.log"
            log.write_text("epoch: 1 ... test_fscore: 38.0\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                parse_log(log)


if __name__ == "__main__":
    unittest.main()
