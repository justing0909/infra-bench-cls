"""
the label review has one rule that matters: no tile counts as checked until a
human said so, and a disagreement needs a second human. these tests pin that
rule and the VIA round trip that feeds it, in a scratch directory so they never
touch the real decisions logs.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from curation.review import decisions, via

SUB = "energy.transmission.substation"
DX = "energy.distribution.substation"


def _e(reviewer, decision, label=""):
    return {"reviewer": reviewer, "decision": decision, "label": label, "note": ""}


class TestTwoPersonRule(unittest.TestCase):
    def test_untouched_tile_is_open_for_everyone(self):
        self.assertEqual(decisions.status({}), "open")
        self.assertEqual(decisions.wants("ed", {}, True), "first")

    def test_a_single_accept_is_enough(self):
        e = {"justin": _e("justin", "accept", SUB)}
        self.assertEqual(decisions.status(e, True), "done")
        self.assertEqual(decisions.final_label(e, True), SUB)
        self.assertEqual(decisions.wants("ed", e, True), "")

    def test_relabel_goes_to_the_other_reviewer_only(self):
        e = {"justin": _e("justin", "relabel", DX)}
        self.assertEqual(decisions.status(e, True), "needs_second")
        self.assertEqual(decisions.wants("ed", e, True), "second")
        self.assertEqual(decisions.wants("justin", e, True), "")

    def test_unsure_also_needs_a_second_look(self):
        self.assertEqual(decisions.status({"justin": _e("justin", "unsure")}, True), "needs_second")

    def test_two_reviewers_agreeing_on_a_relabel_is_done(self):
        e = {"justin": _e("justin", "relabel", DX), "ed": _e("ed", "relabel", DX)}
        self.assertEqual(decisions.status(e, True), "done")
        self.assertEqual(decisions.final_label(e, True), DX)

    def test_disagreement_is_disputed_not_resolved(self):
        e = {"justin": _e("justin", "relabel", DX), "ed": _e("ed", "accept", SUB)}
        self.assertEqual(decisions.status(e, True), "disputed")
        self.assertEqual(decisions.final_label(e, True), "")

    def test_unsure_plus_one_accept_is_still_disputed(self):
        e = {"justin": _e("justin", "unsure"), "ed": _e("ed", "accept", SUB)}
        self.assertEqual(decisions.status(e, True), "disputed")

    def test_reviewer_names_cannot_escape_the_review_dir(self):
        with self.assertRaises(ValueError):
            decisions.log_path("../ed")


class TestOneReviewer(unittest.TestCase):
    def test_a_relabel_is_final(self):
        e = {"justin": _e("justin", "relabel", DX)}
        self.assertEqual(decisions.status(e), "done")
        self.assertEqual(decisions.final_label(e), DX)
        self.assertEqual(decisions.wants("justin", e), "")

    def test_unsure_comes_back_to_the_same_reviewer(self):
        e = {"justin": _e("justin", "unsure")}
        self.assertEqual(decisions.status(e), "unsure")
        self.assertEqual(decisions.final_label(e), "")
        self.assertEqual(decisions.wants("justin", e), "revisit")

    def test_untouched_tile_is_a_first_look(self):
        self.assertEqual(decisions.wants("justin", {}), "first")


class TestViaVerdicts(unittest.TestCase):
    def test_mapping(self):
        f = via.to_decision
        self.assertIsNone(f({}))
        self.assertEqual(f({"verdict": "true", "osm_label": SUB}), ("accept", SUB, ""))
        self.assertEqual(f({"verdict": "false", "correct_class": DX}), ("relabel", DX, ""))
        self.assertEqual(f({"verdict": "unconfirmed", "note": " hazy "}), ("unsure", "", "hazy"))
        decision, label, note = f({"verdict": "false"})
        self.assertEqual((decision, label), ("unsure", ""))
        self.assertIn("no correct class", note)


class TestViaRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        tiles = root / "curated" / "ds" / "images"
        tiles.mkdir(parents=True)
        rows = []
        for i, label in enumerate([SUB, SUB, DX]):
            np.save(tiles / f"a{i}.npy", np.random.default_rng(i).random((9, 60, 60)).astype(np.float32))
            rows.append({"order": i, "asset_id": f"osm_way_{i}", "asset_type": label,
                         "lat": 44.0, "lon": -70.0, "name": "", "name_hint": "",
                         "model_suggests": "", "model_p_top": "", "image": f"ds/images/a{i}.npy"})
        queue = root / "review" / "queue.csv"
        queue.parent.mkdir()
        pd.DataFrame(rows).to_csv(queue, index=False)
        review = root / "review"
        self.patches = [
            mock.patch.object(decisions, "REVIEW_DIR", review),
            mock.patch.object(via, "REVIEW_QUEUE", queue),
            mock.patch.object(via, "CURATED_DIR", root / "curated"),
            mock.patch.object(via, "CHIP_DIR", review / "chips"),
            mock.patch.object(via, "VIA_DIR", review / "via"),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _review(self, reviewer, verdicts, second_opinion=False):
        """export for reviewer, fill verdicts by asset_id, import, return counts."""
        written = via.export(reviewer, 500, "chips/", [], second_opinion)
        for path, _ in written:
            project = json.loads(path.read_text(encoding="utf-8"))
            for meta in project["_via_img_metadata"].values():
                v = verdicts.get(Path(meta["filename"]).stem)
                if v:
                    meta["file_attributes"].update(v)
            path.write_text(json.dumps(project), encoding="utf-8")
        return written, via.import_(reviewer, [p for p, _ in written])

    def test_projects_have_no_preselected_verdict(self):
        written = via.export("justin", 500, "chips/", [])
        self.assertEqual(sum(n for _, n in written), 3)
        for path, _ in written:
            project = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(project["_via_attributes"]["file"]["verdict"]["default_options"], {})
            for key, meta in project["_via_img_metadata"].items():
                self.assertEqual(key, meta["filename"] + str(meta["size"]))
                self.assertNotIn("verdict", meta["file_attributes"])

    def test_one_reviewer_end_to_end(self):
        self._review("justin", {"osm_way_0": {"verdict": "true"},
                                "osm_way_1": {"verdict": "unconfirmed"},
                                "osm_way_2": {"verdict": "false", "correct_class": SUB}})
        # only the unsure tile comes back, as a revisit.
        written, _ = self._review("justin", {"osm_way_1": {"verdict": "false", "correct_class": DX}})
        self.assertEqual([p.name for p, _ in written], [f"justin_revisit_{SUB}_001.json"])
        latest = decisions.load()
        self.assertEqual([decisions.final_label(latest[f"osm_way_{i}"]) for i in range(3)], [SUB, DX, SUB])
        self.assertEqual(via.export("justin", 500, "chips/", []), [])

    def test_two_reviewers_end_to_end(self):
        _, counts = self._review("justin", {
            "osm_way_0": {"verdict": "true"},
            "osm_way_1": {"verdict": "false", "correct_class": DX},
            # osm_way_2 left untouched
        }, second_opinion=True)
        self.assertEqual(counts, {"recorded": 2, "unchanged": 0, "untouched": 1})

        # ed sees the relabel as a second opinion and the untouched tile as a
        # first look, but not the tile justin already accepted.
        written, _ = self._review("ed", {"osm_way_1": {"verdict": "false", "correct_class": DX},
                                         "osm_way_2": {"verdict": "true"}},
                                  second_opinion=True)
        names = sorted(p.name for p, _ in written)
        self.assertEqual(names, [f"ed_first_{DX}_001.json", f"ed_second_{SUB}_001.json"])

        latest = decisions.load()
        for asset, label in [("osm_way_0", SUB), ("osm_way_1", DX), ("osm_way_2", DX)]:
            self.assertEqual(decisions.final_label(latest[asset], True), label)

    def test_reimporting_the_same_project_adds_nothing(self):
        written, _ = self._review("justin", {"osm_way_0": {"verdict": "true"}})
        again = via.import_("justin", [p for p, _ in written])
        self.assertEqual(again["recorded"], 0)
        self.assertEqual(again["unchanged"], 1)

    def test_an_edited_osm_label_field_cannot_change_an_accept(self):
        self._review("justin", {"osm_way_0": {"verdict": "true", "osm_label": DX}})
        self.assertEqual(decisions.load()["osm_way_0"]["justin"]["label"], SUB)


class TestAutoAccept(unittest.TestCase):
    """auto-accept must stay out of a person's way and keep its audit."""

    def setUp(self):
        from curation.review import auto_accept
        self.aa = auto_accept
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "sources").mkdir()
        ids = [f"osm_way_{i}" for i in range(30)]
        pd.DataFrame({"order": range(30), "asset_id": ids, "asset_type": SUB}).to_csv(root / "queue.csv", index=False)
        pd.DataFrame({"asset_id": ids, "hifld_text": "HIFLD: 2 line(s) end here (230 kV)",
                      "auto_ok": True}).to_csv(root / "sources" / "hifld_lines.csv", index=False)
        self.patches = [mock.patch.object(decisions, "REVIEW_DIR", root),
                        mock.patch.object(auto_accept, "REVIEW_DIR", root),
                        mock.patch.object(auto_accept, "REVIEW_QUEUE", root / "queue.csv"),
                        mock.patch.object(auto_accept, "AUDIT", root / "audit.csv")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_audit_is_held_back_and_humans_win(self):
        decisions.append("justin", "osm_way_0", "relabel", DX)
        counts = self.aa.apply()
        table = pd.read_csv(self.aa.AUDIT)
        audit = set(table[table.audit_kind == "held_back"].asset_id)
        self.assertEqual(len(audit), 3)                        # 10% of 30
        latest = decisions.load()
        for a in audit:
            self.assertNotIn("source", latest.get(a, {}))      # left for a person
        if "osm_way_0" not in audit:
            self.assertEqual(counts["human_decided"], 1)
            self.assertNotIn("source", latest["osm_way_0"])    # never overrides a person
        self.assertEqual(counts["accepted"] + counts["human_decided"], 30 - len(audit))

    def test_a_new_source_never_reshuffles_the_audit(self):
        self.aa.apply()
        before = set(pd.read_csv(self.aa.AUDIT).query("audit_kind == 'held_back'").asset_id)
        queue = pd.read_csv(self.aa.REVIEW_QUEUE)
        new = [f"osm_way_{i}" for i in range(100, 110)]
        pd.concat([queue, pd.DataFrame({"order": range(100, 110), "asset_id": new, "asset_type": SUB})]
                  ).to_csv(self.aa.REVIEW_QUEUE, index=False)
        pd.DataFrame({"asset_id": new, "register": "DB", "register_text": "DB: x 5 m away (matches OSM name)",
                      "auto_ok": True}).to_csv(self.aa.REVIEW_DIR / "sources" / "registers.csv", index=False)
        self.aa.apply()
        after = pd.read_csv(self.aa.AUDIT).query("audit_kind == 'held_back'")
        self.assertTrue(before <= set(after.asset_id))         # nothing earlier dropped
        self.assertEqual((after.source == "DB").sum(), 1)       # the new source gets its own pick
        accepted = {a for a, e in decisions.load().items() if "source" in e}
        self.assertFalse(set(after.asset_id) & accepted)        # never an already-accepted tile

    def test_rerunning_adds_nothing_and_keeps_the_same_audit(self):
        self.aa.apply()
        held = lambda: set(pd.read_csv(self.aa.AUDIT).query("audit_kind == 'held_back'").asset_id)
        first = held()
        again = self.aa.apply()
        self.assertEqual(again["accepted"], 0)
        self.assertEqual(held(), first)


class TestTwoAgentRouting(unittest.TestCase):
    """only confident agreement becomes a provisional label (Ed's rule)."""

    CLASSES = [SUB, DX, "not_infrastructure"]

    def _run(self, inv, rev, human=None, threshold=0.85):
        from curation.review import two_agent
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        run = root / "r"
        ids = [r["id"] for r in inv]
        for role, rows in (("investigator", inv), ("reviewer", rev)):
            (run / role).mkdir(parents=True)
            packet = [{"id": i, "evidence_ids": [f"{i}:sentinel2", f"{i}:osm"],
                       "evidence": {"osm": {"label": SUB}}} for i in ids]
            (run / role / "packet.json").write_text(json.dumps(packet))
            (run / role / "decisions.json").write_text(json.dumps(rows))
        (run / "policy.json").write_text(json.dumps({"ids": ids, "classes": self.CLASSES}))
        with mock.patch.object(two_agent, "RUNS_DIR", root), mock.patch.object(decisions, "REVIEW_DIR", root):
            preds = two_agent.finalize("r", threshold)
            for i, (d, label) in (human or {}).items():
                decisions.append("justin", i, d, label)
            result = two_agent.score("r") if human else None
        return {p["id"]: p for p in preds}, result

    @staticmethod
    def _row(i, action, label=None, conf=0.9, q="adequate"):
        r = {"id": i, "action": action, "confidence": conf, "image_quality": q,
             "evidence_ids": [f"{i}:sentinel2"], "reason": "seen"}
        if label:
            r["label"] = label
        return r

    def test_agreement_confirms_and_agreed_relabel_relabels(self):
        preds, _ = self._run([self._row("a", "confirm"), self._row("b", "relabel", DX)],
                             [self._row("a", "classify", SUB), self._row("b", "classify", DX)])
        self.assertEqual((preds["a"]["action"], preds["a"]["label"]), ("confirm", SUB))
        self.assertEqual((preds["b"]["action"], preds["b"]["label"]), ("relabel", DX))

    def test_every_doubt_routes_to_a_person(self):
        preds, _ = self._run(
            [self._row("a", "confirm"), self._row("b", "confirm", conf=0.8),
             self._row("c", "confirm", q="poor"), self._row("d", "request_review")],
            [self._row("a", "classify", DX), self._row("b", "classify", SUB),
             self._row("c", "classify", SUB), self._row("d", "classify", SUB)])
        self.assertIn("agent_disagreement", preds["a"]["reason"])
        self.assertIn("low_confidence", preds["b"]["reason"])
        self.assertIn("poor_imagery", preds["c"]["reason"])
        self.assertIn("agent_abstention", preds["d"]["reason"])
        self.assertTrue(all(p["action"] == "request_review" for p in preds.values()))

    def test_missing_rows_stop_finalize(self):
        with self.assertRaises(ValueError):
            self._run([self._row("a", "confirm"), self._row("b", "confirm")], [self._row("a", "classify", SUB)])

    def test_score_counts_a_confident_wrong_confirm(self):
        _, s = self._run([self._row("a", "confirm"), self._row("b", "confirm")],
                         [self._row("a", "classify", SUB), self._row("b", "classify", SUB)],
                         human={"a": ("accept", SUB), "b": ("relabel", DX)})
        self.assertEqual((s["coverage"], s["wrong_automatic"], s["false_confirms"]), (1.0, 1, 1))
        self.assertEqual(s["selective_error"], 0.5)


class TestClassSample(unittest.TestCase):
    """the per-class sample size rule and the accuracy report."""

    def test_sample_size_rule(self):
        from curation.review.class_sample import sample_size
        self.assertEqual(sample_size(5046), 100)   # capped
        self.assertEqual(sample_size(676), 68)     # 10%, rounded up
        self.assertEqual(sample_size(150), 20)     # floor
        self.assertEqual(sample_size(12), 12)      # whole class when smaller

    def test_report_counts_relabels_as_wrong_and_skips_unsure(self):
        from curation.review import class_sample
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({"asset_id": ["a", "b", "c", "d"], "asset_type": SUB, "class_size": 40,
                          "draw": range(4), "hint": ""}).to_csv(root / "s.csv", index=False)
            with mock.patch.object(class_sample, "SAMPLE", root / "s.csv"), \
                 mock.patch.object(decisions, "REVIEW_DIR", root):
                decisions.append("justin", "a", "accept", SUB)
                decisions.append("justin", "b", "relabel", DX)
                decisions.append("justin", "c", "unsure", "")
                summary, confusion = class_sample.report("justin")
        row = summary.iloc[0]
        self.assertEqual((row.reviewed, row.unsure, row.correct, row["accuracy_%"]), (3, 1, 1, 50.0))
        self.assertEqual(confusion.tiles.sum(), 1)
        self.assertEqual(row.next, "keep going")

    def test_report_works_before_any_decisions(self):
        from curation.review import class_sample
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({"asset_id": ["a"], "asset_type": SUB, "class_size": 40, "draw": [0],
                          "hint": ""}).to_csv(root / "s.csv", index=False)
            with mock.patch.object(class_sample, "SAMPLE", root / "s.csv"), \
                 mock.patch.object(decisions, "REVIEW_DIR", root):
                summary, confusion = class_sample.report("nobody")
        self.assertEqual((summary.iloc[0].reviewed, summary.iloc[0].next, len(confusion)), (0, "keep going", 0))

    def test_a_clean_class_can_stop_at_thirty(self):
        from curation.review import class_sample
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = [f"t{i}" for i in range(60)]
            pd.DataFrame({"asset_id": ids, "asset_type": SUB, "class_size": 600, "draw": range(60),
                          "hint": ""}).to_csv(root / "s.csv", index=False)
            with mock.patch.object(class_sample, "SAMPLE", root / "s.csv"), \
                 mock.patch.object(decisions, "REVIEW_DIR", root):
                for i in ids[:30]:
                    decisions.append("justin", i, "accept", SUB)
                summary, _ = class_sample.report("justin")
        self.assertEqual(summary.iloc[0].next, "enough")


if __name__ == "__main__":
    unittest.main()


class TestGridOperatorVoltages(unittest.TestCase):
    def test_voltages_set_the_type(self):
        from curation.review import grid_operators as g
        self.assertEqual(g._kvs("PARKLANDS ( 66 / 11 KV)"), [66.0, 11.0])
        self.assertEqual(g._kvs("13.2 | 33"), [13.2, 33.0])
        self.assertEqual(g._by_voltage([500], [220]), g.TX)
        self.assertEqual(g._by_voltage([33], [11]), g.DX)
        # transmission in, distribution out: OSM files these under either
        self.assertIsNone(g._by_voltage([132], [33, 13.2]))
        self.assertIsNone(g._by_voltage([]))

    def test_ga_voltage_overrides_zone(self):
        from curation.review import grid_operators as g
        self.assertEqual(g._ga_kind(g.DX, 330), g.TX)
        self.assertIsNone(g._ga_kind(g.DX, 132))
        self.assertEqual(g._ga_kind(g.DX, 66), g.DX)
        self.assertEqual(g._ga_kind(None, None), None)
