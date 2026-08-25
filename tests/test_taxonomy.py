"""
the 10-class re-fit is implemented twice: once in plots/paper_figures.ipynb for
the manuscript figures, once in docs/tools/build_results.py for the results
site. the class list is written a third time inside every evaluation notebook.
nothing keeps those in step, so a change to one silently puts the site and the
paper on different numbers. these tests are that guard.
"""
import ast
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "plots" / "paper_figures.ipynb"
BUILDER = REPO / "docs" / "tools" / "build_results.py"

WANTED = {"CLASS_COUNTS_13", "EXCLUDED_IDX", "KEEP_IDX", "DENOM_10",
          "SECTOR_CLASSES_10", "CLASS_KEYS_13", "_CLASS_DISPLAY_ALL_13",
          "CLASS_DISPLAY_ALL_13"}


def _consts_from_source(src, wanted=WANTED):
    """literal-eval the top-level constant assignments we care about."""
    found = {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return found
    for node in tree.body:
        if not (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name not in wanted:
            continue
        try:
            found[name] = ast.literal_eval(node.value)
        except ValueError:
            pass          # a comprehension, recomputed below
    return found


def _notebook_consts():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    merged = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        merged.update(_consts_from_source("".join(cell["source"])))
    return merged


def _builder_consts():
    return _consts_from_source(BUILDER.read_text(encoding="utf-8"))


class TestRefitConstantsAgree(unittest.TestCase):
    """paper_figures.ipynb and build_results.py must define the same taxonomy."""

    @classmethod
    def setUpClass(cls):
        cls.nb = _notebook_consts()
        cls.bd = _builder_consts()

    def test_both_sources_were_parsed(self):
        self.assertIn("CLASS_COUNTS_13", self.nb, "notebook constants not found")
        self.assertIn("CLASS_COUNTS_13", self.bd, "builder constants not found")

    def test_class_counts_match(self):
        self.assertEqual(self.nb["CLASS_COUNTS_13"], self.bd["CLASS_COUNTS_13"])

    def test_excluded_indices_match(self):
        self.assertEqual(set(self.nb["EXCLUDED_IDX"]), set(self.bd["EXCLUDED_IDX"]))

    def test_sector_map_matches(self):
        self.assertEqual(self.nb["SECTOR_CLASSES_10"], self.bd["SECTOR_CLASSES_10"])

    def test_display_names_match(self):
        a = self.nb.get("_CLASS_DISPLAY_ALL_13")
        b = self.bd.get("CLASS_DISPLAY_ALL_13")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertEqual(a, b)


class TestRefitArithmetic(unittest.TestCase):
    """the numbers the paper quotes have to fall out of these constants."""

    @classmethod
    def setUpClass(cls):
        cls.c = _builder_consts()
        cls.counts = cls.c["CLASS_COUNTS_13"]
        cls.excluded = set(cls.c["EXCLUDED_IDX"])
        cls.keep = [i for i in range(13) if i not in cls.excluded]

    def test_thirteen_classes(self):
        self.assertEqual(len(self.counts), 13)

    def test_three_classes_excluded(self):
        self.assertEqual(len(self.excluded), 3)
        self.assertEqual(len(self.keep), 10)

    def test_thirteen_class_test_set_is_2813(self):
        self.assertEqual(sum(self.counts), 2813)

    def test_ten_class_test_set_is_2689(self):
        self.assertEqual(sum(self.counts[i] for i in self.keep), 2689)

    def test_excluded_classes_are_the_three_smallest_reported(self):
        """wind farm (3), port terminal (3), water works (118)."""
        self.assertEqual(sorted(self.counts[i] for i in self.excluded), [3, 3, 118])

    def test_sector_map_covers_every_retained_class_exactly_once(self):
        seen = [i for idxs in self.c["SECTOR_CLASSES_10"].values() for i in idxs]
        self.assertEqual(sorted(seen), self.keep)
        self.assertEqual(len(seen), len(set(seen)), "a class is in two sectors")

    def test_sector_map_excludes_the_dropped_classes(self):
        seen = {i for idxs in self.c["SECTOR_CLASSES_10"].values() for i in idxs}
        self.assertEqual(seen & self.excluded, set())

    def test_class_keys_and_display_names_are_aligned(self):
        self.assertEqual(len(self.c["CLASS_KEYS_13"]), 13)
        self.assertEqual(len(self.c["CLASS_DISPLAY_ALL_13"]), 13)


class TestEvaluationNotebooksAgree(unittest.TestCase):
    """every evaluation notebook hardcodes its own CLASS_NAMES list. per-class
    metrics are stored by position, so the ORDER has to be identical across
    notebooks or index 4 means solar farm in one run and something else in
    another. per_sector_f1_catchall.ipynb uses short labels for display; that
    is fine as long as it keeps the same order and length."""

    @classmethod
    def setUpClass(cls):
        cls.lists = {}
        for nbp in sorted((REPO / "evaluation").rglob("*.ipynb")):
            nb = json.loads(nbp.read_text(encoding="utf-8"))
            for cell in nb["cells"]:
                if cell["cell_type"] != "code":
                    continue
                got = _consts_from_source("".join(cell["source"]), {"CLASS_NAMES"})
                if "CLASS_NAMES" in got:
                    cls.lists[nbp.relative_to(REPO).as_posix()] = got["CLASS_NAMES"]
                    break

    def test_most_notebooks_declare_class_names(self):
        self.assertGreater(len(self.lists), 20)

    def test_every_list_has_thirteen_entries(self):
        for path, names in self.lists.items():
            self.assertEqual(len(names), 13, path)

    def test_dotted_lists_are_all_identical(self):
        dotted = {p: tuple(v) for p, v in self.lists.items()
                  if all("." in n for n in v)}
        self.assertGreater(len(dotted), 20)
        distinct = set(dotted.values())
        self.assertEqual(len(distinct), 1,
                         f"notebooks disagree on CLASS_NAMES: {len(distinct)} variants")

    def test_dotted_list_matches_the_site_builder(self):
        dotted = next(tuple(v) for v in self.lists.values()
                      if all("." in n for n in v))
        self.assertEqual(list(dotted), _builder_consts()["CLASS_KEYS_13"])

    # per_sector_f1_catchall.ipynb labels the same 13 classes with short names.
    # spelling them out here means a reordering on either side fails the test,
    # which a fuzzy string match would not catch.
    SHORT_ALIASES = {
        "tx_substation": "energy.transmission.substation",
        "dx_substation": "energy.distribution.substation",
        "dx_other":      "energy.distribution.other",
        "power_plant":   "energy.generation.power_plant",
        "solar_farm":    "energy.generation.solar_farm",
        "wind_farm":     "energy.generation.wind_farm",
        "wastewater":    "water.wastewater.plant",
        "water_works":   "water.water_works",
        "storage_tank":  "water.storage_tank",
        "airport":       "transport.airport",
        "train_station": "transport.train_station",
        "port_terminal": "transport.port_terminal",
        "data_center":   "telecom.data_center",
    }

    def test_short_label_lists_map_onto_the_same_order(self):
        dotted = next(tuple(v) for v in self.lists.values()
                      if all("." in n for n in v))
        checked = 0
        for path, names in self.lists.items():
            if all("." in n for n in names):
                continue
            checked += 1
            for i, (short, full) in enumerate(zip(names, dotted)):
                self.assertIn(short, self.SHORT_ALIASES,
                              f"{path}: unknown short label {short!r}")
                self.assertEqual(self.SHORT_ALIASES[short], full,
                                 f"{path} index {i}: {short!r} should be at the "
                                 f"position of {self.SHORT_ALIASES[short]!r}, "
                                 f"found {full!r}")
        self.assertGreaterEqual(checked, 1, "no short-label notebook found")


class TestOntology(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from curation import ontology
        cls.o = ontology

    def test_class_names_are_unique(self):
        names = [c.name for c in self.o.ASSET_CLASSES]
        self.assertEqual(len(names), len(set(names)))

    def test_every_class_has_a_valid_sector(self):
        for c in self.o.ASSET_CLASSES:
            self.assertIn(c.sector, self.o.VALID_SECTORS, c.name)

    def test_every_class_has_a_valid_confidence(self):
        for c in self.o.ASSET_CLASSES:
            self.assertIn(c.confidence, self.o.VALID_CONFIDENCE, c.name)

    def test_area_constraints_need_an_area_bearing_geometry(self):
        for c in self.o.ASSET_CLASSES:
            if c.min_area_m2 is not None:
                self.assertIn(c.require_geometry, self.o.GEOMETRIES_WITH_AREA, c.name)

    def test_sectors_index_covers_every_class(self):
        indexed = {c.name for classes in self.o.SECTORS.values() for c in classes}
        self.assertEqual(indexed, {c.name for c in self.o.ASSET_CLASSES})

    def test_lookup_by_name_round_trips(self):
        for c in self.o.ASSET_CLASSES:
            self.assertIs(self.o.get_class_by_name(c.name), c)

    def test_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            self.o.get_class_by_name("not.a.real.class")

    def test_catchall_substation_is_ordered_after_its_subtypes(self):
        """matchers take the first hit, so the untyped fallback must come last
        or a transmission substation would be labelled untyped."""
        names = [c.name for c in self.o.ASSET_CLASSES]
        catchall = [n for n in names if n.endswith("substation_untyped")]
        if not catchall:
            self.skipTest("no untyped substation catch-all in the ontology")
        last = names.index(catchall[0])
        for specific in ("energy.transmission.substation",
                         "energy.distribution.substation"):
            if specific in names:
                self.assertLess(names.index(specific), last,
                                f"{specific} must precede {catchall[0]}")

    def test_every_class_declares_something_to_match_on(self):
        for c in self.o.ASSET_CLASSES:
            self.assertTrue(c.tags or c.any_of_tags,
                            f"{c.name} has no tags and no any_of_tags")


if __name__ == "__main__":
    unittest.main()


class TestShippedCellSizes(unittest.TestCase):
    """the 1k cap is a target, not a hard ceiling: each class is allocated its
    own rounded share and the parts need not sum to it. these pin the shipped
    numbers so a well-meaning 'fix' to the rounding is caught."""

    TARGET = 1000

    @classmethod
    def setUpClass(cls):
        from curation.paths import CURATED_DIR
        cls.manifests = sorted(CURATED_DIR.glob("dataset_*_v1_1k/manifest.json"))
        if not cls.manifests:
            raise unittest.SkipTest("curated cells not present locally")

    def test_twenty_eight_cells(self):
        self.assertEqual(len(self.manifests), 28)

    def test_total_is_18756(self):
        total = sum(json.loads(m.read_text())["n_tiles"] for m in self.manifests)
        self.assertEqual(total, 18756)

    def test_overshoot_is_bounded_by_the_class_count(self):
        """a cell can exceed the target by at most one per class, since each
        class contributes at most half a tile of rounding error either way."""
        for m in self.manifests:
            obj = json.loads(m.read_text())
            n = obj["n_tiles"]
            n_classes = len(obj.get("asset_types") or {})
            self.assertLessEqual(
                n, self.TARGET + max(n_classes, 1),
                f"{m.parent.name}: {n} tiles exceeds {self.TARGET} by more than "
                f"its {n_classes} classes can explain")
