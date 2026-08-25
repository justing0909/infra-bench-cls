"""
the split artifact is the single thing every one of the 30 evaluation runs
depends on, and a leak in it would inflate every reported number at once.
these tests assert the properties the paper claims for it: block coherence,
determinism across seeds and models, and the rare-class coverage fallback.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from curation.utils.spatial_blocking import (
    compute_spatial_blocks,
    assign_blocks_to_splits,
    save_split_artifact,
    load_split_artifact,
)

REGIONS = ["north-america", "europe", "africa"]
CLASSES = ["energy.transmission.substation", "water.storage_tank",
           "transport.airport", "telecom.data_center"]


def _tiles(n=600, seed=0, rare_class=None, n_rare=0):
    """synthetic tiles spread over three well-separated longitude bands."""
    rng = np.random.default_rng(seed)
    rows = []
    centers = {"north-america": (40.0, -100.0),
               "europe": (50.0, 10.0),
               "africa": (0.0, 20.0)}
    for i in range(n):
        region = REGIONS[i % len(REGIONS)]
        clat, clon = centers[region]
        rows.append({
            "asset_id":   f"osm_node_{i}",
            "region":     region,
            "sector":     "energy",
            "asset_type": CLASSES[i % len(CLASSES)],
            "lat":        clat + rng.uniform(-6, 6),
            "lon":        clon + rng.uniform(-6, 6),
        })
    for j in range(n_rare):
        rows.append({
            "asset_id":   f"osm_node_rare_{j}",
            "region":     "europe",
            "sector":     "energy",
            "asset_type": rare_class,
            "lat":        50.0 + (j * 1e-4),      # deliberately one tight cluster
            "lon":        10.0 + (j * 1e-4),
        })
    return pd.DataFrame(rows)


class TestComputeBlocks(unittest.TestCase):

    def test_requires_lat_lon(self):
        with self.assertRaises(ValueError):
            compute_spatial_blocks(pd.DataFrame({"x": [1]}))

    def test_nearby_tiles_share_a_block(self):
        df = pd.DataFrame([{"lat": 40.0, "lon": -100.0},
                           {"lat": 40.001, "lon": -100.001}])
        out = compute_spatial_blocks(df, block_size_km=200)
        self.assertEqual(out["block_id"].iloc[0], out["block_id"].iloc[1])

    def test_distant_tiles_do_not(self):
        df = pd.DataFrame([{"lat": 40.0, "lon": -100.0},
                           {"lat": 40.0, "lon": -80.0}])
        out = compute_spatial_blocks(df, block_size_km=200)
        self.assertNotEqual(out["block_id"].iloc[0], out["block_id"].iloc[1])

    def test_block_size_is_honored(self):
        """two points ~250 km apart share a 500 km block but not a 200 km one."""
        df = pd.DataFrame([{"lat": 0.0, "lon": 0.0}, {"lat": 0.0, "lon": 2.7}])
        small = compute_spatial_blocks(df, block_size_km=200)
        large = compute_spatial_blocks(df, block_size_km=5000)
        self.assertNotEqual(small["block_id"].iloc[0], small["block_id"].iloc[1])
        self.assertEqual(large["block_id"].iloc[0], large["block_id"].iloc[1])

    def test_input_columns_are_preserved(self):
        df = _tiles(30)
        out = compute_spatial_blocks(df)
        for col in df.columns:
            self.assertIn(col, out.columns)


class TestAssignSplits(unittest.TestCase):

    def setUp(self):
        self.blocked = compute_spatial_blocks(_tiles(600, seed=1))

    def test_fractions_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            assign_blocks_to_splits(self.blocked, 0.7, 0.2, 0.2)

    def test_missing_stratify_column_raises(self):
        with self.assertRaises(ValueError):
            assign_blocks_to_splits(self.blocked, stratify_by="nope")

    def test_unblocked_input_raises(self):
        with self.assertRaises(ValueError):
            assign_blocks_to_splits(_tiles(10))

    def test_every_tile_gets_a_split(self):
        out = assign_blocks_to_splits(self.blocked)
        self.assertFalse(out["split"].isna().any())
        self.assertEqual(set(out["split"]), {"train", "val", "test"})

    def test_block_coherence(self):
        """the core guarantee: one block never straddles two splits."""
        out = assign_blocks_to_splits(self.blocked)
        per_block = out.groupby("block_id")["split"].nunique()
        self.assertTrue((per_block == 1).all(),
                        f"{(per_block > 1).sum()} blocks straddle splits")

    def test_no_asset_id_leaks_across_splits(self):
        out = assign_blocks_to_splits(self.blocked)
        groups = {s: set(g["asset_id"]) for s, g in out.groupby("split")}
        self.assertEqual(groups["train"] & groups["test"], set())
        self.assertEqual(groups["train"] & groups["val"], set())
        self.assertEqual(groups["val"] & groups["test"], set())

    def test_fractions_land_near_target(self):
        out = assign_blocks_to_splits(self.blocked)
        frac = out["split"].value_counts(normalize=True)
        self.assertAlmostEqual(frac["train"], 0.70, delta=0.10)
        self.assertAlmostEqual(frac["val"], 0.15, delta=0.10)
        self.assertAlmostEqual(frac["test"], 0.15, delta=0.10)

    def test_deterministic_for_a_given_seed(self):
        a = assign_blocks_to_splits(self.blocked, seed=42)
        b = assign_blocks_to_splits(self.blocked, seed=42)
        pd.testing.assert_series_equal(a["split"], b["split"])

    def test_seed_actually_changes_the_split(self):
        a = assign_blocks_to_splits(self.blocked, seed=42)
        b = assign_blocks_to_splits(self.blocked, seed=7)
        self.assertTrue((a["split"] != b["split"]).any())

    def test_every_region_is_represented_in_every_split(self):
        out = assign_blocks_to_splits(self.blocked)
        for region, g in out.groupby("region"):
            self.assertEqual(set(g["split"]), {"train", "val", "test"},
                             f"{region} is missing a split")


class TestClassFallback(unittest.TestCase):
    """rare classes sit in one tight cluster, so block assignment alone would
    put all of them in a single split. the fallback trades block coherence for
    coverage on exactly those classes and must say so in split_protocol."""

    def setUp(self):
        self.rare = "transport.port_terminal"
        df = _tiles(400, seed=2, rare_class=self.rare, n_rare=12)
        self.blocked = compute_spatial_blocks(df)

    def test_rare_class_is_absent_from_some_split_without_the_fallback(self):
        out = assign_blocks_to_splits(self.blocked)
        got = set(out.loc[out["asset_type"] == self.rare, "split"])
        self.assertLess(len(got), 3, "fixture no longer exercises the fallback")

    def test_fallback_gives_the_rare_class_all_three_splits(self):
        out = assign_blocks_to_splits(self.blocked, class_fallback=[self.rare])
        got = set(out.loc[out["asset_type"] == self.rare, "split"])
        self.assertEqual(got, {"train", "val", "test"})

    def test_fallback_is_stamped_in_split_protocol(self):
        out = assign_blocks_to_splits(self.blocked, class_fallback=[self.rare])
        rare = out[out["asset_type"] == self.rare]
        rest = out[out["asset_type"] != self.rare]
        self.assertTrue((rare["split_protocol"] == "class_fallback").all())
        self.assertTrue((rest["split_protocol"] == "spatial_block").all())

    def test_unknown_fallback_class_raises(self):
        with self.assertRaises(ValueError):
            assign_blocks_to_splits(self.blocked, class_fallback=["not.a.class"])

    def test_non_fallback_classes_keep_block_coherence(self):
        out = assign_blocks_to_splits(self.blocked, class_fallback=[self.rare])
        rest = out[out["split_protocol"] == "spatial_block"]
        per_block = rest.groupby("block_id")["split"].nunique()
        self.assertTrue((per_block == 1).all())


class TestArtifactRoundTrip(unittest.TestCase):

    def test_save_then_load_preserves_the_mapping(self):
        out = assign_blocks_to_splits(compute_spatial_blocks(_tiles(120, seed=3)))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "split.parquet"
            save_split_artifact(out, str(path))
            loaded = load_split_artifact(str(path))
        self.assertEqual(len(loaded), len(out))
        expected = dict(zip(out["asset_id"].astype(str), out["split"]))
        self.assertEqual(loaded, expected)

    def test_missing_columns_are_rejected(self):
        out = assign_blocks_to_splits(compute_spatial_blocks(_tiles(30, seed=4)))
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                save_split_artifact(out.drop(columns=["sector"]),
                                    str(Path(d) / "x.parquet"))


if __name__ == "__main__":
    unittest.main()


class TestShippedSplitArtifact(unittest.TestCase):
    """the real artifact every evaluation run consumes, not a synthetic one.
    it is committed at data/spatial_split/, so these assert on the split the
    published numbers were actually produced with."""

    @classmethod
    def setUpClass(cls):
        from curation.paths import SPLIT_ARTIFACT, CURATED_DIR
        if not SPLIT_ARTIFACT.exists():
            raise unittest.SkipTest(f"no split artifact at {SPLIT_ARTIFACT}")
        cls.df = pd.read_parquet(SPLIT_ARTIFACT)
        cls.curated = CURATED_DIR

    def test_covers_every_dataset_tile(self):
        import json
        manifests = sorted(self.curated.glob("dataset_*_v1_1k/manifest.json"))
        if not manifests:
            self.skipTest("curated cells not present locally")
        ds = {r["asset_id"] for m in manifests
              for r in json.loads(m.read_text())["records"]}
        sp = set(self.df["asset_id"].astype(str))
        self.assertEqual(ds - sp, set(), "dataset tiles with no split assignment")
        self.assertEqual(sp - ds, set(), "split entries with no tile")

    def test_no_asset_is_in_two_splits(self):
        counts = self.df.groupby("asset_id")["split"].nunique()
        self.assertTrue((counts == 1).all())

    def test_splits_do_not_overlap(self):
        g = {s: set(x["asset_id"]) for s, x in self.df.groupby("split")}
        self.assertEqual(g["train"] & g["test"], set())
        self.assertEqual(g["train"] & g["val"], set())
        self.assertEqual(g["val"] & g["test"], set())

    def test_block_coherence_holds_for_block_assigned_tiles(self):
        blk = self.df[self.df["split_protocol"] == "spatial_block"].copy()
        blk["block"] = (blk["block_id_x"].astype(str) + "_"
                        + blk["block_id_y"].astype(str))
        straddle = blk.groupby("block")["split"].nunique()
        self.assertTrue((straddle == 1).all(),
                        f"{(straddle > 1).sum()} blocks straddle splits")

    def test_fractions_are_near_70_15_15(self):
        f = self.df["split"].value_counts(normalize=True)
        self.assertAlmostEqual(f["train"], 0.70, delta=0.02)
        self.assertAlmostEqual(f["val"], 0.15, delta=0.02)
        self.assertAlmostEqual(f["test"], 0.15, delta=0.02)

    def test_every_class_appears_in_every_split(self):
        for cls, sub in self.df.groupby("asset_type"):
            self.assertEqual(set(sub["split"]), {"train", "val", "test"}, cls)

    def test_only_rare_classes_use_the_fallback(self):
        fb = self.df[self.df["split_protocol"] == "class_fallback"]
        self.assertLess(len(fb), 0.01 * len(self.df),
                        "class_fallback should cover only the rare-class tail")

    def test_alphaearth_embeddings_cover_the_split(self):
        from curation.paths import ALPHAEARTH_EMBEDDINGS
        if not ALPHAEARTH_EMBEDDINGS.exists():
            self.skipTest("embeddings parquet not present")
        ae = pd.read_parquet(ALPHAEARTH_EMBEDDINGS, columns=["asset_id"])
        missing = set(self.df["asset_id"].astype(str)) - set(ae["asset_id"].astype(str))
        self.assertEqual(missing, set(),
                         f"{len(missing)} split assets have no AlphaEarth embedding")
