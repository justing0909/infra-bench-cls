"""
deduplication decides how many assets survive into the dataset, and it was
wrong once already: a scipy KDTree over raw degree pairs with a
threshold/111320 conversion, which only holds at the equator. the
high-latitude cases below are the regression tests for that.
"""
import unittest

import numpy as np
import pandas as pd

from curation.deduplication import Deduplicator, _haversine_m

SUB = "energy.transmission.substation"
EARTH_R = 6_371_000


def _offset(lat, lon, north_m=0.0, east_m=0.0):
    """move a point by a metre offset, accounting for latitude."""
    dlat = north_m / 111_320.0
    dlon = east_m / (111_320.0 * np.cos(np.radians(lat)))
    return lat + dlat, lon + dlon


def _df(points, asset_type=SUB):
    return pd.DataFrame([
        {"asset_id": f"osm_node_{i}", "asset_type": t, "lat": la, "lon": lo}
        for i, (la, lo, t) in enumerate(
            [(la, lo, asset_type) for la, lo in points]
            if not points or len(points[0]) == 2 else points)
    ])


class TestHaversine(unittest.TestCase):

    def test_one_degree_of_latitude_is_about_111km(self):
        d = _haversine_m(0.0, 0.0, 1.0, 0.0)
        self.assertAlmostEqual(d, 111_195, delta=500)

    def test_longitude_shrinks_with_latitude(self):
        at_equator = _haversine_m(0.0, 0.0, 0.0, 1.0)
        at_sixty = _haversine_m(60.0, 0.0, 60.0, 1.0)
        self.assertAlmostEqual(at_sixty / at_equator, 0.5, delta=0.01)

    def test_identical_points_are_zero(self):
        self.assertAlmostEqual(_haversine_m(45.0, 9.0, 45.0, 9.0), 0.0, places=6)


class TestDeduplication(unittest.TestCase):

    def setUp(self):
        self.dedup = Deduplicator(distance_threshold_m=200)

    def test_close_pair_collapses(self):
        a = (40.0, -100.0)
        b = _offset(*a, east_m=100)
        clean, removed = self.dedup.run(_df([a, b]))
        self.assertEqual(len(clean), 1)
        self.assertEqual(len(removed), 1)

    def test_far_pair_survives(self):
        a = (40.0, -100.0)
        b = _offset(*a, east_m=1000)
        clean, _ = self.dedup.run(_df([a, b]))
        self.assertEqual(len(clean), 2)

    def test_different_asset_types_never_collapse(self):
        a = (40.0, -100.0)
        b = _offset(*a, east_m=10)          # 10 m apart, but different classes
        df = pd.DataFrame([
            {"asset_id": "osm_node_1", "asset_type": SUB, "lat": a[0], "lon": a[1]},
            {"asset_id": "osm_node_2", "asset_type": "energy.generation.solar_farm",
             "lat": b[0], "lon": b[1]},
        ])
        clean, removed = self.dedup.run(df)
        self.assertEqual(len(clean), 2)
        self.assertTrue(removed.empty)

    def test_single_asset_passes_through(self):
        clean, removed = self.dedup.run(_df([(12.0, 34.0)]))
        self.assertEqual(len(clean), 1)
        self.assertTrue(removed.empty)

    def test_removed_rows_reference_a_survivor(self):
        a = (40.0, -100.0)
        b = _offset(*a, east_m=50)
        clean, removed = self.dedup.run(_df([a, b]))
        self.assertIn("deduplicated_by", removed.columns)
        self.assertIn(removed["deduplicated_by"].iloc[0], set(clean["asset_id"]))

    def test_counts_are_conserved(self):
        rng = np.random.default_rng(0)
        pts = [(40 + rng.uniform(-1, 1), -100 + rng.uniform(-1, 1))
               for _ in range(200)]
        df = _df(pts)
        clean, removed = self.dedup.run(df)
        self.assertEqual(len(clean) + len(removed), len(df))


class TestHighLatitudeRegression(unittest.TestCase):
    """the KDTree bug: an east-west pair inside the threshold was missed at
    high latitude, because a degree of longitude is shorter there than the
    threshold/111320 conversion assumed. 150 m at latitude 60 is 0.0027 deg
    of longitude, while the old radius was 0.0018 deg, so the pair survived
    when it should have collapsed."""

    def setUp(self):
        self.dedup = Deduplicator(distance_threshold_m=200)

    def test_east_west_pair_collapses_at_latitude_60(self):
        a = (60.0, 10.0)
        b = _offset(*a, east_m=150)
        self.assertLess(_haversine_m(*a, *b), 200)
        clean, _ = self.dedup.run(_df([a, b]))
        self.assertEqual(len(clean), 1, "high-latitude E-W pair was not deduplicated")

    def test_east_west_pair_collapses_at_latitude_70(self):
        a = (70.0, 25.0)
        b = _offset(*a, east_m=180)
        self.assertLess(_haversine_m(*a, *b), 200)
        clean, _ = self.dedup.run(_df([a, b]))
        self.assertEqual(len(clean), 1)

    def test_behaviour_matches_across_latitudes(self):
        """the same metre separation must give the same verdict everywhere."""
        for lat in (0.0, 30.0, 45.0, 60.0, 75.0):
            with self.subTest(lat=lat):
                near = _offset(lat, 5.0, east_m=150)
                far = _offset(lat, 5.0, east_m=400)
                n_near, _ = self.dedup.run(_df([(lat, 5.0), near]))
                n_far, _ = self.dedup.run(_df([(lat, 5.0), far]))
                self.assertEqual(len(n_near), 1, f"150 m pair survived at {lat}")
                self.assertEqual(len(n_far), 2, f"400 m pair collapsed at {lat}")

    def test_isotropy_north_south_versus_east_west(self):
        """a 150 m separation should collapse regardless of bearing."""
        lat = 60.0
        ew = _offset(lat, 10.0, east_m=150)
        ns = _offset(lat, 10.0, north_m=150)
        for label, other in (("east-west", ew), ("north-south", ns)):
            with self.subTest(bearing=label):
                clean, _ = self.dedup.run(_df([(lat, 10.0), other]))
                self.assertEqual(len(clean), 1)


class TestOntologyThresholds(unittest.TestCase):
    """per-class thresholds come from the ontology; solar farms are physically
    much larger than substations and use a wider radius."""

    def test_solar_farm_uses_its_own_threshold(self):
        from curation.ontology import get_class_by_name
        solar = get_class_by_name("energy.generation.solar_farm")
        if solar.dedup_distance_m is None or solar.dedup_distance_m <= 200:
            self.skipTest("solar_farm does not declare a wider dedup radius")
        a = (10.0, 20.0)
        b = _offset(*a, east_m=int(solar.dedup_distance_m * 0.5))
        df = pd.DataFrame([
            {"asset_id": "osm_way_1", "asset_type": "energy.generation.solar_farm",
             "lat": a[0], "lon": a[1]},
            {"asset_id": "osm_way_2", "asset_type": "energy.generation.solar_farm",
             "lat": b[0], "lon": b[1]},
        ])
        # far beyond the 200 m fallback, but inside the class's own radius
        clean, _ = Deduplicator(distance_threshold_m=200).run(df)
        self.assertEqual(len(clean), 1)


if __name__ == "__main__":
    unittest.main()
