"""
two_agent.py
------------
Two independent agents check OSM labels, and only their confident agreement
becomes a provisional label. Adapted from Ed Oughton's week-10 tower workflow
(github.com/edwardoughton/Agentic-GeoAI, week10/scripts/two_agent.py): the
routing rule and the scoring are kept as he wrote them so the numbers compare;
the roles, evidence and imagery are Infra-Bench's.

  investigator  sees the Sentinel-2 chip, NAIP aerial imagery for US tiles,
                the OSM label, name and tags, the secondary-source evidence
                (HIFLD, EIA, registers) and the footprint, and answers
                confirm / relabel / request_review.
  reviewer      sees only the imagery. It is never shown the OSM label, the
                name or any evidence, and names the class itself from the
                image (or request_review).

A tile gets a provisional label only when both agree on the class, neither
abstains, both call the imagery adequate, and both confidences reach the
threshold (0.85 by default, uncalibrated). Everything else goes to a person.
Agreement is not proof, especially when both roles run the same model.

Each role runs in its own fresh agent session that reads only its packet and
the images it names. Packets never contain human decisions, and finalize
never writes to the decisions logs: provisional labels stay in the run.

    python -m curation.review.two_agent prepare --run pilot_v1 --ids ...
    # run each role's TASK.md in a fresh agent session
    python -m curation.review.two_agent finalize --run pilot_v1
    python -m curation.review.two_agent score --run pilot_v1 --reviewer justin
"""

import argparse
import hashlib
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..paths import DATA_DIR, REVIEW_DIR, REVIEW_QUEUE
from . import decisions
from .hifld_lines import in_us
from .via import NOT_INFRA, chip_for

RUNS_DIR = REVIEW_DIR / "agent_runs"
FOOTPRINTS = DATA_DIR / "asset_footprints" / "asset_footprint_share_v1.csv"
HALF_TILE_M = 300
NAIP_PX = 768
ACTIONS = {"investigator": ("confirm", "relabel", "request_review"),
           "reviewer": ("classify", "request_review")}

# what each class means, for both roles: the rules the human reviewer settled
# on in the label review (2026-11-14/15), so agents and people label alike.
CLASS_GUIDE = {
    "energy.transmission.substation": "substation whose main job is the transmission network: 3+ high-voltage lines meet there, "
                                      "or it feeds other substations at subtransmission voltage (roughly 60-138 kV out), or it is "
                                      "fed at 220 kV and up. Large yard, many bays, lattice towers or H-frames. Subtransmission counts here",
    "energy.distribution.substation": "substation fed by 1-2 lines (often a radial spur or one line looping through) whose main output is "
                                      "distribution feeders (about 36 kV and below, e.g. 115/12, 66/22, 33/11 kV): compact fenced "
                                      "yard, 1-3 transformers, wooden-pole feeders leaving (underground in cities)",
    "energy.distribution.other": "a real substation whose type cannot be determined, or a pad-mounted transformer, kiosk "
                                 "or small electrical service building",
    "energy.generation.power_plant": "power station that is not solar or wind (gas, coal, oil, diesel, hydro, nuclear, "
                                     "geothermal, biomass) or a battery storage site: stacks, turbine halls, engine rows, fuel "
                                     "tanks, a dam powerhouse, or rows of battery containers",
    "energy.generation.solar_farm": "ground-mounted utility-scale solar array",
    "energy.generation.wind_farm": "group of wind turbines (pads joined by access roads, long tower shadows)",
    "water.wastewater.plant": "sewage or wastewater treatment works: round clarifiers with arms, rectangular aeration tanks, "
                              "oxidation ditches (oval loops), trickling filters, digesters, drying beds, or a chain of "
                              "stabilisation ponds at a town's edge; brown, dark or green water. Industrial effluent plants count",
    "water.treatment.plant": "drinking-water treatment works (OSM water_works): rectangular settling basins and a row of filter "
                             "cells, or a large treatment building with an intake, chemical tanks and covered clear-water reservoirs. "
                             "Pump stations, wells, offices, depots and outfalls are not treatment plants",
    "water.storage_tank": "any real water tank: ground reservoir tanks, water towers and standpipes, and also household, "
                          "farm and single-facility tanks. Fuel or gas tanks, silos, chimneys, lighthouses and lookout towers are not",
    "transport.train_station": "passenger railway station on a rail line",
    "transport.airport": "airport or airfield with runways",
    "transport.port_terminal": "port or harbour terminal with quays",
    "telecom.data_center": "data centre: large plain windowless building with heavy rooftop or yard cooling, rows of backup "
                           "generators, often its own substation. Warehouses, offices, telephone exchanges and cable "
                           "landing stations are not, unless evidence names a data centre",
    NOT_INFRA: "none of the above at this location, including pump stations, wells, farm or irrigation ponds, manure "
               "lagoons, fish or salt ponds, fuel tanks, and abandoned or demolished facilities",
}

# rules for cases that cross classes, given to both roles.
LABELLING_RULES = (
    "- Judge the facility the mapped asset belongs to. An outline covering one part of a larger site (one clarifier, one "
    "building) takes the site's class. A power plant's switchyard mapped on its own is a substation, labelled by the "
    "voltage at which it connects to the grid: 69 kV and up is transmission, 35 kV and below distribution.\n"
    "- For substations, the highest-level job wins: a 138/69/23 kV hub is transmission; a 161 kV spur ending at a yard "
    "that feeds wooden-pole feeders is distribution, even though high voltage comes in.\n"
    "- A facility visibly abandoned (dry or overgrown basins, missing equipment) before about 2024 is not "
    "infrastructure. A facility built after the Sentinel-2 image keeps the class of what is there now; say so in `reason`.\n"
    "- Always give your best guess with an honest confidence, even when the imagery is limited: a low confidence "
    "is fine and expected, and the confidence decides which tiles a person checks. Use `request_review` and "
    "`image_quality: poor` only when an image is blank, corrupted or cloud-covered, not because the asset is small.")

TARGET = ("Each tile is a 600 m x 600 m area centred on a mapped asset. Judge the asset at the "
          "centre of the tile, not the surroundings. A tile shows one asset class; context "
          "(roads, rail, fields) can help but does not decide it.")

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _naip(lat: float, lon: float, out: Path) -> dict | None:
    """latest NAIP RGB covering the tile, saved as a PNG. US only."""
    import planetary_computer
    import pystac_client
    from ..stac_imagery import STAC_ENDPOINT, _windowed_read_multiband

    dlat = HALF_TILE_M / 111320
    dlon = HALF_TILE_M / (111320 * math.cos(math.radians(lat)))
    bbox = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    catalog = pystac_client.Client.open(STAC_ENDPOINT)
    from shapely.geometry import box, shape

    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    # Planetary Computer throttles at times; retry the search before giving up
    for attempt in range(4):
        try:
            items = list(catalog.search(collections=["naip"], bbox=bbox, limit=20).items())
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    tile = box(*bbox)
    # prefer scenes that cover the whole tile, newest first; a scene that only
    # covers part of it is still used, padded rather than stretched, so the
    # asset stays at the centre and the geometry is undistorted.
    when = lambda i: str(i.datetime or i.properties.get("start_datetime", ""))
    items = sorted(items, key=lambda i: (shape(i.geometry).contains(tile), when(i)), reverse=True)
    for item in items:
        data = None
        for attempt in range(3):
            try:
                signed = planetary_computer.sign(item)
                with rasterio.open(signed.assets["image"].href) as src:
                    native = transform_bounds(CRS.from_epsg(4326), src.crs, *bbox)
                    window = rasterio.windows.from_bounds(*native, transform=src.transform)
                    data = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        if data is None:
            continue
        if data.size == 0 or not data.any():
            continue
        covered = round(float((data.max(axis=0) > 0).mean()), 3)
        rgb = np.moveaxis(data, 0, -1).astype(np.uint8)
        Image.fromarray(rgb).resize((NAIP_PX, NAIP_PX), Image.LANCZOS).save(out)
        return {"path": str(out), "date": str(item.datetime.date() if item.datetime else ""),
                "source_id": item.id, "resolution_m": item.properties.get("gsd"),
                "tile_covered": covered}
    return None


def _osm_tags(ids: list) -> dict:
    parts = pd.Series(ids).str.extract(r"osm_(node|way|relation)_(\d+)")
    out = {}
    for kind, g in parts.groupby(0):
        query = f"[out:json][timeout:120];{kind}(id:{','.join(g[1])});out tags;"
        # Overpass is busy at times (502/504): retry, then try a mirror
        for attempt, url in enumerate(["https://overpass-api.de/api/interpreter"] * 3 +
                                      ["https://overpass.kumi.systems/api/interpreter"] * 2):
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": query}).encode(),
                                         headers={"User-Agent": "infra-bench-cls two-agent review"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    elements = json.load(r)["elements"]
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(10 * (attempt + 1))
        for el in elements:
            out[f"osm_{el['type']}_{el['id']}"] = el.get("tags", {})
    return out


def _task(role: str, classes: list, naip: bool = True) -> str:
    guide = "\n".join(f"- `{c}`: {CLASS_GUIDE.get(c, '')}" for c in classes)
    if role == "investigator":
        role_text = (
            "You verify a candidate label. Each packet row gives the OSM label and cached evidence: "
            "the Sentinel-2 chip (10 m, what the benchmark models see), "
            + ("NAIP aerial imagery for US tiles (about 0.6 m, with its capture date), " if naip else
               "no higher-resolution imagery (this run uses Sentinel-2 only, so many small assets are "
               "hard to see; give your best guess with an honest, possibly low, confidence), ")
            + "the OSM name and tags, secondary-source "
            "evidence (HIFLD transmission lines, EIA power plants, official station and airport "
            "registers, grid operators' substation lists and lines, World Bank grid maps) and the "
            "mapped footprint size. Lines starting with Caution ask you to check, not to overrule. Reconcile conflicts. A missing record is "
            "not negative evidence; dates can disagree. Cite only the channels you used.\n\n"
            "Actions: `confirm` (the OSM label is right), `relabel` (it is wrong; give `label`, one "
            "of the classes below), `request_review` (only when the images are unreadable).")
        example = ('[{"id": "osm_way_1", "action": "relabel", "label": "energy.generation.solar_farm", '
                   '"confidence": 0.9, "image_quality": "adequate", "evidence_ids": '
                   '["osm_way_1:naip", "osm_way_1:osm"], "reason": "Rows of solar panels fill the centre; '
                   'tagged as an untyped substation."}]')
    else:
        role_text = (
            "You classify independently. Each packet row gives only imagery: the Sentinel-2 chip "
            + ("(10 m) and, for US tiles, NAIP aerial imagery (about 0.6 m). " if naip else
               "(10 m) only; many small assets are hard to see at this resolution, so give your best guess "
               "with an honest, possibly low, confidence. ")
            + "You are not told the "
            "candidate label and must not look for it. Name the class you see at the centre.\n\n"
            "Actions: `classify` (give `label`, one of the classes below), `request_review` "
            "(only when the images are unreadable). Give a short visible observation, not a reasoning trace.")
        example = ('[{"id": "osm_way_1", "action": "classify", "label": "energy.generation.solar_farm", '
                   '"confidence": 0.9, "image_quality": "adequate", "evidence_ids": ["osm_way_1:naip"], '
                   '"reason": "Rows of panels across the centre."}]')
    return (f"# {role.title()}\n\n{TARGET}\n\n{role_text}\n\n## Classes\n\n{guide}\n\n"
            f"## Labelling rules\n\n{LABELLING_RULES}\n\n## Rules\n\n"
            "Use a fresh session. Read only `packet.json` in this directory and the image files it "
            "names, and actually view every image. Never read anything else in the repository: not "
            "`data/label_review/decisions_*`, `queue.csv`, `auto_accept_audit.csv`, `class_sample.csv`, "
            "other runs, or the other role's directory. If you cannot view an image, set "
            "`image_quality` to `poor` and use `request_review`. Confidence is an uncalibrated "
            "self-report from 0 to 1, not a probability. `image_quality` is `adequate` or `poor`.\n\n"
            "Save a JSON array as `decisions.json` in this directory, exactly one row per packet id:\n\n"
            f"```json\n{example}\n```\n")


def prepare(run: str, ids: list, naip: bool = True, batch: int = 0) -> Path:
    """packets for both roles. naip=False keeps every tile to Sentinel-2, so all
    regions are judged on the same imagery. batch > 0 splits each role's
    packet into folders of that many tiles, one agent session each."""
    run = RUNS_DIR / run
    if run.exists():
        raise ValueError(f"{run} exists; use a new run name so earlier runs are preserved")
    if len(ids) != len(set(ids)):
        raise ValueError("ids must be distinct")
    q = pd.read_csv(REVIEW_QUEUE, keep_default_na=False).set_index("asset_id")
    missing = [i for i in ids if i not in q.index]
    if missing:
        raise ValueError(f"not in the queue: {missing[:5]}")
    classes = sorted(q.asset_type.unique()) + [NOT_INFRA]
    fp = pd.read_csv(FOOTPRINTS, keep_default_na=False).set_index("asset_id") if FOOTPRINTS.exists() else None
    tags = _osm_tags(ids)

    images = run / "images"
    images.mkdir(parents=True)
    packets = {"investigator": [], "reviewer": []}
    for sid in ids:
        row = q.loc[sid]
        lat, lon = float(row.lat), float(row.lon)
        chip = chip_for({"asset_id": sid, "image": row.image})
        sentinel = images / f"{sid}_sentinel2.png"
        sentinel.write_bytes(chip.read_bytes())
        imagery = {"sentinel2": {"path": str(sentinel), "resolution_m": 10, "date": "2024 composite"}}
        us = bool(in_us(lat, lon))
        if us and naip:
            naip = _naip(lat, lon, images / f"{sid}_naip.png")
            if naip:
                imagery["naip"] = naip
        neutral = {"id": sid, "imagery": imagery,
                   "evidence_ids": [f"{sid}:{k}" for k in imagery]}
        packets["reviewer"].append(neutral)

        footprint = None
        if fp is not None and sid in fp.index and fp.loc[sid].status == "polygon":
            footprint = {"extent_m": float(fp.loc[sid].extent_m), "area_m2": float(fp.loc[sid].area_m2)}
        evidence = {
            "osm": {"label": row.asset_type, "name": row["name"], "tags": tags.get(sid),
                    "available": sid in tags},
            "sources": {"text": row.evidence, "available": bool(row.evidence)},
            "footprint": {"value": footprint, "available": footprint is not None},
        }
        packets["investigator"].append({**neutral, "evidence": evidence,
                                        "evidence_ids": neutral["evidence_ids"] +
                                        [f"{sid}:{k}" for k in evidence]})

    for role, rows in packets.items():
        chunks = [rows[i:i + batch] for i in range(0, len(rows), batch)] if batch else [rows]
        for k, chunk in enumerate(chunks, 1):
            d = run / role / (f"batch_{k:02d}" if batch else "")
            d.mkdir(parents=True)
            (d / "packet.json").write_text(json.dumps(chunk, indent=1, ensure_ascii=False), encoding="utf-8")
            (d / "TASK.md").write_text(_task(role, classes, naip), encoding="utf-8")
    (run / "policy.json").write_text(json.dumps({
        "ids": ids, "classes": classes, "target": TARGET, "mode": "independent fresh agent sessions",
        "naip": naip, "batch": batch,
        "adapted_from": "edwardoughton/Agentic-GeoAI week10/scripts/two_agent.py"}, indent=1), encoding="utf-8")
    return run


def _validate(rows: list, ids: list, role: str, classes: list, packet: list) -> dict:
    allowed = {p["id"]: set(p["evidence_ids"]) for p in packet}
    if sorted(r.get("id") for r in rows) != sorted(ids):
        raise ValueError(f"{role}: must return exactly the packet ids; omissions are not answers")
    for r in rows:
        sid = r["id"]
        if r.get("action") not in ACTIONS[role]:
            raise ValueError(f"{role} {sid}: action must be one of {ACTIONS[role]}")
        if r["action"] in ("relabel", "classify") and r.get("label") not in classes:
            raise ValueError(f"{role} {sid}: label must be one of the classes")
        c = r.get("confidence")
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or not 0 <= c <= 1:
            raise ValueError(f"{role} {sid}: confidence must be a number from 0 to 1")
        if r.get("image_quality") not in ("adequate", "poor"):
            raise ValueError(f"{role} {sid}: image_quality must be adequate or poor")
        cites = r.get("evidence_ids")
        if not isinstance(cites, list) or not cites or not set(cites) <= allowed[sid]:
            raise ValueError(f"{role} {sid}: cite only this tile's channels")
        if not any(c.endswith((":sentinel2", ":naip")) for c in cites):
            raise ValueError(f"{role} {sid}: both roles must look at imagery")
        if not isinstance(r.get("reason"), str) or not r["reason"].strip():
            raise ValueError(f"{role} {sid}: give a short reason")
    return {r["id"]: r for r in rows}


def _role_files(folder: Path) -> tuple:
    """a role's packet rows and decision rows, merged across batch folders."""
    dirs = sorted(d for d in folder.glob("batch_*") if d.is_dir()) or [folder]
    packet, rows = [], []
    for d in dirs:
        packet += json.loads((d / "packet.json").read_text(encoding="utf-8"))
        if not (d / "decisions.json").exists():
            raise FileNotFoundError(f"{d / 'decisions.json'} is missing: run that batch first")
        rows += json.loads((d / "decisions.json").read_text(encoding="utf-8"))
    return packet, rows


def finalize(run: str, threshold: float = 0.85) -> list:
    run = RUNS_DIR / run
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    policy = json.loads((run / "policy.json").read_text(encoding="utf-8"))
    ids, classes = policy["ids"], policy["classes"]
    agents = {}
    for role in ("investigator", "reviewer"):
        packet, rows = _role_files(run / role)
        agents[role] = _validate(rows, ids, role, classes, packet)
    osm_label = {p["id"]: p["evidence"]["osm"]["label"] for p in _role_files(run / "investigator")[0]}

    predictions, queue = [], []
    for sid in ids:
        a, b = agents["investigator"][sid], agents["reviewer"][sid]
        # the class each agent stands behind
        a_cls = osm_label[sid] if a["action"] == "confirm" else a.get("label")
        b_cls = b.get("label")
        reasons = []
        if "request_review" in (a["action"], b["action"]):
            reasons.append("agent_abstention")
        elif a_cls != b_cls:
            reasons.append("agent_disagreement")
        if min(a["confidence"], b["confidence"]) < threshold:
            reasons.append("low_confidence")
        if "poor" in (a["image_quality"], b["image_quality"]):
            reasons.append("poor_imagery")
        if reasons:
            action, label = "request_review", ""
        else:
            action = "confirm" if a_cls == osm_label[sid] else "relabel"
            label = a_cls
        predictions.append({"id": sid, "osm_label": osm_label[sid], "action": action, "label": label,
                            "reason": ", ".join(reasons) or "independent agents agree above the routing threshold",
                            "investigator": a, "reviewer": b})
        if reasons:
            queue.append({"id": sid, "routing_reasons": reasons})

    (run / "predictions.json").write_text(json.dumps(predictions, indent=1, ensure_ascii=False), encoding="utf-8")
    (run / "human_review.json").write_text(json.dumps(queue, indent=1), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({
        "total": len(ids), "provisionally_labelled": len(ids) - len(queue), "human_review": len(queue),
        "threshold": threshold, "confidence_calibrated": False,
        "input_sha256": {str(f.relative_to(run)): _sha(f) for r in ("investigator", "reviewer")
                         for f in sorted((run / r).rglob("decisions.json"))},
    }, indent=1), encoding="utf-8")
    return predictions


def score(run: str, reviewer: str = "justin") -> dict:
    """coverage and selective error against a person's decisions, as in Ed's score()."""
    run = RUNS_DIR / run
    preds = json.loads((run / "predictions.json").read_text(encoding="utf-8"))
    latest = decisions.load()
    ref = {}
    for p in preds:
        e = latest.get(p["id"], {}).get(reviewer)
        if e:
            ref[p["id"]] = (None if e["decision"] == "unsure"
                            else p["osm_label"] if e["decision"] == "accept" else e["label"])
    scored = [p for p in preds if p["id"] in ref]
    auto = [p for p in scored if p["action"] != "request_review"]
    resolved = [p for p in auto if ref[p["id"]] is not None]
    wrong = [p for p in resolved if p["label"] != ref[p["id"]]]

    def agent_accuracy(role):
        rows = [(p[role], ref[p["id"]], p["osm_label"]) for p in scored if ref[p["id"]] is not None]
        named = [(d.get("label") if d["action"] in ("relabel", "classify") else
                  osm if d["action"] == "confirm" else None, r) for d, r, osm in rows]
        answered = [(x, r) for x, r in named if x is not None]
        return {"answered": len(answered), "abstained": len(named) - len(answered),
                "correct": sum(x == r for x, r in answered)}

    return {"run": run.name, "scored": len(scored), "automatic": len(auto),
            "request_review": len(scored) - len(auto),
            "coverage": round(len(auto) / len(scored), 3) if scored else None,
            "resolved_automatic": len(resolved), "wrong_automatic": len(wrong),
            "false_confirms": sum(p["action"] == "confirm" for p in wrong),
            "false_relabels": sum(p["action"] == "relabel" for p in wrong),
            "selective_error": round(len(wrong) / len(resolved), 3) if resolved else None,
            "investigator": agent_accuracy("investigator"), "reviewer": agent_accuracy("reviewer"),
            "wrong": [{"id": p["id"], "said": p["label"], "human": ref[p["id"]]} for p in wrong]}


def _routed(a: dict, b: dict, osm: str, threshold: float) -> tuple:
    """the routing rule from finalize, at any threshold: (action, label)."""
    a_cls = osm if a["action"] == "confirm" else a.get("label")
    if ("request_review" in (a["action"], b["action"]) or a_cls != b.get("label")
            or min(a["confidence"], b["confidence"]) < threshold or "poor" in (a["image_quality"], b["image_quality"])):
        return "request_review", ""
    return ("confirm" if a_cls == osm else "relabel"), a_cls


def calibrate(run: str, split_csv: str, reviewer: str = "justin", target_error: float = 0.05) -> dict:
    """choose the confidence threshold on the calibration half of a spatially
    blocked split, then report coverage and selective error on the held-out half
    at that threshold, overall and by class and region. Also lists the held-out
    and calibration tiles where both agents agree with each other, confidently,
    against the person, for adjudication."""
    run_dir = RUNS_DIR / run
    policy = json.loads((run_dir / "policy.json").read_text(encoding="utf-8"))
    ids, classes = policy["ids"], policy["classes"]
    agents = {r: _validate(_role_files(run_dir / r)[1], ids, r, classes, _role_files(run_dir / r)[0])
              for r in ("investigator", "reviewer")}
    osm = {p["id"]: p["evidence"]["osm"]["label"] for p in _role_files(run_dir / "investigator")[0]}
    latest = decisions.load()
    ref = {}
    for sid in ids:
        e = latest.get(sid, {}).get(reviewer)
        if e and e["decision"] != "unsure":
            ref[sid] = osm[sid] if e["decision"] == "accept" else e["label"]
    split = pd.read_csv(split_csv).set_index("asset_id")

    def metrics(sids: list, t: float) -> dict:
        rows = [(s, *_routed(agents["investigator"][s], agents["reviewer"][s], osm[s], t)) for s in sids if s in ref]
        auto = [(s, lab) for s, act, lab in rows if act != "request_review"]
        wrong = [s for s, lab in auto if lab != ref[s]]
        return {"tiles": len(rows), "automatic": len(auto),
                "coverage": round(len(auto) / len(rows), 3) if rows else None,
                "wrong": len(wrong), "selective_error": round(len(wrong) / len(auto), 3) if auto else None}

    calib = [s for s in ids if split.loc[s, "half"] == "calibration"]
    held = [s for s in ids if split.loc[s, "half"] == "heldout"]
    sweep = {round(t, 2): metrics(calib, t) for t in np.arange(0.50, 1.0001, 0.05)}
    ok = [t for t, m in sweep.items() if m["selective_error"] is not None and m["selective_error"] <= target_error]
    chosen = min(ok) if ok else 1.0   # the lowest threshold that meets the target keeps the most coverage
    by = {}
    for col in ("cls", "region"):
        by[col] = {g: metrics([s for s in held if split.loc[s, col] == g], chosen)
                   for g in sorted(split.loc[held, col].unique())}
    # confident agent agreement against the person, on either half, for a second human look
    disputes = []
    for s in ids:
        if s not in ref:
            continue
        act, lab = _routed(agents["investigator"][s], agents["reviewer"][s], osm[s], chosen)
        if act != "request_review" and lab != ref[s]:
            disputes.append({"id": s, "half": split.loc[s, "half"], "osm": osm[s], "human": ref[s], "agents": lab,
                             "investigator": agents["investigator"][s]["reason"], "reviewer": agents["reviewer"][s]["reason"]})
    return {"run": run, "target_error": target_error, "chosen_threshold": chosen,
            "calibration_sweep": sweep, "heldout": metrics(held, chosen),
            "heldout_by_class": by["cls"], "heldout_by_region": by["region"], "disputes": disputes}


def main():
    p = argparse.ArgumentParser(description="two independent agents check OSM labels")
    p.add_argument("stage", choices=["prepare", "finalize", "score", "calibrate"])
    p.add_argument("--run", required=True)
    p.add_argument("--ids", nargs="+")
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--reviewer", default="justin")
    p.add_argument("--no-naip", action="store_true", help="Sentinel-2 only, for every tile")
    p.add_argument("--batch", type=int, default=0, help="tiles per agent session (0: one packet)")
    p.add_argument("--ids-file", help="a text file of asset ids, one per line")
    p.add_argument("--split", help="calibrate: csv with asset_id, cls, region, half (calibration/heldout)")
    p.add_argument("--target-error", type=float, default=0.05, help="calibrate: selective error to stay under")
    args = p.parse_args()
    if args.stage == "prepare":
        ids = list(args.ids or [])
        if args.ids_file:
            ids += [l.strip() for l in Path(args.ids_file).read_text(encoding="utf-8").splitlines() if l.strip()]
        print(prepare(args.run, ids, naip=not args.no_naip, batch=args.batch))
    elif args.stage == "finalize":
        finalize(args.run, args.threshold)
        print((RUNS_DIR / args.run / "summary.json").read_text(encoding="utf-8"))
    elif args.stage == "calibrate":
        out = calibrate(args.run, args.split, args.reviewer, args.target_error)
        (RUNS_DIR / args.run / "calibration.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                             encoding="utf-8")
        print(json.dumps({k: v for k, v in out.items() if k != "disputes"}, indent=1))
        print(f"{len(out['disputes'])} confident disagreements with the person; see calibration.json")
    else:
        print(json.dumps(score(args.run, args.reviewer), indent=1))


if __name__ == "__main__":
    main()
