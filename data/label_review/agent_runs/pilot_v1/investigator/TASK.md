# Investigator

Each tile is a 600 m x 600 m area centred on a mapped asset. Judge the asset at the centre of the tile, not the surroundings. A tile shows one asset class; context (roads, rail, fields) can help but does not decide it.

You verify a candidate label. Each packet row gives the OSM label and cached evidence: the Sentinel-2 chip (10 m, what the benchmark models see), NAIP aerial imagery for US tiles (about 0.6 m, with its capture date), the OSM name and tags, secondary-source evidence (HIFLD transmission lines, EIA power plants, official station and airport registers) and the mapped footprint size. Reconcile conflicts. A missing record is not negative evidence; dates can disagree. Cite only the channels you used.

Actions: `confirm` (the OSM label is right), `relabel` (it is wrong; give `label`, one of the classes below), `request_review` (you cannot tell).

## Classes

- `energy.distribution.other`: a substation whose type OSM does not state (small or minor substations)
- `energy.distribution.substation`: distribution substation: steps transmission down to local distribution; small fenced yard, transformers, pole-mounted feeders leaving
- `energy.generation.power_plant`: power station that is not solar or wind (gas, coal, hydro, nuclear, diesel, biomass)
- `energy.generation.solar_farm`: ground-mounted utility-scale solar array
- `energy.generation.wind_farm`: group of wind turbines
- `energy.transmission.substation`: high-voltage substation (roughly 100 kV and up): large switchyard, lattice-tower lines entering
- `telecom.data_center`: data centre building
- `transport.airport`: airport or airfield with runways
- `transport.port_terminal`: port or harbour terminal with quays
- `transport.train_station`: passenger railway station on a rail line
- `water.storage_tank`: water storage tank or reservoir tank
- `water.treatment.plant`: drinking-water treatment works
- `water.wastewater.plant`: sewage / wastewater treatment works: round clarifiers, aeration basins
- `not_infrastructure`: none of the above at this location

## Rules

Use a fresh session. Read only `packet.json` in this directory and the image files it names, and actually view every image. Never read anything else in the repository: not `data/label_review/decisions_*`, `queue.csv`, `auto_accept_audit.csv`, `class_sample.csv`, other runs, or the other role's directory. If you cannot view an image, set `image_quality` to `poor` and use `request_review`. Confidence is an uncalibrated self-report from 0 to 1, not a probability. `image_quality` is `adequate` or `poor`.

Save a JSON array as `decisions.json` in this directory, exactly one row per packet id:

```json
[{"id": "osm_way_1", "action": "relabel", "label": "energy.generation.solar_farm", "confidence": 0.9, "image_quality": "adequate", "evidence_ids": ["osm_way_1:naip", "osm_way_1:osm"], "reason": "Rows of solar panels fill the centre; tagged as an untyped substation."}]
```
