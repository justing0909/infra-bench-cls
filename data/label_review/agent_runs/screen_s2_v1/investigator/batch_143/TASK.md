# Investigator

Each tile is a 600 m x 600 m area centred on a mapped asset. Judge the asset at the centre of the tile, not the surroundings. A tile shows one asset class; context (roads, rail, fields) can help but does not decide it.

You verify a candidate label. Each packet row gives the OSM label and cached evidence: the Sentinel-2 chip (10 m, what the benchmark models see), no higher-resolution imagery (this run uses Sentinel-2 only, so many small assets are hard to see; give your best guess with an honest, possibly low, confidence), the OSM name and tags, secondary-source evidence (HIFLD transmission lines, EIA power plants, official station and airport registers, grid operators' substation lists and lines, World Bank grid maps) and the mapped footprint size. Lines starting with Caution ask you to check, not to overrule. Reconcile conflicts. A missing record is not negative evidence; dates can disagree. Cite only the channels you used.

Actions: `confirm` (the OSM label is right), `relabel` (it is wrong; give `label`, one of the classes below), `request_review` (only when the images are unreadable).

## Classes

- `energy.distribution.other`: a real substation whose type cannot be determined, or a pad-mounted transformer, kiosk or small electrical service building
- `energy.distribution.substation`: substation fed by 1-2 lines (often a radial spur or one line looping through) whose main output is distribution feeders (about 36 kV and below, e.g. 115/12, 66/22, 33/11 kV): compact fenced yard, 1-3 transformers, wooden-pole feeders leaving (underground in cities)
- `energy.generation.power_plant`: power station that is not solar or wind (gas, coal, oil, diesel, hydro, nuclear, geothermal, biomass) or a battery storage site: stacks, turbine halls, engine rows, fuel tanks, a dam powerhouse, or rows of battery containers
- `energy.generation.solar_farm`: ground-mounted utility-scale solar array
- `energy.generation.wind_farm`: group of wind turbines (pads joined by access roads, long tower shadows)
- `energy.transmission.substation`: substation whose main job is the transmission network: 3+ high-voltage lines meet there, or it feeds other substations at subtransmission voltage (roughly 60-138 kV out), or it is fed at 220 kV and up. Large yard, many bays, lattice towers or H-frames. Subtransmission counts here
- `telecom.data_center`: data centre: large plain windowless building with heavy rooftop or yard cooling, rows of backup generators, often its own substation. Warehouses, offices, telephone exchanges and cable landing stations are not, unless evidence names a data centre
- `transport.airport`: airport or airfield with runways
- `transport.port_terminal`: port or harbour terminal with quays
- `transport.train_station`: passenger railway station on a rail line
- `water.storage_tank`: any real water tank: ground reservoir tanks, water towers and standpipes, and also household, farm and single-facility tanks. Fuel or gas tanks, silos, chimneys, lighthouses and lookout towers are not
- `water.treatment.plant`: drinking-water treatment works (OSM water_works): rectangular settling basins and a row of filter cells, or a large treatment building with an intake, chemical tanks and covered clear-water reservoirs. Pump stations, wells, offices, depots and outfalls are not treatment plants
- `water.wastewater.plant`: sewage or wastewater treatment works: round clarifiers with arms, rectangular aeration tanks, oxidation ditches (oval loops), trickling filters, digesters, drying beds, or a chain of stabilisation ponds at a town's edge; brown, dark or green water. Industrial effluent plants count
- `not_infrastructure`: none of the above at this location, including pump stations, wells, farm or irrigation ponds, manure lagoons, fish or salt ponds, fuel tanks, and abandoned or demolished facilities

## Labelling rules

- Judge the facility the mapped asset belongs to. An outline covering one part of a larger site (one clarifier, one building) takes the site's class. A power plant's switchyard mapped on its own is a substation, labelled by the voltage at which it connects to the grid: 69 kV and up is transmission, 35 kV and below distribution.
- For substations, the highest-level job wins: a 138/69/23 kV hub is transmission; a 161 kV spur ending at a yard that feeds wooden-pole feeders is distribution, even though high voltage comes in.
- A facility visibly abandoned (dry or overgrown basins, missing equipment) before about 2024 is not infrastructure. A facility built after the Sentinel-2 image keeps the class of what is there now; say so in `reason`.
- Always give your best guess with an honest confidence, even when the imagery is limited: a low confidence is fine and expected, and the confidence decides which tiles a person checks. Use `request_review` and `image_quality: poor` only when an image is blank, corrupted or cloud-covered, not because the asset is small.

## Rules

Use a fresh session. Read only `packet.json` in this directory and the image files it names, and actually view every image. Never read anything else in the repository: not `data/label_review/decisions_*`, `queue.csv`, `auto_accept_audit.csv`, `class_sample.csv`, other runs, or the other role's directory. If you cannot view an image, set `image_quality` to `poor` and use `request_review`. Confidence is an uncalibrated self-report from 0 to 1, not a probability. `image_quality` is `adequate` or `poor`.

Save a JSON array as `decisions.json` in this directory, exactly one row per packet id:

```json
[{"id": "osm_way_1", "action": "relabel", "label": "energy.generation.solar_farm", "confidence": 0.9, "image_quality": "adequate", "evidence_ids": ["osm_way_1:naip", "osm_way_1:osm"], "reason": "Rows of solar panels fill the centre; tagged as an untyped substation."}]
```
