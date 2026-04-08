# Infrastructure Asset Ontology

Defines the asset taxonomy used for OSM querying, labeling, and corpus
construction. OSM tags are the primary source of asset locations globally.

This ontology is a working document — it will be refined as the pipeline
scales and as additional data sources are incorporated.

---

## Status by sector

| Sector | Status | Notes |
|---|---|---|
| Energy | Active | See full hierarchy below |
| Transport | Stub | To be developed |
| Water | Stub | To be developed |
| Telecom | Stub | To be developed |

---

## Energy

Hierarchy: `energy → generation / transmission / distribution / other`

Voltage threshold assumption: transmission = ≥69kV, distribution = <69kV.
This is the standard US convention — may vary internationally.

### Generation

| Asset Type | OSM Tags | Visual Confidence | Notes |
|---|---|---|---|
| `energy.generation.power_plant` | `power=plant` | High | Large footprint, distinctive from overhead |
| `energy.generation.generator` | `power=generator` | Medium | Varies by type |
| `energy.generation.solar_farm` | `power=generator` + `generator:source=solar` | High | Very distinctive spectral signature |
| `energy.generation.wind_farm` | `power=generator` + `generator:source=wind` | High | Distinctive shadow pattern |

### Transmission

| Asset Type | OSM Tags | Visual Confidence | Notes |
|---|---|---|---|
| `energy.transmission.substation` | `power=substation` + `substation=transmission` | High | Large switching yard, busbars visible at 30cm |
| `energy.transmission.line` | `power=line` + `voltage>=69000` | Medium | Edge asset — harder to classify as discrete node |
| `energy.transmission.tower` | `power=tower` | Low–Medium | Visible at 30cm, very small at Sentinel-2 (10m) |

### Distribution

| Asset Type | OSM Tags | Visual Confidence | Notes |
|---|---|---|---|
| `energy.distribution.substation` | `power=substation` + `substation=distribution` | High | Smaller than transmission substations |
| `energy.distribution.substation_untyped` | `power=substation` (no subtype tag) | Medium | Common in OSM — subtype not always mapped |
| `energy.distribution.line` | `power=line` + `voltage<69000` | Low | Edge asset |
| `energy.distribution.transformer` | `power=transformer` | Low | Likely below Sentinel-2 resolution |
| `energy.distribution.pole` | `power=pole` | Low | Likely below Sentinel-2 and NAIP resolution |

### Other

| Asset Type | OSM Tags | Visual Confidence | Notes |
|---|---|---|---|
| `energy.other.cable` | `power=cable` | Low | Underground — not visible in imagery |
| `energy.other.switch` | `power=switch` | Low | Too small for most imagery resolutions |

---

## Transport (stub)

*To be developed. Planned asset types: airports, train stations, rail yards,
highway interchanges.*

---

## Water (stub)

*To be developed. Planned asset types: water treatment plants, wastewater
facilities, pumping stations, reservoirs.*

---

## Telecom (stub)

*To be developed. Planned asset types: communication towers, data centers,
switching facilities.*

---

## Notes on OSM tag consistency

- `substation=transmission` vs `substation=distribution` is not consistently
  tagged across OSM. Many substations are mapped as `power=substation` only,
  with no subtype. The pipeline handles these as
  `energy.distribution.substation_untyped` and applies lower confidence.
- Visual confidence ratings are qualitative estimates based on 30cm (Maxar)
  resolution. At Sentinel-2 (10m), confidence drops one level for most assets.
- Assets rated "Low" visual confidence (poles, cables, switches, underground
  infrastructure) are included in the ontology for completeness but will likely
  be filtered during QC due to poor visual distinguishability.