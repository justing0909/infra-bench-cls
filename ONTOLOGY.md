# Infrastructure Asset Ontology

Defines the asset taxonomy used for OSM querying, labeling, and corpus
construction. OSM tags are the primary source of asset locations globally.

This ontology is a working document that will be refined as the pipeline
scales and as additional data sources are incorporated.

---

# Status by sector

| Sector | Status | Notes |
|---|---|---|
| Energy | Active | See full hierarchy below |
| Transport | Draft | Initial asset classes defined |
| Water | Draft | Initial asset classes defined |
| Telecom | Draft | Initial asset classes defined |

---

# Inclusion standard: system role / infrastructure scale

In addition to `asset_type`, each asset should be interpreted through a second
axis: its **role in the infrastructure system** and the **scale at which it
operates**.

This helps standardize inclusion decisions across domains and prevents small
building-level or component-level features from overwhelming the corpus.

## System role classes

| Class | Meaning | Default corpus behavior |
|---|---|---|
| `core_infra` | Utility-scale, network-critical, or regionally meaningful infrastructure assets | Include |
| `distributed_edge` | Building-scale, customer-side, or highly local infrastructure assets | Exclude by default |
| `component_only` | Subcomponents of a larger mapped facility, not meaningful as standalone assets | Collapse upward or exclude |
| `non_visual_or_too_fine` | Assets that are underground, too small, or not reliably visible in imagery | Exclude from imagery corpus |

## Design rule

The core imagery corpus should prioritize **facility-scale infrastructure**.

Assets labeled `distributed_edge` should generally be excluded from the main
training corpus unless a future task explicitly studies distributed systems.

Assets labeled `component_only` should not be treated as standalone training
examples when a higher-level facility representation exists.

## Examples by domain

### Energy
- `core_infra`: substations, utility-scale solar farms, wind farms, power plants  
- `distributed_edge`: rooftop solar, small private backup generators  
- `component_only`: individual solar panel blocks inside a mapped solar plant  

### Transport
- `core_infra`: airports, rail yards, major train stations, ports, major interchanges  
- `distributed_edge`: private parking lots, small building-level transport features  
- `component_only`: internal subfeatures of a larger transport facility  

### Water
- `core_infra`: water treatment plants, wastewater plants, reservoirs, major pumping stations  
- `distributed_edge`: private wells, building-scale tanks, small local cisterns  
- `component_only`: internal basins or process units within a larger plant  

### Telecom
- `core_infra`: communication towers, switching sites, data centers  
- `distributed_edge`: rooftop antennas serving a single building, home dishes  
- `component_only`: internal hardware or yard subcomponents  

---

# Energy

Hierarchy: `energy → generation / transmission / distribution / other`

Voltage threshold assumption: transmission = ≥69kV, distribution = <69kV,
which is the standard US convention and may vary internationally.

---

# Generation

| Asset Type | OSM Tags | Visual Confidence | Notes |
|---|---|---|---|
| `energy.generation.power_plant` | `power=plant` | High | Large footprint, distinctive from overhead |
| `energy.generation.generator` | `power=generator` | Medium | Varies by type |
| `energy.generation.solar_farm` | facility-scale solar plant or mapped solar site | High | `core_infra`, utility-scale generation facility |
| `energy.generation.solar_facility_inferred` | clustered `power=generator` + `generator:source=solar` features with no enclosing plant/site | Medium–High | `core_infra` when cluster evidence is strong |
| `energy.generation.solar_rooftop_distributed` | `power=generator` + `generator:source=solar` + `location=roof` | Low | `distributed_edge`, excluded from core corpus |
| `energy.generation.solar_component` | individual solar generator polygons inside a facility | Low | `component_only`, collapsed upward |
| `energy.generation.wind_farm` | `power=generator` + `generator:source=wind` | High | Distinctive turbine pattern |

---

# Transmission

| Asset Type | OSM Tags | Visual Confidence | Notes |
|---|---|---|---|
| `energy.transmission.substation` | `power=substation` + `substation=transmission` | High | Large switching yard |
| `energy.transmission.line` | `power=line` + `voltage>=69000` | Medium | Linear infrastructure |
| `energy.transmission.tower` | `power=tower` | Low–Medium | Small asset |

---

# Distribution

| Asset Type | OSM Tags | Visual Confidence | Notes |
|---|---|---|---|
| `energy.distribution.substation` | `power=substation` + `substation=distribution` | High | Smaller switching facility |
| `energy.distribution.substation_untyped` | `power=substation` | Medium | Subtype often missing in OSM |
| `energy.distribution.line` | `power=line` + `voltage<69000` | Low | Edge infrastructure |
| `energy.distribution.transformer` | `power=transformer` | Low | Often below imagery resolution |
| `energy.distribution.pole` | `power=pole` | Low | Extremely small |

---

# Other

| Asset Type | OSM Tags | Visual Confidence | Notes |
|---|---|---|---|
| `energy.other.cable` | `power=cable` | Low | Underground |
| `energy.other.switch` | `power=switch` | Low | Too small for most imagery |

---

# Transport

| Asset Type | OSM Tags | Visual Confidence | Notes |
|---|---|---|---|
| `transport.airport` | `aeroway=aerodrome` | High | Regional aviation infrastructure |
| `transport.train_station` | `railway=station` | High | Passenger rail node |
| `transport.rail_yard` | `railway=yard` | High | Freight logistics facility |
| `transport.port_terminal` | `harbour=yes` or port relations | High | Maritime logistics infrastructure |
| `transport.highway_interchange` | motorway junction features | Medium | Network junction |
| `transport.parking` | `amenity=parking` | Low | Local infrastructure |
| `transport.driveway` | `service=driveway` | Low | Building-scale access |

---

# Water

| Asset Type | OSM Tags | Visual Confidence | Notes |
|---|---|---|---|
| `water.water_works` | `man_made=water_works` | High | Water works facility |
| `water.wastewater.plant` | `man_made=wastewater_plant` | High | Wastewater processing |
| `water.reservoir` | `landuse=reservoir` | High | System-scale storage |
| `water.pumping_station` | `man_made=pumping_station` | Medium | Infrastructure node |
| `water.tower` | `man_made=water_tower` | Medium | Water storage |
| `water.private_well` | `man_made=water_well` | Low | Building-scale |
| `water.treatment_basin` | internal plant basins | Low | Component-level feature |

---

# Telecom

| Asset Type | OSM Tags | Visual Confidence | Notes |
|---|---|---|---|
| `telecom.communication_tower` | `man_made=tower` | High | Telecom infrastructure node |
| `telecom.exchange` | `telecom=exchange` | High | Switching facility |
| `telecom.data_center` | `building=data_center` | High | Digital infrastructure |
| `telecom.broadcast_site` | `tower:type=broadcast` | High | Broadcast infrastructure |
| `telecom.rooftop_antenna` | rooftop telecom equipment | Low | Building-scale |
| `telecom.home_dish` | satellite dishes | Low | Residential |

---

# Notes on OSM tag consistency

- `substation=transmission` vs `substation=distribution` is not consistently
  tagged across OSM. Many substations are mapped as `power=substation` only,
  with no subtype. The pipeline handles these as
  `energy.distribution.substation_untyped`.

- Visual confidence ratings are qualitative estimates based on 30 cm imagery
  (Maxar). At Sentinel-2 resolution (10 m), confidence drops one level for
  most assets.

- Assets rated **Low visual confidence** (poles, cables, switches, underground
  infrastructure) are included for ontology completeness but may be filtered
  during QC due to poor visual distinguishability.

- Solar infrastructure in OSM is frequently mapped hierarchically. A single
  solar facility may appear as:
  - a plant polygon or site relation
  - many individual `power=generator` solar polygons
  - rooftop solar installations

  For corpus construction:

  - rooftop solar (`location=roof`) is classified as `distributed_edge`
  - solar generators inside mapped facilities are treated as `component_only`
  - standalone generator clusters may be collapsed into
    `energy.generation.solar_facility_inferred`

This ensures the corpus represents **facility-level infrastructure** rather
than individual solar panel geometries.
