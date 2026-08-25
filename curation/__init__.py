"""
curation
--------
shared curation machinery for the two Infra-Bench CLS dataset parts.

modules here are used by both parts:
  ontology        13-class taxonomy and OSM tag matchers
  sources         PBF extraction via pyosmium
  deduplication   BallTree + haversine spatial dedup
  stac_imagery    Planetary Computer tile fetcher
  qc              tile quality control
  triage          rule-based confidence triage
  dataset         manifest + images assembly
  helpers         shared dataclasses and the modality registry
  utils           IO, spatial blocking, timing log

the two parts have their own subpackages:
  sectors/        the sampled cross-sector benchmark dataset
  substations/    the full unsampled substation dataset
"""
