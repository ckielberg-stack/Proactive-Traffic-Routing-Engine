# 2026-06-28 - DATEX Safety Context Source Boundary

## Context

TRAFIK-033 expanded the DATEX II-style operator export so weather-derived road
surface risk and authoritative Situation accident/roadwork deviations reach
downstream consumers alongside incidents and VMS speed-management recommendations.

## What I Learned

The safest DATEX source boundary is the tick's derived safety context, not each
raw upstream feed independently. `WeatherAdjustment` already applies the
conservative worst-of rule across WeatherMeasurepoint, RoadCondition, and SMHI,
so exporting one corridor weather situation avoids duplicate or conflicting
warnings when multiple feeds describe the same slippery-road risk.

Situation accident/roadwork deviations need to remain separate from
SPEEDMANAGEMENTID VMS proxy records. Accident and roadwork records describe
capacity loss and safety hazards; speed-management records only describe whether
a human operator has already acted on a VMS recommendation.

## Reuse Rules

- Export weather DATEX records from `WeatherAdjustment`, with RoadCondition and
  SMHI details as supporting context only.
- Emit at most one corridor weather safety situation per snapshot; do not create
  one record per raw weather source unless the product explicitly needs station
  granularity.
- Keep Situation accident/roadwork records in safety-situation export paths, not
  proxy ground-truth matching paths.
- Treat the current DATEX output as a PTRE-style XML exchange until real NTS XSD
  schemas and a downstream consumer are present.

## Failure Signals

- DATEX contains multiple slippery-road records for one tick because RoadCondition
  and SMHI were exported independently.
- An accident or roadwork deviation changes `operatorActionStatus` for a VMS
  recommendation.
- DATEX tests pass by string search but the XML is not well-formed.
- Dry weather with no proactive HALKA emits a weather safety record.

## Next Checklist

- Parse DATEX XML in tests whenever adding new record blocks.
- Reset operator API safety context between tests so weather or Situation records
  do not leak across endpoint cases.
- Revisit formal XSD validation only when the repo has the target DATEX/NTS schema
  bundle or a real downstream ingest contract.
