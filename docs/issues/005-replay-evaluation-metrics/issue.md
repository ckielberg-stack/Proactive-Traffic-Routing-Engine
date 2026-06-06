# Issue 005 - Add Replay-Based Prediction Evaluation Metrics

## Problem

TravelTime calibration currently scores at corridor level: any prediction during any congested TravelTimeRoute segment can count as a hit. This does not measure ETA accuracy, distance error, or false alarms.

## Impact

The project cannot reliably compare prediction changes, tune constants, or validate whether VMS recommendations are early and correct.

## Suggested Approach

- Define replay fixtures from persisted tick data with capacity states, sensor readings, travel times, VMS statuses, and predictions.
- Add metrics for ETA error, queue-tail distance error, precision/recall, false positive rate, missed congestion, and lead time before proxy VMS activation.
- Build a command that replays historical JSONL data through the evaluator.
- Keep metrics versioned so model changes can be compared.

## Acceptance Criteria

- A replay command runs without live API access.
- Evaluation reports per-segment and corridor-level metrics.
- Tests cover successful hit, late hit, early false positive, missed congestion, and expired prediction cases.
- A baseline metrics artifact can be generated from sample data.

## References

- `src/evaluation_logger.py`
- `src/travel_time_calibrator.py`
- `data/*/sensor_data.jsonl`
- `tests/test_evaluation_logger.py`
