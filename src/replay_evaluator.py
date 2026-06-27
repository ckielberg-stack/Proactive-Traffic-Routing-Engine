"""Offline replay metrics for queue prediction quality.

The live calibrator answers a narrow question: did the corridor show any
prediction while TravelTimeRoute was congested?  This module is deliberately
stricter.  It replays persisted JSONL records, matches predictions to
TravelTimeRoute congestion by route-linear span, and reports versioned metrics
that can be compared across model changes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from config import (
    E4_NORTHBOUND_CORRIDOR_LENGTH_KM,
    E4_NORTHBOUND_TRAVEL_TIME_ROUTE_IDS,
)

METRICS_VERSION = "replay-eval-v2"
CONGESTED_STATUSES = {"slow", "heavy"}
DEFAULT_PREDICTION_EXPIRY_MINUTES = 30.0
DEFAULT_LATE_HIT_TOLERANCE_MINUTES = 5.0


@dataclass(frozen=True)
class RouteSpan:
    route_id: str
    start_km: float
    end_km: float

    @property
    def midpoint_km(self) -> float:
        return (self.start_km + self.end_km) / 2.0


@dataclass(frozen=True)
class ReplayPrediction:
    prediction_id: str
    timestamp: datetime
    camera_id: str
    origin_chainage_km: float
    growth_speed_kmh: float
    lengths_at_minutes: dict[float, float]
    residual_correction_minutes: float
    corrected_eta_minutes_by_target: dict[str, float]


@dataclass(frozen=True)
class CongestionEvent:
    event_id: str
    timestamp: datetime
    route_id: str
    status: str
    delay_seconds: float


@dataclass(frozen=True)
class VMSActivation:
    timestamp: datetime
    vms_id: str


def load_jsonl_records(paths: list[Path]) -> list[dict[str, Any]]:
    """Load JSONL records from files or directories without live API access."""
    records: list[dict[str, Any]] = []
    for path in paths:
        files = _expand_input_path(path)
        for file_path in files:
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        record["_source_file"] = str(file_path)
                        records.append(record)
    return records


def evaluate_replay_records(
    records: list[dict[str, Any]],
    route_ids: list[str] | None = None,
    corridor_length_km: float = E4_NORTHBOUND_CORRIDOR_LENGTH_KM,
    prediction_expiry_minutes: float = DEFAULT_PREDICTION_EXPIRY_MINUTES,
    late_hit_tolerance_minutes: float = DEFAULT_LATE_HIT_TOLERANCE_MINUTES,
) -> dict[str, Any]:
    """Evaluate persisted records and return versioned replay metrics."""
    route_order = route_ids or E4_NORTHBOUND_TRAVEL_TIME_ROUTE_IDS
    spans = _build_route_spans(records, route_order, corridor_length_km)
    predictions = _parse_predictions(records)
    congestions = _parse_congestions(records, spans)
    activations = _parse_vms_activations(records)

    matched_prediction_ids: set[str] = set()
    matched_congestion_ids: set[str] = set()
    segment_stats = {
        route_id: {
            "congested_count": 0,
            "hit_count": 0,
            "missed_count": 0,
            "mean_eta_error_minutes": None,
            "mean_abs_eta_error_minutes": None,
            "mean_distance_error_km": None,
        }
        for route_id in spans
    }
    segment_eta_errors: dict[str, list[float]] = {route_id: [] for route_id in spans}
    segment_distance_errors: dict[str, list[float]] = {
        route_id: [] for route_id in spans
    }
    matches: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    eta_errors: list[float] = []
    corrected_eta_errors: list[float] = []
    distance_errors: list[float] = []
    lead_times: list[float] = []
    vms_lead_times: list[float] = []

    predictions_by_id = {p.prediction_id: p for p in predictions}
    for congestion in congestions:
        segment_stats[congestion.route_id]["congested_count"] += 1
        span = spans[congestion.route_id]
        candidates = [
            _score_prediction_for_congestion(
                prediction,
                congestion,
                span,
                prediction_expiry_minutes,
                late_hit_tolerance_minutes,
            )
            for prediction in predictions
        ]
        candidates = [candidate for candidate in candidates if candidate is not None]
        if not candidates:
            segment_stats[congestion.route_id]["missed_count"] += 1
            misses.append({
                "event_id": congestion.event_id,
                "timestamp": congestion.timestamp.isoformat(),
                "route_id": congestion.route_id,
                "status": "missed_congestion",
            })
            continue

        best = min(
            candidates,
            key=lambda item: (
                abs(item["eta_error_minutes"]),
                abs(item["distance_error_km"]),
            ),
        )
        matched_congestion_ids.add(congestion.event_id)
        matched_prediction_ids.add(best["prediction_id"])
        segment_stats[congestion.route_id]["hit_count"] += 1
        segment_eta_errors[congestion.route_id].append(best["eta_error_minutes"])
        segment_distance_errors[congestion.route_id].append(best["distance_error_km"])
        eta_errors.append(best["eta_error_minutes"])
        corrected_eta_errors.append(best["corrected_eta_error_minutes"])
        distance_errors.append(best["distance_error_km"])
        lead_times.append(best["lead_time_minutes"])

        activation = _find_next_activation(
            activations,
            predictions_by_id[best["prediction_id"]].timestamp,
            prediction_expiry_minutes,
        )
        if activation is not None:
            vms_lead_times.append(best["lead_time_minutes"])
            best["vms_activation_time"] = activation.timestamp.isoformat()
            best["vms_id"] = activation.vms_id

        matches.append(best)

    false_positive_predictions = [
        prediction for prediction in predictions
        if prediction.prediction_id not in matched_prediction_ids
    ]
    replay_end = _latest_timestamp(records)
    expired_prediction_ids = [
        prediction.prediction_id for prediction in false_positive_predictions
        if replay_end is not None
        and replay_end
        > prediction.timestamp + timedelta(minutes=prediction_expiry_minutes)
    ]

    for route_id, stats in segment_stats.items():
        eta_values = segment_eta_errors[route_id]
        distance_values = segment_distance_errors[route_id]
        if eta_values:
            stats["mean_eta_error_minutes"] = round(mean(eta_values), 3)
            stats["mean_abs_eta_error_minutes"] = round(
                mean(abs(value) for value in eta_values), 3
            )
        if distance_values:
            stats["mean_distance_error_km"] = round(mean(distance_values), 3)

    prediction_count = len(predictions)
    congested_count = len(congestions)
    matched_prediction_count = len(matched_prediction_ids)
    matched_congestion_count = len(matched_congestion_ids)
    false_positive_count = len(false_positive_predictions)
    missed_count = congested_count - matched_congestion_count

    source_files = sorted({
        str(record.get("_source_file"))
        for record in records
        if record.get("_source_file")
    })
    return {
        "version": METRICS_VERSION,
        "source_files": source_files,
        "config": {
            "prediction_expiry_minutes": prediction_expiry_minutes,
            "late_hit_tolerance_minutes": late_hit_tolerance_minutes,
            "corridor_length_km": corridor_length_km,
        },
        "corridor": {
            "prediction_count": prediction_count,
            "congested_segment_count": congested_count,
            "matched_prediction_count": matched_prediction_count,
            "matched_congestion_count": matched_congestion_count,
            "missed_congestion_count": missed_count,
            "false_positive_count": false_positive_count,
            "expired_prediction_count": len(expired_prediction_ids),
            "precision": _ratio(matched_prediction_count, prediction_count),
            "recall": _ratio(matched_congestion_count, congested_count),
            "false_positive_rate": _ratio(false_positive_count, prediction_count),
            "mean_eta_error_minutes": _rounded_mean(eta_errors),
            "mean_abs_eta_error_minutes": _rounded_abs_mean(eta_errors),
            "mean_corrected_eta_error_minutes": _rounded_mean(corrected_eta_errors),
            "mean_abs_corrected_eta_error_minutes": (
                _rounded_abs_mean(corrected_eta_errors)
            ),
            "mean_abs_eta_error_delta_minutes": _eta_error_delta(
                eta_errors,
                corrected_eta_errors,
            ),
            "mean_distance_error_km": _rounded_mean(distance_errors),
            "mean_lead_time_minutes": _rounded_mean(lead_times),
            "mean_vms_lead_time_minutes": _rounded_mean(vms_lead_times),
        },
        "segments": segment_stats,
        "matches": matches,
        "misses": misses,
        "false_positives": [
            {
                "prediction_id": prediction.prediction_id,
                "timestamp": prediction.timestamp.isoformat(),
                "camera_id": prediction.camera_id,
                "status": (
                    "expired_prediction"
                    if prediction.prediction_id in expired_prediction_ids
                    else "unmatched_prediction"
                ),
            }
            for prediction in false_positive_predictions
        ],
    }


def evaluate_replay_paths(
    paths: list[Path],
    output_path: Path | None = None,
    route_ids: list[str] | None = None,
    corridor_length_km: float = E4_NORTHBOUND_CORRIDOR_LENGTH_KM,
) -> dict[str, Any]:
    """Load replay records from paths, evaluate them, and optionally write JSON."""
    records = load_jsonl_records(paths)
    metrics = evaluate_replay_records(
        records,
        route_ids=route_ids,
        corridor_length_km=corridor_length_km,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay persisted JSONL ticks and report prediction metrics."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="JSONL files or directories containing sensor_data.jsonl files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional path for the metrics JSON artifact.",
    )
    parser.add_argument(
        "--corridor-length-km",
        type=float,
        default=E4_NORTHBOUND_CORRIDOR_LENGTH_KM,
    )
    args = parser.parse_args(argv)

    metrics = evaluate_replay_paths(
        args.paths,
        output_path=args.output,
        corridor_length_km=args.corridor_length_km,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _expand_input_path(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("**/sensor_data.jsonl"))
    return [path]


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _parse_predictions(records: list[dict[str, Any]]) -> list[ReplayPrediction]:
    predictions: list[ReplayPrediction] = []
    for idx, record in enumerate(records):
        if record.get("type") != "queue_prediction":
            continue
        timestamp = _parse_timestamp(record.get("timestamp"))
        if timestamp is None:
            continue
        lengths = _parse_lengths(record.get("lengths_at_minutes"))
        try:
            origin_chainage = float(record.get("origin_chainage_km", 0.0))
            growth_speed = float(record.get("growth_speed_kmh", 0.0))
            residual_correction = float(
                record.get("residual_correction_minutes", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            continue
        if origin_chainage <= 0 or growth_speed <= 0:
            continue
        predictions.append(ReplayPrediction(
            prediction_id=str(record.get("prediction_id") or f"prediction-{idx}"),
            timestamp=timestamp,
            camera_id=str(record.get("camera_id", "")),
            origin_chainage_km=origin_chainage,
            growth_speed_kmh=growth_speed,
            lengths_at_minutes=lengths,
            residual_correction_minutes=residual_correction,
            corrected_eta_minutes_by_target=_parse_target_etas(
                record.get("corrected_eta_minutes_by_target")
            ),
        ))
    return sorted(predictions, key=lambda prediction: prediction.timestamp)


def _parse_lengths(raw: Any) -> dict[float, float]:
    if not isinstance(raw, dict):
        return {}
    lengths: dict[float, float] = {}
    for key, value in raw.items():
        try:
            minute = float(key)
            distance = float(value)
        except (TypeError, ValueError):
            continue
        if minute >= 0 and distance >= 0:
            lengths[minute] = distance
    return lengths


def _parse_target_etas(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    etas: dict[str, float] = {}
    for key, value in raw.items():
        try:
            eta = float(value)
        except (TypeError, ValueError):
            continue
        if eta >= 0:
            etas[str(key)] = eta
    return etas


def _parse_congestions(
    records: list[dict[str, Any]],
    spans: dict[str, RouteSpan],
) -> list[CongestionEvent]:
    events: list[CongestionEvent] = []
    for idx, record in enumerate(records):
        if record.get("type") != "travel_time":
            continue
        route_id = str(record.get("route_id", ""))
        if route_id not in spans:
            continue
        status = str(record.get("traffic_status", "unknown"))
        if status not in CONGESTED_STATUSES:
            continue
        timestamp = _parse_timestamp(record.get("timestamp"))
        if timestamp is None:
            continue
        try:
            delay_seconds = float(record.get("delay_seconds", 0.0) or 0.0)
        except (TypeError, ValueError):
            delay_seconds = 0.0
        events.append(CongestionEvent(
            event_id=str(record.get("event_id") or f"congestion-{idx}"),
            timestamp=timestamp,
            route_id=route_id,
            status=status,
            delay_seconds=delay_seconds,
        ))
    return sorted(events, key=lambda event: event.timestamp)


def _parse_vms_activations(records: list[dict[str, Any]]) -> list[VMSActivation]:
    activations: list[VMSActivation] = []
    for record in records:
        if record.get("type") != "vms_status" or not record.get("is_active"):
            continue
        timestamp = _parse_timestamp(record.get("timestamp"))
        if timestamp is None:
            continue
        activations.append(VMSActivation(
            timestamp=timestamp,
            vms_id=str(record.get("vms_id", "")),
        ))
    return sorted(activations, key=lambda activation: activation.timestamp)


def _build_route_spans(
    records: list[dict[str, Any]],
    route_ids: list[str],
    corridor_length_km: float,
) -> dict[str, RouteSpan]:
    lengths_m = {route_id: 0.0 for route_id in route_ids}
    for record in records:
        if record.get("type") != "travel_time":
            continue
        route_id = str(record.get("route_id", ""))
        if route_id not in lengths_m:
            continue
        try:
            length_m = float(record.get("length_meters", 0.0) or 0.0)
        except (TypeError, ValueError):
            length_m = 0.0
        if length_m > 0 and lengths_m[route_id] <= 0:
            lengths_m[route_id] = length_m

    present_route_ids = [route_id for route_id in route_ids if lengths_m[route_id] > 0]
    if not present_route_ids:
        return {}
    total_length_m = sum(lengths_m[route_id] for route_id in present_route_ids)
    if total_length_m <= 0:
        return {}

    spans: dict[str, RouteSpan] = {}
    start_km = 0.0
    for idx, route_id in enumerate(present_route_ids):
        route_length_km = lengths_m[route_id] / total_length_m * corridor_length_km
        end_km = (
            corridor_length_km
            if idx == len(present_route_ids) - 1
            else start_km + route_length_km
        )
        spans[route_id] = RouteSpan(route_id, start_km, end_km)
        start_km = end_km
    return spans


def _score_prediction_for_congestion(
    prediction: ReplayPrediction,
    congestion: CongestionEvent,
    span: RouteSpan,
    expiry_minutes: float,
    late_hit_tolerance_minutes: float,
) -> dict[str, Any] | None:
    lead_time_minutes = (
        congestion.timestamp - prediction.timestamp
    ).total_seconds() / 60.0
    if lead_time_minutes < -late_hit_tolerance_minutes:
        return None
    if lead_time_minutes > expiry_minutes:
        return None

    distance_to_midpoint = prediction.origin_chainage_km - span.midpoint_km
    if distance_to_midpoint < 0:
        return None

    predicted_eta_minutes = (
        distance_to_midpoint / prediction.growth_speed_kmh * 60.0
        if prediction.growth_speed_kmh > 0
        else 0.0
    )
    corrected_eta_minutes = prediction.corrected_eta_minutes_by_target.get(
        f"route:{congestion.route_id}",
        max(predicted_eta_minutes + prediction.residual_correction_minutes, 0.0),
    )
    queue_length_at_event = _queue_length_at_minutes(
        prediction,
        max(lead_time_minutes, 0.0),
    )
    predicted_tail_km = prediction.origin_chainage_km - queue_length_at_event
    distance_error_km = predicted_tail_km - span.midpoint_km

    coverage_start = min(predicted_tail_km, prediction.origin_chainage_km)
    coverage_end = max(predicted_tail_km, prediction.origin_chainage_km)
    overlaps_span = coverage_start <= span.end_km and coverage_end >= span.start_km
    immediate_late_hit = (
        lead_time_minutes < 0
        and abs(lead_time_minutes) <= late_hit_tolerance_minutes
        and abs(distance_to_midpoint) <= 0.25
    )
    if not overlaps_span and not immediate_late_hit:
        return None

    eta_error = lead_time_minutes - predicted_eta_minutes
    corrected_eta_error = lead_time_minutes - corrected_eta_minutes
    return {
        "event_id": congestion.event_id,
        "prediction_id": prediction.prediction_id,
        "route_id": congestion.route_id,
        "camera_id": prediction.camera_id,
        "congestion_time": congestion.timestamp.isoformat(),
        "prediction_time": prediction.timestamp.isoformat(),
        "status": "late_hit" if lead_time_minutes < 0 else "hit",
        "lead_time_minutes": round(lead_time_minutes, 3),
        "predicted_eta_minutes": round(predicted_eta_minutes, 3),
        "corrected_eta_minutes": round(corrected_eta_minutes, 3),
        "eta_error_minutes": round(eta_error, 3),
        "corrected_eta_error_minutes": round(corrected_eta_error, 3),
        "distance_error_km": round(distance_error_km, 3),
        "predicted_tail_chainage_km": round(predicted_tail_km, 3),
        "segment_midpoint_km": round(span.midpoint_km, 3),
    }


def _queue_length_at_minutes(
    prediction: ReplayPrediction,
    elapsed_minutes: float,
) -> float:
    if elapsed_minutes <= 0:
        return 0.0
    if not prediction.lengths_at_minutes:
        return prediction.growth_speed_kmh * (elapsed_minutes / 60.0)

    points = sorted(prediction.lengths_at_minutes.items())
    if elapsed_minutes <= points[0][0]:
        first_minute, first_distance = points[0]
        if first_minute <= 0:
            return first_distance
        return first_distance * (elapsed_minutes / first_minute)

    for (left_minute, left_distance), (right_minute, right_distance) in zip(
        points,
        points[1:],
    ):
        if left_minute <= elapsed_minutes <= right_minute:
            span = right_minute - left_minute
            if span <= 0:
                return right_distance
            fraction = (elapsed_minutes - left_minute) / span
            return left_distance + (right_distance - left_distance) * fraction

    last_minute, last_distance = points[-1]
    extra_minutes = elapsed_minutes - last_minute
    return last_distance + prediction.growth_speed_kmh * (extra_minutes / 60.0)


def _find_next_activation(
    activations: list[VMSActivation],
    prediction_time: datetime,
    expiry_minutes: float,
) -> VMSActivation | None:
    latest = prediction_time + timedelta(minutes=expiry_minutes)
    for activation in activations:
        if prediction_time <= activation.timestamp <= latest:
            return activation
    return None


def _latest_timestamp(records: list[dict[str, Any]]) -> datetime | None:
    timestamps = [
        timestamp for timestamp in (
            _parse_timestamp(record.get("timestamp")) for record in records
        )
        if timestamp is not None
    ]
    return max(timestamps) if timestamps else None


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _rounded_mean(values: list[float]) -> float | None:
    return round(mean(values), 3) if values else None


def _rounded_abs_mean(values: list[float]) -> float | None:
    return round(mean(abs(value) for value in values), 3) if values else None


def _eta_error_delta(
    base_errors: list[float],
    corrected_errors: list[float],
) -> float | None:
    base_abs = _rounded_abs_mean(base_errors)
    corrected_abs = _rounded_abs_mean(corrected_errors)
    if base_abs is None or corrected_abs is None:
        return None
    return round(corrected_abs - base_abs, 3)


if __name__ == "__main__":
    raise SystemExit(main())
