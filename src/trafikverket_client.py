
"""Trafikverket HTTP and image client helpers for the tick pipeline."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
import requests

from config import API_URL, MAX_RETRIES, RETRY_BACKOFF

logger = logging.getLogger("mainloop")


def api_request(xml_query: str, retries: int = MAX_RETRIES) -> dict | None:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                API_URL,
                data=xml_query.encode("utf-8"),
                headers={"Content-Type": "text/xml; charset=utf-8"},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            wait = RETRY_BACKOFF ** attempt
            logger.warning(f"API error (attempt {attempt}/{retries}): {e} — waiting {wait}s")
            if attempt < retries:
                time.sleep(wait)
    logger.error(f"API call failed after {retries} attempts")
    return None


def _val(d: dict, key: str):
    """Get nested .Value from a Trafikverket observation field."""
    sub = d.get(key, {})
    return sub.get("Value") if isinstance(sub, dict) else None


def _nested(d: dict, *keys):
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
        else:
            return None
    return d


def fetch_image_bytes(url: str, retries: int = MAX_RETRIES) -> bytes | None:
    """Fetch image from URL into memory. No disk I/O."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            wait = RETRY_BACKOFF ** attempt
            logger.warning(f"Image fetch error (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(wait)
    return None


def decode_frame(raw_bytes: bytes) -> np.ndarray | None:
    """Decode JPEG bytes to a BGR numpy array in memory."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
