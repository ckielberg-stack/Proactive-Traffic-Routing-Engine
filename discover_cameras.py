#!/usr/bin/env python3
"""
Hitta trafikkameror längs E4 Södertälje → Stockholm.

Kör detta en gång för att se vilka kameror som finns i området.
Kopiera sedan önskade kamera-IDs till config.py → CAMERA_IDS.

Användning:
    python discover_cameras.py
"""
import json
import sys

import requests

from config import API_KEY, API_URL, BBOX


def discover_cameras() -> list[dict]:
    """Hämtar alla aktiva kameror och filtrerar på bounding box."""
    if not API_KEY:
        print("❌ Saknar API-nyckel. Sätt TRAFIKVERKET_API_KEY i .env")
        sys.exit(1)

    xml_query = f"""
    <REQUEST>
        <LOGIN authenticationkey="{API_KEY}" />
        <QUERY objecttype="Camera" schemaversion="1">
            <FILTER>
                <AND>
                    <EQ name="Active" value="true" />
                    <EQ name="HasFullSizePhoto" value="true" />
                </AND>
            </FILTER>
        </QUERY>
    </REQUEST>
    """

    print(f"🔍 Söker kameror i bounding box: {BBOX}")
    print(f"   API: {API_URL}\n")

    response = requests.post(
        API_URL,
        data=xml_query.encode("utf-8"),
        headers={"Content-Type": "text/xml; charset=utf-8"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    results = data.get("RESPONSE", {}).get("RESULT", [])
    if not results:
        print("⚠️  Inget resultat – kolla API-nyckel.")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return []

    cameras = results[0].get("Camera", [])

    # Filtrera klientsidan med bounding box
    in_area = []
    for cam in cameras:
        geom = cam.get("Geometry", {}).get("WGS84", "")
        if not geom:
            continue
        try:
            parts = geom.replace("POINT (", "").replace(")", "").strip().split()
            lng, lat = float(parts[0]), float(parts[1])
            if (
                BBOX["min_lat"] <= lat <= BBOX["max_lat"]
                and BBOX["min_lng"] <= lng <= BBOX["max_lng"]
            ):
                in_area.append(cam)
        except (ValueError, IndexError):
            continue

    print(f"   Totalt {len(cameras)} kameror hämtade, {len(in_area)} inom bounding box\n")
    return in_area


def print_cameras(cameras: list[dict]) -> None:
    """Skriver ut kameror i ett lättläst format."""
    if not cameras:
        print("Inga kameror hittades i området.")
        return

    print(f"📷 Hittade {len(cameras)} kamera(or):\n")
    print(f"{'#':<4} {'ID':<35} {'Namn':<30} {'Riktning':<10} {'Typ'}")
    print("-" * 100)

    for i, cam in enumerate(cameras, 1):
        cam_id = cam.get("Id", "?")
        name = cam.get("Name", cam.get("Description", "?"))
        direction = str(cam.get("Direction", "?"))
        cam_type = cam.get("Type", "?")
        photo_url = cam.get("PhotoUrl", "")

        print(f"{i:<4} {cam_id:<35} {name:<30} {direction:<10} {cam_type}")

        geom = cam.get("Geometry", {}).get("WGS84", "")
        if geom:
            print(f"     📍 {geom}")

        if photo_url:
            print(f"     🖼  {photo_url}")
        print()

    print("=" * 100)
    print("\n💡 Kopiera önskade IDs till config.py → CAMERA_IDS")
    print('   Exempel: CAMERA_IDS = ["SE_STA_CAMERA_0_xxx", "SE_STA_CAMERA_0_yyy"]')


def main():
    cameras = discover_cameras()
    print_cameras(cameras)

    if cameras:
        with open("discovered_cameras.json", "w", encoding="utf-8") as f:
            json.dump(cameras, f, indent=2, ensure_ascii=False)
        print(f"\n📁 Rå-data sparad i discovered_cameras.json")


if __name__ == "__main__":
    main()
