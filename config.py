"""
Konfiguration för Trafikverkets kamera- och sensordata-insamling.

Sträcka: E4/E20 Hallunda → Stockholm (Karlbergskanalen)
46 trafikflödeskameror längs sträckan.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API ---
API_KEY = os.getenv("TRAFIKVERKET_API_KEY")
API_URL = "https://api.trafikinfo.trafikverket.se/v2/data.json"

# --- Bounding box: Hallunda → Stockholm (E4/E20) ---
BBOX = {
    "min_lat": 59.24,   # Hallunda
    "max_lat": 59.35,   # Tomteboda / Karlberg
    "min_lng": 17.83,
    "max_lng": 18.04,
}

# --- Kamera-IDs att övervaka ---
# Alla 53 E4/E20 trafikflödeskameror Hallunda → Stockholm
# Sorterade syd → nord efter latitud
CAMERA_IDS: list[str] = [
    # — Hallunda / Fittja / Vårby —
    "SE_STA_CAMERA_Orion_466",       # Tpl Hallunda
    "SE_STA_CAMERA_Orion_417",       # Hallunda norra
    "SE_STA_CAMERA_Pacific_530",     # Brunna
    "SE_STA_CAMERA_Orion_437",       # Slagsta
    "SE_STA_CAMERA_Pacific_529",     # Trafikplats Fittja
    "SE_STA_CAMERA_Orion_436",       # Fittja
    "SE_STA_CAMERA_Pacific_528",     # Trafikplats Vårby
    # — Kungens kurva / Bredäng —
    "SE_STA_CAMERA_Orion_412",       # Tpl Kungens kurva
    "SE_STA_CAMERA_0_50438294",      # Tpl Bredäng norra
    # — Fruängen / Västertorp / Solberga —
    "SE_STA_CAMERA_0_50438290",      # Fruängen
    "SE_STA_CAMERA_0_50438292",      # Fruängen södra
    "SE_STA_CAMERA_0_50438288",      # Fruängen norra
    "SE_STA_CAMERA_0_50438286",      # Tpl Västertorp södra
    "SE_STA_CAMERA_0_50438284",      # Tpl Västertorp
    "SE_STA_CAMERA_0_50438282",      # Tpl Västertorp norra
    "SE_STA_CAMERA_0_50438280",      # Solberga södra
    "SE_STA_CAMERA_0_50438278",      # Solberga
    "SE_STA_CAMERA_0_50438276",      # Solberga norra
    # — Västberga —
    "SE_STA_CAMERA_0_50438758",      # Tpl Västberga Södra
    "SE_STA_CAMERA_0_50438756",      # Tpl Västberga
    "SE_STA_CAMERA_0_50438754",      # Tpl Västberga Norra
    "SE_STA_CAMERA_0_50438752",      # Västberga Allé Södra
    "SE_STA_CAMERA_0_50438750",      # Västberga Allé
    # — Nyboda —
    "SE_STA_CAMERA_0_50438748",      # Tpl Nyboda Södra
    "SE_STA_CAMERA_0_50438746",      # Tpl Nyboda
    "SE_STA_CAMERA_0_50438740",      # Tpl Nyboda Östra
    "SE_STA_CAMERA_0_50438738",      # Nybodahöjden
    # — Midsommarkransen / Nybohov —
    "SE_STA_CAMERA_0_50438736",      # Midsommarkransens Gymnasium
    "SE_STA_CAMERA_0_50438734",      # Tpl Nybohov Södra
    "SE_STA_CAMERA_0_50438732",      # Tpl Nybohov
    "SE_STA_CAMERA_0_50438730",      # Tpl Nybohov Norra
    # — Gröndal —
    "SE_STA_CAMERA_0_50438728",      # Kontrollplats Gröndal
    "SE_STA_CAMERA_0_50438726",      # Tpl Gröndal
    "SE_STA_CAMERA_0_50438724",      # Tpl Gröndal Norra
    # — Essingen —
    "SE_STA_CAMERA_0_50438722",      # Tpl Stora Essingen Södra
    "SE_STA_CAMERA_0_50438720",      # Tpl Stora Essingen
    "SE_STA_CAMERA_0_50438718",      # Tpl Stora Essingen Norra
    "SE_STA_CAMERA_0_50438716",      # Tpl Lilla Essingen Södra
    "SE_STA_CAMERA_0_50438714",      # Tpl Lilla Essingen
    # — Fredhäll —
    "SE_STA_CAMERA_0_50438708",      # Tpl Fredhäll Södra
    "SE_STA_CAMERA_0_50438704",      # Tpl Fredhäll
    "SE_STA_CAMERA_0_50438702",      # Tpl Fredhäll Norra
    # — Kristineberg —
    "SE_STA_CAMERA_0_50438700",      # Tpl Kristineberg
    "SE_STA_CAMERA_0_50438696",      # Tpl Kristineberg Norra
    # — Hornsberg / Karlbergskanalen —
    "SE_STA_CAMERA_0_50438694",      # Hornsberg
    "SE_STA_CAMERA_0_50438692",      # Karlbergskanalen
]

# Camera coordinates for dashboard map (lat, lng) — south to north
CAMERA_COORDS: dict[str, tuple[float, float]] = {
    "SE_STA_CAMERA_Orion_466":      (59.2417, 17.8366),
    "SE_STA_CAMERA_Orion_417":      (59.2431, 17.8378),
    "SE_STA_CAMERA_Pacific_530":    (59.2476, 17.8435),
    "SE_STA_CAMERA_Orion_437":      (59.2506, 17.8516),
    "SE_STA_CAMERA_Pacific_529":    (59.2525, 17.8565),
    "SE_STA_CAMERA_Orion_436":      (59.2543, 17.8619),
    "SE_STA_CAMERA_Pacific_528":    (59.2544, 17.8748),
    "SE_STA_CAMERA_Orion_412":      (59.2725, 17.9142),
    "SE_STA_CAMERA_0_50438294":     (59.2890, 17.9542),
    "SE_STA_CAMERA_0_50438290":     (59.2891, 17.9671),
    "SE_STA_CAMERA_0_50438292":     (59.2894, 17.9594),
    "SE_STA_CAMERA_0_50438288":     (59.2893, 17.9707),
    "SE_STA_CAMERA_0_50438286":     (59.2892, 17.9761),
    "SE_STA_CAMERA_0_50438284":     (59.2893, 17.9815),
    "SE_STA_CAMERA_0_50438282":     (59.2891, 17.9853),
    "SE_STA_CAMERA_0_50438280":     (59.2894, 17.9883),
    "SE_STA_CAMERA_0_50438278":     (59.2898, 17.9908),
    "SE_STA_CAMERA_0_50438276":     (59.2913, 17.9958),
    "SE_STA_CAMERA_0_50438758":     (59.2936, 18.0007),
    "SE_STA_CAMERA_0_50438756":     (59.2960, 18.0041),
    "SE_STA_CAMERA_0_50438754":     (59.2972, 18.0057),
    "SE_STA_CAMERA_0_50438752":     (59.2988, 18.0100),
    "SE_STA_CAMERA_0_50438750":     (59.2994, 18.0130),
    "SE_STA_CAMERA_0_50438748":     (59.3000, 18.0170),
    "SE_STA_CAMERA_0_50438746":     (59.3012, 18.0203),
    "SE_STA_CAMERA_0_50438740":     (59.3009, 18.0238),
    "SE_STA_CAMERA_0_50438738":     (59.3028, 18.0209),
    "SE_STA_CAMERA_0_50438736":     (59.3043, 18.0192),
    "SE_STA_CAMERA_0_50438734":     (59.3061, 18.0146),
    "SE_STA_CAMERA_0_50438732":     (59.3079, 18.0107),
    "SE_STA_CAMERA_0_50438730":     (59.3095, 18.0084),
    "SE_STA_CAMERA_0_50438728":     (59.3124, 18.0056),
    "SE_STA_CAMERA_0_50438726":     (59.3150, 18.0033),
    "SE_STA_CAMERA_0_50438724":     (59.3172, 18.0008),
    "SE_STA_CAMERA_0_50438722":     (59.3190, 17.9983),
    "SE_STA_CAMERA_0_50438720":     (59.3212, 17.9969),
    "SE_STA_CAMERA_0_50438718":     (59.3228, 17.9982),
    "SE_STA_CAMERA_0_50438716":     (59.3245, 18.0013),
    "SE_STA_CAMERA_0_50438714":     (59.3255, 18.0040),
    "SE_STA_CAMERA_0_50438708":     (59.3299, 18.0100),
    "SE_STA_CAMERA_0_50438704":     (59.3312, 18.0103),
    "SE_STA_CAMERA_0_50438702":     (59.3326, 18.0103),
    "SE_STA_CAMERA_0_50438700":     (59.3341, 18.0100),
    "SE_STA_CAMERA_0_50438696":     (59.3368, 18.0111),
    "SE_STA_CAMERA_0_50438694":     (59.3389, 18.0114),
    "SE_STA_CAMERA_0_50438692":     (59.3410, 18.0112),
}

# Chainage datum used by physics and VMS: km from Hallunda heading northbound.
E4_NORTHBOUND_CORRIDOR_LENGTH_KM: float = 15.8

# Offline route reference, preserving the curated E4 northbound monitoring order.
E4_NORTHBOUND_ROUTE_POINTS: list[tuple[float, float]] = [
    CAMERA_COORDS[camera_id]
    for camera_id in CAMERA_IDS
    if camera_id in CAMERA_COORDS
]

# --- Insamling ---
INTERVAL_SECONDS = 60  # Hur ofta bilder och sensordata hämtas
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # Sekunder, multipliceras exponentiellt

# --- TrafficFlow Sensor SiteIds ---
# Curated northbound E4 measurement stations along the Hallunda → Kristineberg
# corridor.  Each SiteId has 1-4 lane entries in the API.
# Discovered via bbox query (lat 59.24–59.35, lng 17.83–18.04) and filtered
# to northBound / northWestBound / northEastBound stations.
# Sorted south → north by latitude.
SENSOR_SITE_IDS: list[int] = [
    # — Hallunda / Kungens Kurva (lat ~59.25) —
    1274,   # lat 59.2549, northbound approach
    1286,   # lat 59.2549, northbound
    # — Bredäng / Mälarhöjden (lat ~59.29) —
    2851,   # lat 59.2937, northBound
    2842,   # lat 59.2961, northBound
    2603,   # lat 59.2972, northBound
    2631,   # lat 59.2980, northBound
    2634,   # lat 59.2981, northBound
    2629,   # lat 59.2989, northBound
    2612,   # lat 59.2994, northBound
    2610,   # lat 59.2995, northBound
    2625,   # lat 59.2997, northBound
    2626,   # lat 59.2998, northBound
    # — Liljeholmen / Gullmarsplan (lat ~59.30) —
    2651,   # lat 59.3007, northBound
    2653,   # lat 59.3010, northBound
    2654,   # lat 59.3011, northBound
    2645,   # lat 59.3021, northBound
    2648,   # lat 59.3026, northBound
    2619,   # lat 59.3027, northBound
    2659,   # lat 59.3042, northBound
    2663,   # lat 59.3061, northWestBound
    # — Årsta / Johanneshov (lat ~59.31) —
    2682,   # lat 59.3150, northWestBound
    2694,   # lat 59.3212, northBound
    2706,   # lat 59.3256, northEastBound
    # — Essingeleden / Kristineberg (lat ~59.33) —
    2768,   # lat 59.3272, northBound
    2767,   # lat 59.3272, northBound
    2766,   # lat 59.3272, northBound
    2790,   # lat 59.3321, northBound
    2786,   # lat 59.3326, northBound
    2788,   # lat 59.3341, northBound
    2817,   # lat 59.3389, northBound
]

# Sensor station coordinates (lat, lng) — for nearest-camera matching
SENSOR_COORDS: dict[int, tuple[float, float]] = {
    1274: (59.2549, 17.860),
    1286: (59.2549, 17.860),
    2851: (59.2937, 17.996),
    2842: (59.2961, 18.000),
    2603: (59.2972, 18.004),
    2631: (59.2980, 18.007),
    2634: (59.2981, 18.007),
    2629: (59.2989, 18.010),
    2612: (59.2994, 18.012),
    2610: (59.2995, 18.012),
    2625: (59.2997, 18.013),
    2626: (59.2998, 18.013),
    2651: (59.3007, 18.017),
    2653: (59.3010, 18.018),
    2654: (59.3011, 18.018),
    2645: (59.3021, 18.020),
    2648: (59.3026, 18.021),
    2619: (59.3027, 18.021),
    2659: (59.3042, 18.019),
    2663: (59.3061, 18.015),
    2682: (59.3150, 18.003),
    2694: (59.3212, 17.997),
    2706: (59.3256, 18.004),
    2768: (59.3272, 18.010),
    2767: (59.3272, 18.010),
    2766: (59.3272, 18.010),
    2790: (59.3321, 18.010),
    2786: (59.3326, 18.010),
    2788: (59.3341, 18.010),
    2817: (59.3389, 18.011),
}

# --- Road Speed Limits per Sensor Station ---
# Posted speed limit (km/h) for each sensor location.
# Stations not listed here use DEFAULT_ROAD_SPEED_LIMIT.
# The E4 corridor through Stockholm is generally 70 km/h.
SENSOR_ROAD_SPEED_LIMITS: dict[int, int] = {
    # Hallunda / Kungens Kurva — 70 km/h
    1274: 70, 1286: 70,
    # Bredäng / Mälarhöjden — 70 km/h
    2851: 70, 2842: 70, 2603: 70, 2631: 70, 2634: 70,
    2629: 70, 2612: 70, 2610: 70, 2625: 70, 2626: 70,
    # Liljeholmen / Gullmarsplan — 70 km/h
    2651: 70, 2653: 70, 2654: 70, 2645: 70, 2648: 70,
    2619: 70, 2659: 70, 2663: 70,
    # Årsta / Johanneshov — 70 km/h
    2682: 70, 2694: 70, 2706: 70,
    # Essingeleden / Kristineberg — 70 km/h
    2768: 70, 2767: 70, 2766: 70, 2790: 70, 2786: 70,
    2788: 70, 2817: 70,
}
DEFAULT_ROAD_SPEED_LIMIT: int = 70

# --- Sensor Anomaly Detection Thresholds ---
# Flag as "warning" when speed drops below this fraction of the speed limit
SENSOR_SPEED_DROP_RATIO: float = 0.50     # 50% → e.g. 35 km/h on a 70 road
# Flag as "severe" (triggers VMS recommendation) below this fraction
SENSOR_SEVERE_DROP_RATIO: float = 0.35    # 35% → e.g. 24.5 km/h on a 70 road
# --- TravelTimeRoute IDs (E4/E20 corridor, Stockholm) ---
# Discovered from TravelTimeRoute API (schemaversion 1.5), CountyNo=1.
# Route IDs are strings matching the Trafikverket 'Id' field.
E4_NORTHBOUND_TRAVEL_TIME_ROUTE_IDS: list[str] = [
    "724",    # E4/E20 N Hallunda S (146b) - Hallunda N (146a)
    "725",    # E4/E20 N Hallunda N (146a) - Fittja (147)
    "726",    # E4/E20 N Fittja (147) - Vårby (148)
    "634",    # E4/E20 N Bredäng (152) - Västertorp (153)
    "635",    # E4/E20 N Västertorp (153) - Västberga (154)
    "637",    # E4/E20 N Nyboda (Södertäljevägen till Essingeleden)
    "640",    # E4/E20 N Nyboda (155) - Nybohov (156)
    "641",    # E4/E20 N Nybohov (156) - Gröndal (157)
    "642",    # E4/E20 N Gröndal (157) - Lilla Essingen
    "643",    # E4/E20 N Lilla Essingen (159) - Fredhäll (160)
    "10522",  # E4/E20 N Trafikplats Karlberg Norra
    "10523",  # E4/E20 N Karlberg (163) – Norrtull (164)
    "10524",  # E4/E20 N Trafikplats Karlberg Södra
    "10525",  # E4 N Norrtull (164) - Haga Södra (165)
]

# Combined fetch list covers Hallunda → Karlberg in both directions.
E4_TRAVEL_TIME_ROUTE_IDS: list[str] = [
    *E4_NORTHBOUND_TRAVEL_TIME_ROUTE_IDS,
    # Southbound
    "709",    # E4/E20 S Hallunda N (146a) – Hallunda S (146b)
    "625",    # E4/E20 S Nybohov (156) - Nyboda (155)
    "626",    # E4/E20 S Nyboda (Essingeleden till Södertäljevägen)
    "629",    # E4/E20 S Nyboda (155) - Västberga (154)
    "624",    # E4/E20 S Gröndal (157) - Nybohov (156)
    "631",    # E4/E20 S Västertorp (153) - Bredäng (152)
    "778",    # E4 S Eugeniatunneln – Karlberg (163)
]
