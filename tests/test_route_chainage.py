from src.route_chainage import build_route_chainage_map, find_nearest_by_chainage


def test_projection_follows_route_order_when_latitude_decreases() -> None:
    route_points = [
        (0.0, 0.0),
        (0.0, 0.01),
        (-0.01, 0.02),
    ]
    positions = {
        "A": (0.0, 0.0),
        "B": (0.0, 0.01),
        "C": (-0.01, 0.02),
    }

    chainages = build_route_chainage_map(positions, route_points, 10.0)

    assert chainages["A"] == 0.0
    assert chainages["A"] < chainages["B"] < chainages["C"]


def test_same_latitude_points_are_separated_by_route_distance() -> None:
    route_points = [
        (0.0, 0.0),
        (0.0, 0.01),
    ]
    positions = {
        "A": (0.0, 0.0),
        "B": (0.0, 0.01),
    }

    chainages = build_route_chainage_map(positions, route_points, 4.0)

    assert chainages["A"] == 0.0
    assert chainages["B"] == 4.0


def test_empty_or_degenerate_routes_return_empty_maps() -> None:
    positions = {"A": (0.0, 0.0)}

    assert build_route_chainage_map(positions, [], 10.0) == {}
    assert build_route_chainage_map(positions, [(0.0, 0.0)], 10.0) == {}
    assert build_route_chainage_map(positions, [(0.0, 0.0), (0.0, 0.01)], 0.0) == {}


def test_find_nearest_by_chainage() -> None:
    nearest = find_nearest_by_chainage(
        4.9,
        {
            "A": 0.0,
            "B": 5.0,
            "C": 10.0,
        },
    )

    assert nearest == "B"
