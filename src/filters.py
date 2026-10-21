"""Stage 3b -- conventional metadata filters.   OWNER: Person B

PS SIH26227 section 2.2.1 requires that semantic ranking still compose with
ordinary spatial / temporal / sensor filters. These return a `subset` of row
indices which is passed straight into retrieval -- filtering narrows the
candidate set, semantic similarity orders it.
"""

from datetime import date, datetime


def _as_date(value):
    if value is None or isinstance(value, date):
        return value
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(str(value)[:10], fmt).date()
        except ValueError:
            continue
    return None


def by_date_range(index, start=None, end=None):
    start, end = _as_date(start), _as_date(end)
    keep = []
    for i, m in enumerate(index.metadata):
        d = _as_date(m.get("date"))
        if d is None:
            continue
        if start and d < start:
            continue
        if end and d > end:
            continue
        keep.append(i)
    return keep


def by_sensor(index, sensor):
    wanted = {s.lower() for s in ([sensor] if isinstance(sensor, str) else sensor)}
    return [i for i, m in enumerate(index.metadata)
            if str(m.get("sensor", "")).lower() in wanted]


def by_source(index, substring):
    s = substring.lower()
    return [i for i, m in enumerate(index.metadata)
            if s in str(m.get("source", "")).lower()]


def by_aoi(index, min_lon, min_lat, max_lon, max_lat):
    """Area of interest. Keeps tiles whose geographic bounds intersect the box.

    Tiles with no bounds (non-georeferenced benchmark data) are skipped rather
    than silently included -- an AOI filter that quietly returns unlocated
    tiles would be worse than one that returns nothing.
    """
    keep = []
    for i, m in enumerate(index.metadata):
        b = m.get("bounds")
        if not b:
            continue
        left, bottom, right, top = b
        if right < min_lon or left > max_lon or top < min_lat or bottom > max_lat:
            continue
        keep.append(i)
    return keep


def combine(*subsets):
    """Intersection of filters. Ignores None (an unset filter constrains nothing)."""
    active = [set(s) for s in subsets if s is not None]
    if not active:
        return None
    out = set.intersection(*active)
    return sorted(out)
