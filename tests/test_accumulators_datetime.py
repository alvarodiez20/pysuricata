from datetime import datetime, timezone

from pysuricata.accumulators.datetime import DatetimeAccumulator


def _to_ns(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1_000_000_000)


def test_datetime_accumulator_min_max_and_buckets():
    acc = DatetimeAccumulator("ts")
    dts = [
        _to_ns(datetime(2024, 1, 1, 0, 0, 0)),
        _to_ns(datetime(2024, 1, 1, 12, 0, 0)),
        _to_ns(datetime(2024, 1, 2, 13, 0, 0)),
        None,
    ]
    acc.update(dts)
    s = acc.finalize()
    assert s.name == "ts"
    assert s.count == 3
    assert s.missing == 1
    assert s.min_ts is not None and s.max_ts is not None
    assert s.min_ts <= s.max_ts
    assert len(s.by_hour) == 24
    assert len(s.by_dow) == 7
    assert len(s.by_month) == 12

