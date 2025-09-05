from pysuricata.accumulators.boolean import BooleanAccumulator


def test_boolean_accumulator_counts():
    acc = BooleanAccumulator("flag")
    acc.update([True, False, None, True, False, False])
    s = acc.finalize()
    assert s.name == "flag"
    assert s.count == s.true_n + s.false_n
    assert s.missing >= 1
    # dtype string present
    assert s.dtype_str == "boolean"

