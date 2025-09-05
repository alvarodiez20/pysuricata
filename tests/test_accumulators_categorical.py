from pysuricata.accumulators.categorical import CategoricalAccumulator


def test_categorical_accumulator_top_items_and_missing():
    acc = CategoricalAccumulator("c")
    data = ["a", "b", "a", None, "c", "a", "b", " ", ""]
    acc.update(data)
    s = acc.finalize()
    assert s.name == "c"
    assert s.count >= 6  # 3 non-missing a/b/a + c + a + b + " " + ""
    assert s.missing >= 1
    assert s.unique_est >= 1
    assert isinstance(s.top_items, list)
    # empty string counted
    assert s.empty_zero >= 1

