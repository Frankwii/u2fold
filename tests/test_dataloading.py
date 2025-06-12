from u2fold.data.dataset_splits import DatasetSplits, SplitData
import pytest


def test_split_data_to_tuple():
    input = SplitData(
        training="foo",
        validation="bar",
        test="baz"
    )

    output = input.to_tuple()

    expected = ("foo", "bar", "baz")

    assert expected == output

def test_split_data_map_without_params():
    input = SplitData(
        training="foo",
        validation="bar",
        test="baz"
    )

    f = lambda s: s.capitalize()

    output = input.map(f)

    expected = SplitData(
        training="Foo",
        validation="Bar",
        test="Baz"
    )

    assert expected == output

def test_split_data_map_with_params():
    input = SplitData(
        0.1,
        0.2,
        0.3
    )

    multipliers = SplitData(
        10,
        5,
        10/3
    )

    multiply = lambda x,y: x * y


    output = input.map(multiply, params=multipliers)

    expected = SplitData(1, 1, 1)

    assert expected == output


def test_dataset_splits_should_raise_when_invalid_splits():

    invalid_sum_cases = [(0.5, 0.5, 0.1), (0.3, 0.3, 0.3), (0.33, 0.33, 0.33)]

    for case in invalid_sum_cases:
        with pytest.raises(ValueError, match="must sum to 1"):
            DatasetSplits(*case)

    nonpositive_value_cases = [(0, 0.5, 0.5), (-0.1, 0.6, 0.5)]

    for case in nonpositive_value_cases:
        with pytest.raises(ValueError, match="must be positive"):
            DatasetSplits(*case)

