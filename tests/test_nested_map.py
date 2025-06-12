from u2fold.utils.nested_map import nested_map


def f(x):
    return 2 * x


def id(x):
    return x


def literal_a(s: str):
    return "a"


def test_nested_lists_with_identity():
    structure = [[1, 2], [3, 4]]

    expected_output = [[1, 2], [3, 4]]

    assert nested_map(id, structure) == expected_output


def test_nested_lists_with_times_two():
    structure = [[1, 2], [3, 4]]

    expected_output = [[2, 4], [6, 8]]

    assert nested_map(f, structure) == expected_output


def test_with_tuples():
    structure = ((1, 2), (3, 4))
    expected_output = ((2, 4), (6, 8))

    assert nested_map(f, structure) == expected_output


def test_with_mixed():
    structure = ([1, 2], (3, 4))
    expected_output = ([2, 4], (6, 8))

    assert nested_map(f, structure) == expected_output


def test_with_mixed_2():
    structure = [[1, 2], (3, 4)]
    expected_output = [[2, 4], (6, 8)]

    assert nested_map(f, structure) == expected_output


def test_with_strings():
    structure = [(["foo", "bar"]), ["baz", "foobar"]]
    expected_output = [(["a", "a"]), ["a", "a"]]

    assert nested_map(literal_a, structure) == expected_output


def test_dicts():
    structure = [{"foo": 1, "bar": 2}, {"foo": 3, "bar": 4}]
    expected_output = [{"foo": 2, "bar": 4}, {"foo": 6, "bar": 8}]

    assert nested_map(f, structure) == expected_output


def test_generator():
    structure = (x for x in range(3))
    expected_output = (f(x) for x in range(3))

    output = nested_map(f, structure)

    assert list(output) == list(expected_output)

    # generator was consumed
    assert len(list(output)) == 0
