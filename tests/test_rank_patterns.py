from low_rank_eval.lora.rank_patterns import contiguous_depth_groups


def test_depth_groups_are_contiguous_and_complete() -> None:
    groups = contiguous_depth_groups(35)
    assert list(groups[0] + groups[1] + groups[2]) == list(range(35))
    assert [len(group) for group in groups] == [12, 12, 11]
