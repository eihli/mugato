import pytest
import torch

from mugato.utils import normalize_to_between_minus_one_plus_one


def test_normalize_to_between_minus_one_plus_one():
    xs = torch.tensor([-2, -1, 0, 1, 2, 4])
    norm_xs = normalize_to_between_minus_one_plus_one(xs)
    # The relative differences between values should remain the same.
    orig_ratio = (xs[-3] - xs[-2]) / (xs[-2] - xs[-1])
    norm_ratio =  (norm_xs[-3] - norm_xs[-2]) / (norm_xs[-2] - norm_xs[-1])
    assert orig_ratio == pytest.approx(norm_ratio), (
        f"Expected ratio {orig_ratio} to be close to {norm_ratio}"
    )
    # Verify values are normalized between -1 and +1
    assert torch.all(norm_xs >= -1) and torch.all(norm_xs <= 1)
