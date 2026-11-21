"""Shape tests for the BBS AQA 3D CNN.

Run with:  pytest -q
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.bbsaqa_3dcnn import BBSAQA3DCNN  # noqa: E402


def _model():
    return BBSAQA3DCNN(num_actions=7, num_scores=5).eval()


def test_forward_output_shapes():
    """Two clips in, one action logit vector and one score grid per clip."""
    model = _model()
    batch = torch.randn(2, 3, 16, 224, 224)
    with torch.no_grad():
        out = model(batch)

    assert out["action_logits"].shape == (2, 7)
    assert out["action_scores"].shape == (2, 7, 5)


def test_heads_are_independent_of_batch_size():
    """Batch dimension is preserved for both heads."""
    model = _model()
    for batch_size in (1, 3):
        batch = torch.randn(batch_size, 3, 16, 224, 224)
        with torch.no_grad():
            out = model(batch)
        assert out["action_logits"].shape[0] == batch_size
        assert out["action_scores"].shape[:2] == (batch_size, 7)


def test_score_range_matches_bbs_scale():
    """The score head emits one logit per BBS score level (0-4)."""
    model = _model()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 16, 224, 224))
    assert out["action_scores"].shape[-1] == 5
