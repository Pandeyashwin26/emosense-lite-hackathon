import pytest
from utils.emotion_utils import fuse_modalities


def test_fuse_all_neutral():
    f = [0, 0, 0, 1, 0]
    a = [0, 0, 0, 1, 0]
    t = [0, 0, 0, 1, 0]
    res = fuse_modalities(f, a, t, weights=(0.4, 0.3, 0.3))
    assert res["label"] == "neutral"


def test_fuse_face_happy():
    f = [1, 0, 0, 0, 0]
    a = [0, 0, 0, 1, 0]
    t = [0, 0, 0, 1, 0]
    res = fuse_modalities(f, a, t, weights=(0.6, 0.2, 0.2))
    assert res["label"] == "happy"


def test_fuse_equal_weights():
    f = [1,0,0,0,0]
    a = [0,1,0,0,0]
    t = [0,0,1,0,0]
    fused = fuse_modalities(f, a, t, weights=(1/3,1/3,1/3))
    assert 'probs' in fused
    assert len(fused['probs']) == 5
    # Ensure fused is normalized
    assert abs(sum(fused['probs']) - 1.0) < 1e-6


def test_fuse_missing_modalities():
    neutral = [0,0,0,1,0]
    fused = fuse_modalities(neutral, neutral, neutral)
    assert fused['label'] == 'neutral' or fused['probs'][3] > 0.9
