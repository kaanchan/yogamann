import numpy as np
import pytest

from pose_oks import COCO_SIGMAS, oks

N = 17


def _pose():
    rng = np.random.default_rng(0)
    return rng.uniform(100, 400, size=(N, 2))


def test_identical_pose_scores_one():
    gt = _pose()
    assert oks(gt.copy(), gt, scale=200.0) == pytest.approx(1.0)


def test_score_decreases_with_distance():
    gt = _pose()
    near = oks(gt + 2.0, gt, scale=200.0)
    far = oks(gt + 40.0, gt, scale=200.0)
    assert 1.0 > near > far > 0.0


def test_matches_hand_computed_value():
    gt = np.zeros((N, 2))
    pred = gt.copy()
    pred[:, 0] = 10.0  # every keypoint off by 10 px in x
    scale = 100.0
    expected = np.mean(np.exp(-(10.0 ** 2) / (2 * scale ** 2 * (2 * COCO_SIGMAS) ** 2)))
    assert oks(pred, gt, scale=scale) == pytest.approx(expected)


def test_invisible_gt_keypoints_are_ignored():
    gt = _pose()
    pred = gt.copy()
    pred[5] += 300.0  # wildly wrong, but marked not visible in gt
    visible = np.ones(N, dtype=bool)
    visible[5] = False
    assert oks(pred, gt, scale=200.0, visible=visible) == pytest.approx(1.0)


def test_no_visible_keypoints_returns_zero():
    gt = _pose()
    assert oks(gt, gt, scale=200.0, visible=np.zeros(N, dtype=bool)) == 0.0


def test_accepts_wholebody_133_by_truncating_to_body():
    gt = np.vstack([_pose(), np.zeros((133 - N, 2))])
    pred = gt.copy()
    pred[N:] = 999.0  # face/hand/foot points must not affect the body score
    assert oks(pred, gt, scale=200.0) == pytest.approx(1.0)


def test_rejects_bad_scale():
    gt = _pose()
    with pytest.raises(ValueError):
        oks(gt, gt, scale=0.0)


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        oks(np.zeros((17, 2)), np.zeros((16, 2)), scale=100.0)
