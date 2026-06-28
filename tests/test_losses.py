import torch

from pepcl.pepcl import PepCL, PepCL_Micro


def test_logsmoothap_loss_is_finite_and_backpropagates():
    scores = torch.tensor([3.0, 2.0, -1.0, 0.5, -2.0], requires_grad=True)
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])

    loss_fn = PepCL._LogSmoothAPScore(alpha=10.0)
    loss = loss_fn(scores, labels)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()

    loss.backward()

    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all().item()


def test_logsmoothap_better_ranking_has_lower_loss():
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

    good_scores = torch.tensor([3.0, 2.0, -1.0, -2.0])
    bad_scores = torch.tensor([-2.0, -1.0, 2.0, 3.0])

    loss_fn = PepCL._LogSmoothAPScore(alpha=10.0)

    good_loss = loss_fn(good_scores, labels)
    bad_loss = loss_fn(bad_scores, labels)

    assert torch.isfinite(good_loss).item()
    assert torch.isfinite(bad_loss).item()
    assert good_loss.item() < bad_loss.item()


def test_soft_spearman_loss_is_finite_and_backpropagates():
    pred = torch.tensor([0.1, 0.4, 0.2, 0.8, 0.5], requires_grad=True)
    target = torch.tensor([0.0, 0.6, 0.3, 1.0, 0.7])

    loss_fn = PepCL._SoftSpearmanLoss(tau=0.1)
    loss = loss_fn(pred, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()

    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()


def test_soft_spearman_better_ordering_has_lower_loss():
    target = torch.tensor([0.1, 0.2, 0.7, 0.9])

    good_pred = torch.tensor([0.1, 0.2, 0.7, 0.9])
    bad_pred = torch.tensor([0.9, 0.7, 0.2, 0.1])

    loss_fn = PepCL._SoftSpearmanLoss(tau=0.1)

    good_loss = loss_fn(good_pred, target)
    bad_loss = loss_fn(bad_pred, target)

    assert torch.isfinite(good_loss).item()
    assert torch.isfinite(bad_loss).item()
    assert good_loss.item() < bad_loss.item()


def test_micro_losses_are_available_and_finite():
    scores = torch.tensor([2.0, -1.0, 1.5, -0.5], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss_fn = PepCL_Micro._LogSmoothAPScore(alpha=10.0)
    loss = loss_fn(scores, labels)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()

    loss.backward()

    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all().item()