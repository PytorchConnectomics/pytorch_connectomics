from types import SimpleNamespace

import torch.nn as nn

from connectomics.training.losses import (
    GradNormLossWeighter,
    UncertaintyLossWeighter,
    build_loss_weighter,
)


def _cfg(
    strategy=None,
    *,
    gradnorm_alpha=0.5,
    gradnorm_lambda=1.0,
    gradnorm_parameter_strategy="last",
    legacy_flat=False,
):
    if legacy_flat:
        loss = SimpleNamespace(
            strategy=strategy,
            gradnorm_alpha=gradnorm_alpha,
            gradnorm_lambda=gradnorm_lambda,
            gradnorm_parameter_strategy=gradnorm_parameter_strategy,
        )
    else:
        loss = SimpleNamespace(
            loss_balancing=SimpleNamespace(
                strategy=strategy,
                gradnorm_alpha=gradnorm_alpha,
                gradnorm_lambda=gradnorm_lambda,
                gradnorm_parameter_strategy=gradnorm_parameter_strategy,
            )
        )
    return SimpleNamespace(model=SimpleNamespace(loss=loss))


def test_build_loss_weighter_uses_nested_uncertainty_strategy():
    weighter = build_loss_weighter(_cfg(strategy="uncertainty"), num_tasks=3)

    assert isinstance(weighter, UncertaintyLossWeighter)


def test_build_loss_weighter_uses_nested_gradnorm_settings():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))

    weighter = build_loss_weighter(
        _cfg(
            strategy="gradnorm",
            gradnorm_alpha=0.25,
            gradnorm_lambda=2.5,
            gradnorm_parameter_strategy="first",
        ),
        num_tasks=3,
        model=model,
    )

    assert isinstance(weighter, GradNormLossWeighter)
    assert weighter.alpha == 0.25
    assert weighter.gradnorm_lambda == 2.5
    assert len(weighter.shared_parameters) == 1
    assert weighter.shared_parameters[0] is next(model.parameters())


def test_build_loss_weighter_keeps_legacy_flat_strategy_support():
    weighter = build_loss_weighter(_cfg(strategy="uncertainty", legacy_flat=True), num_tasks=2)

    assert isinstance(weighter, UncertaintyLossWeighter)
