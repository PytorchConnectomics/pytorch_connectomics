"""CPU contracts for opt-in training speed controls."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from monai.data import MetaTensor

from connectomics.config.pipeline import from_dict
from connectomics.models.architectures.mednext_models import (
    MEDNEXT_AVAILABLE,
    build_mednext,
    build_mednext_custom,
)
from connectomics.training.lightning.callbacks import EMAWeightsCallback
from connectomics.training.lightning.model import ConnectomicsModule
from connectomics.training.lightning.trainer import _select_ddp_strategy_kwargs, create_trainer
from connectomics.training.optimization.build import build_optimizer


@pytest.mark.parametrize(
    "arch,ds,freeze,heads,explicit,unused",
    [
        ("mednext", False, False, {}, False, True),
        ("mednext", False, True, {}, False, False),
        ("mednext", False, True, {"affinity": {"out_channels": 3}}, False, True),
        ("mednext_custom", False, True, {"affinity": {"out_channels": 3}}, False, True),
        ("mednext", True, True, {}, False, True),
        ("monai_unet", False, False, {}, False, False),
        ("monai_unet", False, False, {}, True, True),
    ],
)
@pytest.mark.parametrize("static", [False, True])
def test_ddp(arch, ds, freeze, heads, explicit, unused, static):
    cfg = from_dict(
        {
            "model": {
                "arch": {"type": arch},
                "loss": {"deep_supervision": ds},
                "mednext_freeze_dummy_tensor": freeze,
                "heads": heads,
            },
            "optimization": {"ddp_static_graph": static, "ddp_find_unused_parameters": explicit},
        }
    )
    assert _select_ddp_strategy_kwargs(cfg) == {
        "find_unused_parameters": unused,
        "static_graph": static,
    }


@pytest.mark.parametrize("flag", [None, True, False])
def test_trainer_flags(flag, monkeypatch):
    cfg = from_dict(
        {"system": {"num_gpus": 0}, "optimization": {"benchmark": flag, "deterministic": flag}}
    )
    monkeypatch.setattr("connectomics.training.lightning.trainer.pl.Trainer", lambda **kw: kw)
    result = create_trainer(cfg, mode="test")
    assert result["benchmark"] is flag and result["deterministic"] is flag


def test_optimizer_groups_and_fallback(caplog):
    cfg = from_dict({})
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.LayerNorm(3))
    other = deepcopy(model)
    old = build_optimizer(cfg, model)
    assert len(old.param_groups) == len(list(model.parameters()))
    cfg.optimization.optimizer.merge_param_groups = True
    cfg.optimization.optimizer.fused = True
    new = build_optimizer(cfg, other)
    assert "using foreach=None" in caplog.text

    def settings(opt):
        return [(g["lr"], g["weight_decay"]) for g in opt.param_groups for _ in g["params"]]

    assert sorted(settings(old)) == sorted(settings(new))
    assert len(new.param_groups) == len(set(settings(old)))
    for a, b in zip(model.parameters(), other.parameters()):
        a.grad = torch.ones_like(a)
        b.grad = a.grad.clone()
    old.step()
    new.step()
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=0)


def test_ema_reference():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
    module = SimpleNamespace(model=model, device=torch.device("cpu"))
    cb = EMAWeightsCallback()
    cb.on_train_start(SimpleNamespace(global_step=0), module)
    reference = deepcopy(model.state_dict())
    for _ in range(5):
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.add_(0.1)
                reference[name].mul_(0.9).add_(p, alpha=0.1)
            for name, b in model.named_buffers():
                b.add_(1)
                reference[name] = b.clone()
        cb._update_ema(module, 0.9)
    assert cb._ema_state.keys() == reference.keys()
    for key, value in reference.items():
        torch.testing.assert_close(value, cb._ema_state[key], atol=1e-7, rtol=0)


@pytest.mark.parametrize("mode", [None, "max-autotune"])
@pytest.mark.skipif(not hasattr(torch.nn.Module, "compile"), reason="Module.compile unavailable")
def test_compile_and_metatensor(mode, monkeypatch, caplog):
    cfg = from_dict(
        {
            "model": {
                "compile": True,
                "compile_mode": mode,
                "loss": {"losses": [{"function": "BCEWithLogitsLoss", "weight": 1.0}]},
            }
        }
    )
    module = ConnectomicsModule(cfg, model=torch.nn.Conv3d(1, 1, 1))
    calls = []
    monkeypatch.setattr(torch.nn.Module, "compile", lambda self, **kw: calls.append((self, kw)))
    module.configure_model()
    assert not calls and "requires CUDA" in caplog.text
    from pytorch_lightning.accelerators import CUDAAccelerator

    module._trainer = SimpleNamespace(accelerator=CUDAAccelerator())
    module.configure_model()
    module.configure_model()
    assert calls == [(module.model, {} if mode is None else {"mode": mode})]
    module._trainer = None
    module.log_dict = lambda *a, **kw: None
    x = torch.randn(1, 1, 4, 4, 4)
    y = torch.rand_like(x)
    types = []
    module.model.register_forward_pre_hook(lambda m, args: types.append(type(args[0])))
    plain = module.training_step({"image": x, "label": y}, 0)
    meta = module.training_step({"image": MetaTensor(x), "label": MetaTensor(y)}, 0)
    assert types == [torch.Tensor, torch.Tensor]
    torch.testing.assert_close(plain, meta)


@pytest.mark.skipif(not hasattr(torch.nn.Module, "compile"), reason="Module.compile unavailable")
def test_real_compile_keys():
    model = torch.nn.Linear(2, 2)
    state, names = deepcopy(model.state_dict()), list(dict(model.named_parameters()))
    model.compile()
    assert list(model.state_dict()) == list(state)
    assert list(dict(model.named_parameters())) == names
    model.load_state_dict(state, strict=True)
    x = torch.ones(1, 2)
    torch.testing.assert_close(
        model(x), torch.nn.functional.linear(x, state["weight"], state["bias"])
    )


@pytest.mark.parametrize("standard", [False, True])
@pytest.mark.skipif(
    not MEDNEXT_AVAILABLE,
    reason="nnunet_mednext unavailable",
)
def test_mednext_resume_and_freeze(standard):
    cfg = from_dict(
        {
            "model": {
                "arch": {"type": "mednext_custom"},
                "loss": {"losses": [{"function": "BCEWithLogitsLoss", "weight": 1.0}]},
                "mednext": {"base_channels": 2, "exp_r": 2, "block_counts": [1] * 9},
            }
        }
    )
    builder = build_mednext if standard else build_mednext_custom
    model = builder(cfg)
    assert model.model.dummy_tensor.requires_grad
    optimizer = build_optimizer(cfg, model)
    x = torch.randn(1, 1, 32, 32, 32)
    model(x).sum().backward()
    optimizer.step()
    state, optstate = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
    resumed = builder(cfg)
    resumed.load_state_dict(state, strict=True)
    opt = build_optimizer(cfg, resumed)
    opt.load_state_dict(optstate)
    resumed(x).sum().backward()
    opt.step()
    cfg.model.mednext_freeze_dummy_tensor = True
    frozen = builder(cfg)
    frozen.load_state_dict(state, strict=True)
    frozen(x).sum().backward()
    assert "model.dummy_tensor" in frozen.state_dict()
    assert all(p.grad is not None for p in frozen.parameters() if p.requires_grad)


@pytest.mark.parametrize("scaler", [None, object()])
def test_fused_clipping_without_scaler(scaler, monkeypatch):
    cfg = from_dict(
        {
            "model": {
                "loss": {"losses": [{"function": "BCEWithLogitsLoss", "weight": 1.0}]},
            }
        }
    )
    module = ConnectomicsModule(cfg, model=torch.nn.Linear(4, 4))
    monkeypatch.setattr(
        ConnectomicsModule,
        "trainer",
        property(lambda self: SimpleNamespace(precision_plugin=SimpleNamespace(scaler=scaler))),
    )
    delegated = []
    module.clip_gradients = lambda *args: delegated.append(args)
    optimizer = torch.optim.AdamW(module.model.parameters())
    optimizer.defaults["fused"] = True
    for p in module.model.parameters():
        p.grad = torch.full_like(p, 10.0)
    module.configure_gradient_clipping(optimizer, 1.0, "norm")
    norm = torch.stack([p.grad.norm() for p in module.model.parameters()]).norm()
    if scaler is None:
        assert not delegated and norm.item() == pytest.approx(1.0, rel=1e-5)
    else:
        assert delegated == [(optimizer, 1.0, "norm")] and norm.item() > 1.0


def _small_mednext_cfg(**model):
    return from_dict(
        {
            "model": {
                "arch": {"type": "mednext_custom"},
                "loss": {"losses": [{"function": "BCEWithLogitsLoss", "weight": 1.0}]},
                "mednext": {"base_channels": 2, "exp_r": 2, "block_counts": [1] * 9},
                **model,
            }
        }
    )


@pytest.mark.parametrize(
    "model_cfg,match",
    [
        ({"mednext_depthwise_impl": "shift_sum"}, "requires model.compile"),
        ({"mednext_depthwise_impl": "bogus", "compile": True}, "must be 'native' or 'shift_sum'"),
    ],
)
def test_shift_sum_config_errors(model_cfg, match):
    with pytest.raises(ValueError, match=match):
        build_mednext_custom(_small_mednext_cfg(**model_cfg))


@pytest.mark.parametrize("standard", [False, True])
@pytest.mark.skipif(
    not MEDNEXT_AVAILABLE,
    reason="nnunet_mednext unavailable",
)
def test_shift_sum_matches_native(standard):
    from connectomics.models.architectures.mednext_models import ShiftSumDepthwiseConv3d

    builder = build_mednext if standard else build_mednext_custom
    torch.manual_seed(0)
    native = builder(_small_mednext_cfg())
    shifted = builder(_small_mednext_cfg(mednext_depthwise_impl="shift_sum", compile=True))
    assert list(shifted.state_dict()) == list(native.state_dict())
    shifted.load_state_dict(native.state_dict(), strict=True)
    convs = [
        m
        for m in native.modules()
        if type(m) is torch.nn.Conv3d and m.groups == m.in_channels == m.out_channels > 1
    ]
    expected = [m for m in convs if m.stride == (1, 1, 1)]
    swapped = [m for m in shifted.modules() if isinstance(m, ShiftSumDepthwiseConv3d)]
    assert len(swapped) == len(expected) > 0
    # stride-2 depthwise downsampling convs exist and stay native
    assert any(m.stride != (1, 1, 1) for m in convs)
    assert all(m.stride == (1, 1, 1) for m in swapped)
    x = torch.randn(2, 1, 32, 32, 32)
    out_native, out_shifted = native(x), shifted(x)
    torch.testing.assert_close(out_shifted, out_native, rtol=1e-5, atol=1e-5)
    out_native.square().mean().backward()
    out_shifted.square().mean().backward()
    for (name, a), (_, b) in zip(native.named_parameters(), shifted.named_parameters()):
        if a.grad is None:
            assert b.grad is None, name
            continue
        torch.testing.assert_close(b.grad, a.grad, rtol=1e-4, atol=1e-6, msg=name)


def test_shift_sum_autocast_dtype():
    from connectomics.models.architectures.mednext_models import ShiftSumDepthwiseConv3d

    conv = torch.nn.Conv3d(4, 4, 3, padding=1, groups=4)
    shifted = deepcopy(conv)
    shifted.__class__ = ShiftSumDepthwiseConv3d
    x = torch.randn(1, 4, 8, 8, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ref, out = conv(x), shifted(x)
    assert out.dtype == ref.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("compile_enabled", [False, True])
def test_metatensor_input_contract(compile_enabled):
    cfg = from_dict(
        {
            "model": {
                "compile": compile_enabled,
                "loss": {"losses": [{"function": "BCEWithLogitsLoss", "weight": 1.0}]},
            }
        }
    )
    model = torch.nn.Conv3d(1, 1, 1)
    module = ConnectomicsModule(cfg, model=model)
    module.log_dict = lambda *a, **kw: None
    x, y = torch.randn(1, 1, 4, 4, 4), torch.rand(1, 1, 4, 4, 4)
    seen = []
    model.register_forward_pre_hook(lambda m, args: seen.append(type(args[0])))
    loss = module.training_step({"image": MetaTensor(x), "label": MetaTensor(y)}, 0)
    loss.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    cfg.model.compile = False
    plain = module.training_step({"image": x, "label": y}, 0)
    plain.backward()
    assert seen[0] is (torch.Tensor if compile_enabled else MetaTensor)
    torch.testing.assert_close(loss, plain)
    for p, grad in zip(model.parameters(), grads):
        torch.testing.assert_close(p.grad, grad)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
@pytest.mark.parametrize(
    "shadow_dtype,param_dtype",
    [
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_ema_bit_exact(device, shadow_dtype, param_dtype):
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)).to(
        device=device, dtype=param_dtype
    )
    model[0].bias.requires_grad_(False)
    module = SimpleNamespace(model=model, device=torch.device(device))
    cb = EMAWeightsCallback()
    cb.on_train_start(SimpleNamespace(global_step=0), module)
    cb._ema_state = {k: v.to(shadow_dtype) for k, v in cb._ema_state.items()}
    del cb._ema_state["1.bias"]
    ref = deepcopy(cb)
    for _ in range(4):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.125)
            for name, p in model.named_parameters():
                if p.requires_grad:
                    ref._update_single(name, p.detach(), 0.9, module.device, use_decay=True)
        cb._update_ema(module, 0.9)
        for name in ref._ema_state:
            assert torch.equal(cb._ema_state[name], ref._ema_state[name]), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_ema_device_fallback():
    model = torch.nn.Linear(3, 3).cuda()
    module = SimpleNamespace(model=model, device=torch.device("cuda"))
    cb = EMAWeightsCallback()
    cb.on_train_start(SimpleNamespace(global_step=0), module)
    cb._ema_state = {k: v.cpu() for k, v in cb._ema_state.items()}
    ref = deepcopy(cb)
    with torch.no_grad():
        for name, p in model.named_parameters():
            ref._update_single(name, p.detach(), 0.9, module.device, use_decay=True)
    cb._update_ema(module, 0.9)
    for name in ref._ema_state:
        assert torch.equal(cb._ema_state[name], ref._ema_state[name])


@pytest.mark.parametrize("api", ["compile", "get_autocast_dtype"])
def test_shift_sum_missing_api(api, monkeypatch):
    monkeypatch.delattr(torch.nn.Module if api == "compile" else torch, api, raising=False)
    with pytest.raises(RuntimeError, match="shift_sum requires"):
        build_mednext_custom(_small_mednext_cfg(compile=True, mednext_depthwise_impl="shift_sum"))


def test_compile_missing_api(monkeypatch):
    from pytorch_lightning.accelerators import CUDAAccelerator

    module = ConnectomicsModule(_small_mednext_cfg(compile=True), model=torch.nn.Linear(2, 2))
    module._trainer = SimpleNamespace(accelerator=CUDAAccelerator())
    monkeypatch.delattr(torch.nn.Module, "compile", raising=False)
    with pytest.raises(RuntimeError, match="Module.compile"):
        module.configure_model()


def test_shift_sum_cpu_rejected():
    module = ConnectomicsModule(
        _small_mednext_cfg(compile=True, mednext_depthwise_impl="shift_sum"),
        model=torch.nn.Linear(2, 2),
    )
    with pytest.raises(ValueError, match="requires CUDA"):
        module.configure_model()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kernel,bias", [(3, True), (5, True), (3, False)])
def test_shift_sum_eager(dtype, kernel, bias):
    from connectomics.models.architectures.mednext_models import ShiftSumDepthwiseConv3d

    native = torch.nn.Conv3d(2, 2, kernel, padding=kernel // 2, groups=2, bias=bias).to(dtype)
    shifted = deepcopy(native)
    shifted.__class__ = ShiftSumDepthwiseConv3d
    a = torch.randn(1, 2, 6, 6, 6, dtype=dtype, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    x, y = native(a), shifted(b)
    torch.testing.assert_close(x, y)
    x.square().sum().backward()
    y.square().sum().backward()
    torch.testing.assert_close(a.grad, b.grad)
    for p, q in zip(native.parameters(), shifted.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize(
    "change",
    [
        {"stride": 2},
        {"dilation": 2},
        {"padding": 0},
        {"padding_mode": "reflect"},
        {"groups": 1},
        {"kernel_size": 2},
        {},
    ],
)
def test_shift_sum_eligibility(change):
    from connectomics.models.architectures.mednext_models import _is_shift_sum_eligible

    args = {"in_channels": 2, "out_channels": 2, "kernel_size": 3, "padding": 1, "groups": 2}
    args.update(change)
    assert _is_shift_sum_eligible(torch.nn.Conv3d(**args)) == (not change)


@pytest.mark.parametrize("neighbor", ["reject", "zero", "none", "sgd", "cpu", "bf16", "unfused"])
def test_fused_preflight(neighbor, monkeypatch):
    import connectomics.training.lightning.trainer as owner

    cfg = from_dict(
        {
            "system": {"num_gpus": 1},
            "optimization": {
                "precision": "16-mixed",
                "gradient_clip_val": 1.0,
                "optimizer": {"name": "adamw", "fused": True},
            },
        }
    )
    if neighbor == "zero":
        cfg.optimization.gradient_clip_val = 0
    if neighbor == "none":
        cfg.optimization.gradient_clip_val = None
    if neighbor == "sgd":
        cfg.optimization.optimizer.name = "sgd"
    if neighbor == "cpu":
        cfg.system.num_gpus = 0
    if neighbor == "bf16":
        cfg.optimization.precision = "bf16-mixed"
    if neighbor == "unfused":
        cfg.optimization.optimizer.fused = False
    monkeypatch.setattr(owner, "resolve_accelerator_type", lambda *a: "cuda")
    monkeypatch.setattr(owner, "get_accelerator_device_count", lambda *a: 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(owner.pl, "Trainer", lambda **kw: kw)
    if neighbor == "reject":
        with pytest.raises(
            ValueError,
            match=(
                r"optimization\.optimizer\.fused=true cannot be combined with "
                r"optimization\.gradient_clip_val > 0 under fp16 GradScaler precision; "
                r"use bf16-mixed precision or set optimization\.optimizer\.fused=false"
            ),
        ):
            create_trainer(cfg, mode="test")
    else:
        create_trainer(cfg, mode="test")


def _fit_module(compile_enabled=False, fused=False):
    cfg = from_dict(
        {
            "model": {
                "compile": compile_enabled,
                "loss": {"losses": [{"function": "BCEWithLogitsLoss", "weight": 1.0}]},
            },
            "optimization": {
                "optimizer": {"name": "adamw", "fused": fused},
                "scheduler": {"name": "none"},
            },
        }
    )
    module = ConnectomicsModule(cfg, model=torch.nn.Conv3d(1, 1, 1))
    module.configure_optimizers = lambda: build_optimizer(cfg, module.model)
    return module


def _fit_data():
    generator = torch.Generator().manual_seed(17)
    # Identical batches make resumption independent of loader iterator position.
    item = {
        "image": torch.randn(1, 4, 4, 4, generator=generator),
        "label": torch.rand(1, 4, 4, 4, generator=generator),
    }
    return torch.utils.data.DataLoader([item] * 16, batch_size=1)


def _fit_trainer(steps, accelerator="cpu", **kwargs):
    import pytorch_lightning as pl

    return pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_steps=steps,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        **kwargs,
    )


def test_cpu_trainer_accumulation():
    trainer = _fit_trainer(2)
    trainer.fit(_fit_module(), train_dataloaders=_fit_data())
    assert trainer.global_step == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compiled_fused_checkpoint_continuation(tmp_path):
    if not hasattr(torch.nn.Module, "compile"):
        pytest.skip("torch.nn.Module.compile unavailable")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bf16 unavailable")
    torch.manual_seed(7)
    a = _fit_module(True, True)
    initial = deepcopy(a.state_dict())
    full = _fit_trainer(4, "gpu", precision="bf16-mixed", deterministic=True)
    full.fit(a, train_dataloaders=_fit_data())
    b = _fit_module(True, True)
    b.load_state_dict(initial)
    first = _fit_trainer(2, "gpu", precision="bf16-mixed", deterministic=True)
    first.fit(b, train_dataloaders=_fit_data())
    path = tmp_path / "resume.ckpt"
    first.save_checkpoint(path)
    saved = torch.load(path, weights_only=False)
    assert not any("_orig_mod." in k for k in saved["state_dict"])
    c = _fit_module(True, True)
    resumed = _fit_trainer(4, "gpu", precision="bf16-mixed", deterministic=True)
    resumed.fit(c, train_dataloaders=_fit_data(), ckpt_path=path)
    assert full.global_step == resumed.global_step == 4
    # Compiled reductions can differ in order across fresh model instances.
    for p, q in zip(a.parameters(), c.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
    left, right = [t.optimizers[0].state_dict()["state"] for t in (full, resumed)]
    assert left.keys() == right.keys()
    for key in left:
        for name in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(left[key][name], right[key][name], rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("autocast", [False, True])
def test_compiled_shift_sum_stack(autocast):
    from connectomics.models.architectures.mednext_models import ShiftSumDepthwiseConv3d

    if not hasattr(torch.nn.Module, "compile") or not hasattr(torch, "get_autocast_dtype"):
        pytest.skip("compile/autocast APIs unavailable")
    if autocast and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bf16 unavailable")
    torch.manual_seed(11)
    native = torch.nn.Sequential(
        *[
            torch.nn.Conv3d(2, 2, k, padding=k // 2, groups=2, bias=bias)
            for k, bias in [(3, True), (5, True), (3, False)]
        ]
    ).cuda()
    shifted, reference = deepcopy(native), deepcopy(native).double()
    for conv in shifted:
        conv.__class__ = ShiftSumDepthwiseConv3d
    shifted.compile()
    x = torch.randn(1, 2, 6, 6, 6, device="cuda")
    a, b, c = [
        x.to(dtype).detach().requires_grad_()
        for dtype in (torch.float32, torch.float32, torch.float64)
    ]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        left, right = native(a), shifted(b)
    exact = reference(c)
    tol = 2e-2 if autocast else 1e-5  # Both outputs round to bf16 under autocast.
    torch.testing.assert_close(left, right, rtol=tol, atol=tol)
    left.sum().backward()
    right.sum().backward()
    exact.sum().backward()
    pairs = [(a.grad, b.grad, c.grad)] + [
        (p.grad, q.grad, r.grad)
        for p, q, r in zip(native.parameters(), shifted.parameters(), reference.parameters())
    ]
    for left_grad, right_grad, _exact_grad in pairs:
        torch.testing.assert_close(left_grad, right_grad, rtol=tol, atol=tol)
    # Compare forward accuracy with the same quantized operands for bf16.
    if autocast:
        reference.load_state_dict(
            {k: v.bfloat16().double() for k, v in native.state_dict().items()}
        )
        exact = reference(x.bfloat16().double()).bfloat16().double()
    native_error = (left.double() - exact).abs().max()
    shift_error = (right.double() - exact).abs().max()
    assert shift_error <= native_error + (1e-7 if not autocast else 1e-3)
