import pytest
import torch

from nsml.data import BurgersConfig, NS2DConfig
from nsml.models import ARCH, build_stepper
from nsml.train import TrainConfig, unroll_length

META = {
    "burgers": {"name": "burgers", **BurgersConfig().__dict__},
    "ns2d": {"name": "ns2d", **NS2DConfig().__dict__},
}


@pytest.mark.parametrize("dataset,name", [(d, m) for d, models in ARCH.items() for m in models])
def test_learned_steppers_preserve_shape_and_backprop(dataset, name):
    meta = META[dataset]
    shape = (2, meta["n"]) if dataset == "burgers" else (2, meta["n"], meta["n"])
    stepper = build_stepper(name, meta, torch.device("cpu"))
    x = torch.randn(shape)
    y = stepper(x)
    assert y.shape == x.shape
    y.pow(2).mean().backward()
    assert all(p.grad is not None for p in stepper.parameters())


def test_fno_is_translation_equivariant():
    stepper = build_stepper("fno", META["burgers"], torch.device("cpu"))
    x = torch.randn(1, 256)
    shifted = stepper(torch.roll(x, 17, -1))
    assert torch.allclose(shifted, torch.roll(stepper(x), 17, -1), atol=1e-5)


def test_unroll_schedule_ramps_to_max():
    cfg = TrainConfig(iters=100, max_unroll=4)
    lengths = [unroll_length(it, cfg) for it in range(100)]
    assert lengths[0] == 1 and lengths[-1] == 4 and lengths == sorted(lengths)


def test_solver_rollouts_keep_native_resolution_state():
    """A finer baseline must not be re-sampled to the data grid between output frames."""
    from nsml.solvers import NS2DSpectral, kolmogorov_forcing
    from nsml.train import rollout

    meta = META["ns2d"]
    stepper = build_stepper("spectral-128", meta, torch.device("cpu"))
    w0 = torch.randn(1, 64, 64)
    frames = rollout(stepper, w0, 3)
    native = stepper.native
    reference = NS2DSpectral(128, meta["nu"], native.solver.dt, drag=meta["drag"],
                             forcing=kolmogorov_forcing(128, meta["k_f"]))
    expected = reference.simulate(w0, t_final=3 * meta["save_every"], save_every=meta["save_every"], n_out=64)
    assert torch.allclose(frames[:, 1:], expected[:, 1:], atol=1e-4)
