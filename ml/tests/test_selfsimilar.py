import torch

from nsml.selfsimilar import DTYPE, ProfileNet, effective_exponent, exact_profile


def test_exact_profiles_solve_the_profile_equation():
    y = torch.linspace(0, 5, 401, dtype=DTYPE).unsqueeze(-1).requires_grad_(True)
    for i in range(3):
        lam = 1 / (2 * i + 2)
        u = exact_profile(y, i)
        du = torch.autograd.grad(u.sum(), y)[0]
        residual = -lam * u + ((1 + lam) * y + u) * du
        assert residual.abs().max() < 1e-10
        assert abs(exact_profile(torch.ones(1, 1, dtype=DTYPE), i).item() + 0.5) < 1e-12


def test_profile_net_builds_in_the_normalisation():
    net = ProfileNet()
    y = torch.tensor([[0.0], [1.0]], dtype=DTYPE, requires_grad=True)
    u = net.profile(y)
    du = torch.autograd.grad(u.sum(), y)[0]
    assert abs(du[0].item() + 1) < 1e-12      # U'(0) = -1
    assert abs(u[1].item() + 0.5) < 1e-12     # U(1) = -1/2
    assert torch.allclose(net.profile(-y.detach()), -net.profile(y.detach()))  # odd


def test_characteristics_recover_stable_and_unstable_exponents():
    s = torch.logspace(-4, -9, 11, dtype=DTYPE)
    assert (effective_exponent(0.0, s) - 0.5).abs().max() < 1e-3
    assert (effective_exponent(1 / 27, s) - 0.25).abs().max() < 1e-3
