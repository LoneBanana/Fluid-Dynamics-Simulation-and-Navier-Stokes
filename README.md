# Fluid Dynamics Simulation & Navier-Stokes

A project about simulating (candidate) Navier-Stokes blowup, and explaining it.

- **Project goal:** simulate Navier-Stokes blowup scenarios in real time, fast enough to run
  on most computers, in C++/OpenGL with no external libraries.
- **Phase 1 (this repo, `ml/`):** play with machine-learning models and test whether they can
  simulate fluid dynamics accurately and in a fun way.
- **Final goal:** a Next.js web app backed by FastAPI that gives a visual, quick and
  informative guide to neural networks, Navier-Stokes, and where the two meet.

## Repository layout

```
ml/                 Phase 1: PyTorch playground (importable package `nsml`)
  nsml/solvers/     reference spectral and finite-volume solvers (ground truth + baselines)
  nsml/models/      FNO, U-Net, dilated ResNet, hybrid solver+CNN, shared stepper interface
  nsml/selfsimilar.py  PINN that discovers self-similar blowup profiles
  scripts/          data generation, training, evaluation, blowup experiment
  results/          figures, GIFs and metrics from the Phase 1 runs
  tests/            solver and model correctness tests
```

See [`ml/README.md`](ml/README.md) for how to run everything and for the Phase 1 findings.

## Phase 1 in three lines

- **Hybrids win:** a cheap classical solver step plus a small learned correction was the most
  accurate model on both 1D shocks and 2D turbulence. Pure neural surrogates (FNO, U-Net,
  ResNet) were worse than a plain solver on turbulence and even invented runaway "blowups".
- **Solvers stay the backbone:** at these sizes, refining a classical grid bought the same
  accuracy for fewer FLOPs, so the C++ simulator should be built on spectral/finite-volume
  solvers, with ML as an optional correction layer.
- **Networks can find singularities:** a PINN rediscovered Burgers' blowup exponents (the
  stable λ = 1/2 to 1e-11 and the *unstable* λ = 1/4 to 3e-7), a miniature of the 2025
  discovery of unstable fluid singularities.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e "ml[dev]"
cd ml && ../.venv/bin/python -m pytest
```
