# Fluid Dynamics Simulation and Navier-Stokes

Simulating candidate Navier-Stokes blowup with classical solvers and machine learning, and explaining what is going on.

## Overview

Whether smooth solutions of the 3D Navier-Stokes equations can blow up in finite time is an open problem. This project works toward a real-time simulator for blowup-style scenarios, written in C++/OpenGL with no external libraries, and eventually a web app (Next.js front end, FastAPI back end) that explains neural networks, Navier-Stokes, and where the two meet.

Phase 1, committed here in `ml/`, is a PyTorch playground that asks a narrower question: can machine-learning models simulate fluid dynamics accurately, and which of them are worth carrying into the C++ simulator? Every model is compared against classical solvers run through the same code path.

## What's inside

```
ml/
  nsml/solvers/       pseudo-spectral Burgers and 2D Navier-Stokes (integrating-factor RK4,
                      2/3 dealiasing), MUSCL-Godunov finite volumes for Burgers
  nsml/models/        FNO, U-Net, dilated ResNet, and a hybrid "solver step + CNN correction";
                      steppers.py puts networks and solvers behind one interface
  nsml/selfsimilar.py PINN that searches for self-similar blowup profiles of inviscid Burgers,
                      trained with Adam then Levenberg-Marquardt
  nsml/train.py       unrolled (multi-step) training loop
  nsml/evaluate.py    rollouts, error / correlation / spectrum metrics, FLOP estimates
  scripts/            data generation, training, evaluation, blowup experiment
  results/            figures, GIFs, metrics.json and summary.md for each experiment
  tests/              solver, model and self-similar profile tests
```

The three Phase 1 experiments:

1. **Burgers shocks (1D):** learn the time-stepper of viscous Burgers as shocks form and merge.
2. **2D turbulence:** learn Kolmogorov flow at Re = 1000 on a 64x64 grid from 256x256 reference simulations.
3. **Blowup profiles:** a physics-informed neural network (PINN) learns both the self-similar profile and the blowup exponent of inviscid Burgers.

Phase 1 results, from `ml/results/`:

- A hybrid (cheap classical step plus a small learned correction) was the most accurate model on both problems: 0.16% mean error on Burgers, and on 2D turbulence it stayed correlated with the truth for 6.9 time units versus 3.2 for a plain 64x64 solver.
- Pure neural surrogates (FNO, U-Net, ResNet) did worse than a plain solver on 2D turbulence, and their enstrophy drifted upward in long rollouts, which looks like a blowup that 2D flow cannot have.
- Measured in FLOPs, refining a classical grid reached the same accuracy more cheaply than the hybrids at these sizes, so the planned simulator will be built on classical solvers with ML as an optional layer.
- The PINN recovered the stable exponent λ = 1/2 to within 1e-11 and the unstable λ = 1/4 to within 3e-7.

[`ml/README.md`](ml/README.md) has the full tables, figures, caveats and background.

## Getting started

Requires Python 3.11+. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e "ml[dev]"
cd ml && ../.venv/bin/python -m pytest
```

The full sequence of commands to regenerate data, train and evaluate every model is in [`ml/README.md`](ml/README.md#reproduce). Generating the 2D turbulence data and training take from minutes to about an hour each. Datasets and checkpoints are not committed and are rebuilt by the scripts.

## Status

- Phase 1 (`ml/`): done. Each model was trained with one seed on a laptop GPU, so the conclusions hold for that budget.
- Phase 2: a C++ port of the 2D spectral solver with an OpenGL viewer is in progress and not yet committed.
- 3D simulation and the web app have not been started.

## Tech

Python, PyTorch, NumPy, Matplotlib, pytest. Runs on Apple-silicon GPUs (MPS), CUDA or CPU.
