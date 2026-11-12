# Nonequilibrium path-integral methods for quantum free energies

This repo is a research sandbox for developing and testing nonequilibrium
methods for quantum free-energy calculations. It includes the supporting
path-integral machinery: PIMD, thermostats, PIMC, nonequilibrium switching,
and an experimental Gaussian-process regression active-learning workflow.

This is the branch currently under development. The code has been cleaned up,
consolidated, and optimized with Numba, so I recommend using it instead of
`main` for new work. The integrators and thermostats in `main`, however, were
curated in the Group, so that branch remains the trusted reference if something
here looks suspicious.

## Implemented methods

- Path-integral molecular dynamics (PIMD) with NVE, Andersen, PILE-L, and
  PILE-G propagation.
- Switching PIMD (SPIMD) and parent/child non-equilibrium switching.
- Basic, adaptive, and enhanced path-integral Monte Carlo (PIMC).
- Harmonic, shifted harmonic, anisotropic harmonic, double-well, asymmetric
  double-well, and Morse-plus-harmonic model potentials.
- BAR, Crooks, convergence, autocorrelation, and analytical model diagnostics for free energy difference calculations.
- A GPyTorch controller for uncertainty-guided selection of switching
  trajectories and iterative free-energy estimation using GPR.



## Repository layout

```text
pimd/                       Core systems, forces, integrators and simulations
controller/                 GPR and active-learning controller
analysis_scripts/           Free-energy and simulation diagnostics
analysis_scripts/gpr/       GPR-specific diagnostics
examples/                   Small tracked input examples
main_PIMD.py                Equilibrium PIMD entry point
main_SPIMD.py               Switching-PIMD entry point
main_NE_PIMD.py             Parent/child non-equilibrium entry point
main_PIMC.py                PIMC entry point
active_learning.sh          Iterative GPR/SPIMD workflow
tests/                      Small automated regression tests
```

## Installation

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

OpenMM is optional and is used only by
`analysis_scripts/noneq_openmm_ho.py` for comparison calculations.

## Quick start

Run a short harmonic PIMD calculation from the repository root:

```bash
python3 main_PIMD.py examples/harmonic/input.txt
```

The four `main_*.py` files use the same text-input format and select the
corresponding simulation method. Paths in an input file are interpreted from
the directory where the command is executed.

Run the regression tests without writing bytecode:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

## Active-learning workflow

`active_learning.sh` joins PIMC pool generation, GPR prediction, explicit
SPIMD evaluation, BAR data merging, and model retraining. It expects initial
forward and reverse seed calculations. Their locations can be supplied without
editing the script:

```bash
SEED_FORWARD=/path/to/forward/runs \
SEED_REVERSE=/path/to/reverse/runs \
VENV_PATH=/path/to/venv \
bash active_learning.sh
```

Example switching inputs are stored in `examples/active_learning`. The Slurm
resource directives and numerical parameters at the top of
`active_learning.sh` remain research defaults used in the MPSD cluster  
(should be checked for each machine and physical system)

## Current limitations

- The entry-point scripts still produce verbose debugging output.
- Cluster submission settings and convergence parameters must be selected for
  each system and computing environment.

## ToDo

- Implement and validate quantum free-energy calculations using an
  Einstein-solid reference, both with direct evaluations and with
  Gaussian-process-regression (GPR) predictions.
