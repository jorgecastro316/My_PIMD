# Nonequilibrium path-integral methods for quantum free energies

This repo is a research sandbox for developing and testing
nonequilibrium methods for quantum free-energy calculations. It provides the
supporting path-integral machinery—including PIMD, thermostatting, PIMC,
nonequilibrium switching protocols, and an experimental Gaussian-process
active-learning workflow—needed to construct and evaluate these methods.

## Implemented methods

- Path-integral molecular dynamics (PIMD) with NVE, Andersen, PILE-L, and
  PILE-G propagation routes.
- Switching PIMD (SPIMD) and parent/child non-equilibrium switching.
- Basic, adaptive, and enhanced path-integral Monte Carlo (PIMC).
- Harmonic, shifted harmonic, anisotropic harmonic, double-well, asymmetric
  double-well, and Morse-plus-harmonic model potentials.
- BAR, Crooks, convergence, autocorrelation, and analytical model diagnostics.
- A GPyTorch controller for uncertainty-guided selection of switching
  trajectories and iterative free-energy estimation.



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

## Current status

The core is a working research prototype rather than a production simulation
package. Short smoke tests currently pass for one- and multi-bead PIMD, SPIMD,
non-equilibrium parent/child switching, basic PIMC, adaptive PIMC, and enhanced
PIMC. The Morse-plus-harmonic force has also been checked against finite
differences. 


## Current status

This repository is a working research prototype rather than a
production-ready simulation package. The included short smoke tests pass for
one- and multi-bead PIMD, SPIMD, nonequilibrium parent–child switching, and the
basic, adaptive, and enhanced PIMC workflows. The analytical forces for the
Morse-plus-harmonic model also agree with finite-difference derivatives.

These checks establish basic execution and internal consistency, but not
long-time convergence or quantitative physical accuracy.

## Current limitations

- The entry-point scripts still produce verbose debugging output.
- Automated tests cover the core mechanics but not long-time convergence or
  the complete active-learning workflow.
- Cluster submission settings and convergence parameters must be selected for
  each system and computing environment.

## ToDo

- Implement and validate quantum free-energy calculations using an
  Einstein-solid reference, both with direct evaluations and with
  Gaussian-process-regression (GPR) predictions.
