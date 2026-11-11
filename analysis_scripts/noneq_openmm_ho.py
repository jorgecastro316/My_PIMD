#!/usr/bin/env python3
"""
OpenMM reference implementation of the classical nonequilibrium switching
protocol for the 3D anisotropic harmonic oscillator.

The script mirrors the parent/child workflow used by run_NEQ_PIMD with:
    • parent Langevin dynamics sampling state A (lambda = 0)
    • child nonequilibrium trajectories linearly ramping lambda → 1
      between two anisotropic HO states
Work is recorded as ΔE_tot between the end and start of the child run so it can
be compared directly against the text files produced by Recorder.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - import resolution depends on the environment
    import openmm as mm
    from openmm import app, unit
except ImportError:
    try:
        from simtk import openmm as mm, app, unit  # type: ignore
    except ImportError as exc:
        raise SystemExit("OpenMM is required to run this script.") from exc

# Make the repository root importable so we can reuse pimd.utils helpers.
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pimd.utils import (  # type: ignore  # pylint: disable=wrong-import-position
    get_atomic_masses_au,
    load_xyz_file_with_units,
    parse_file_field,
    read_input_file,
)

# --- Physical constants (double precision so we stay close to the classical code) ---
BOHR_TO_NM = np.float64(0.0529177210903)
AU_TIME_TO_FS = np.float64(0.02418884)
AU_MASS_TO_AMU = np.float64(1.0 / 1822.888486209)  # electron masses → dalton
DALTON_TO_KG = np.float64(1.66053906660e-27)
AVOGADRO = np.float64(6.02214076e23)
C_CM_PER_S = np.float64(2.99792458e10)
KJMOL_PER_HARTREE = np.float64(2625.499638)
HARTREE_PER_KJMOL = np.float64(1.0 / KJMOL_PER_HARTREE)


def _to_quantity(vec: np.ndarray, length_unit) -> unit.Quantity:
    """Return an OpenMM Quantity for coordinate arrays."""
    return unit.Quantity(vec.copy(), length_unit)


def _load_geometry(path_spec) -> Tuple[List[str], np.ndarray]:
    """Read an XYZ file (or list thereof) using the same helper as the classical code."""
    geom_path = parse_file_field(path_spec)
    if isinstance(geom_path, list):
        if not geom_path:
            raise ValueError("Received an empty geometry file list.")
        geom_path = geom_path[0]
    data = load_xyz_file_with_units(geom_path, unit_type="position")
    symbols = data["symbols"]
    coords_bohr = np.asarray(data["coords"], dtype=np.float64)
    return symbols, coords_bohr


def _build_topology(symbols: Sequence[str]) -> app.Topology:
    """Construct a minimal Topology for the supplied symbols."""
    top = app.Topology()
    chain = top.addChain()
    residue = top.addResidue("MOL", chain)
    for sym in symbols:
        element = app.Element.getBySymbol(sym)
        top.addAtom(sym, element, residue)
    return top


def _freq_cm_to_omega(freq_cm: Iterable[float]) -> np.ndarray:
    """Convert array-like cm^-1 values to angular frequencies in s^-1."""
    freq_cm = np.asarray(list(freq_cm), dtype=np.float64)
    return 2.0 * math.pi * C_CM_PER_S * freq_cm


def _force_constants_kjmol(
    masses_amu: np.ndarray, freqs_cm: Sequence[float]
) -> np.ndarray:
    """
    Return per-atom, per-dimension Hooke constants (kJ/mol/nm^2)
    for anisotropic HO frequencies.
    """
    if len(freqs_cm) != 3:
        raise ValueError("anisotropic_ho requires exactly three frequencies (cm^-1).")
    omega = _freq_cm_to_omega(freqs_cm)
    omega2 = omega * omega  # shape (3,)
    masses = np.asarray(masses_amu, dtype=np.float64).reshape(-1, 1) * DALTON_TO_KG
    k_si = masses * omega2  # kg / s^2 == N/m
    k_j_per_nm2 = k_si * 1.0e-18  # convert from (J/m^2) to (J/nm^2)
    return k_j_per_nm2 * (AVOGADRO / 1000.0)


def _build_anisotropic_system(
    symbols: Sequence[str],
    coords_nm: np.ndarray,
    freqA_cm: Sequence[float],
    freqB_cm: Sequence[float],
    *,
    debug: bool = False,
) -> Tuple[mm.System, app.Topology]:
    """Create an OpenMM System with a CustomExternalForce implementing V(λ)."""
    masses_au = get_atomic_masses_au(symbols)
    masses_amu = masses_au * AU_MASS_TO_AMU
    if debug:
        print(f"[debug] Building anisotropic system; masses_amu={masses_amu}")
    system = mm.System()
    for mass in masses_amu:
        system.addParticle(mass * unit.dalton)

    top = _build_topology(symbols)

    force = mm.CustomExternalForce(
        "0.5*((1-lambda)*kAx + lambda*kBx)*(x-x0)^2"
        " + 0.5*((1-lambda)*kAy + lambda*kBy)*(y-y0)^2"
        " + 0.5*((1-lambda)*kAz + lambda*kBz)*(z-z0)^2"
    )
    force.addGlobalParameter("lambda", 0.0)
    for name in ("x0", "y0", "z0", "kAx", "kAy", "kAz", "kBx", "kBy", "kBz"):
        force.addPerParticleParameter(name)

    kA = _force_constants_kjmol(masses_amu, freqA_cm)
    kB = _force_constants_kjmol(masses_amu, freqB_cm)
    for idx, (pos, ka, kb) in enumerate(zip(coords_nm, kA, kB)):
        params = [
            np.float64(pos[0]),
            np.float64(pos[1]),
            np.float64(pos[2]),
            np.float64(ka[0]),
            np.float64(ka[1]),
            np.float64(ka[2]),
            np.float64(kb[0]),
            np.float64(kb[1]),
            np.float64(kb[2]),
        ]
        force.addParticle(idx, params)
        if debug:
            print(f"[debug] Particle {idx}: pos(nm)={pos}, kA={ka}, kB={kb}")

    system.addForce(force)
    return system, top


def _build_integrator(
    thermostat: Optional[str],
    temperature: float,
    tau_au: float,
    timestep_fs: float,
    seed: int,
) -> mm.Integrator:
    """Return a Langevin or Verlet integrator matching the requested thermostat."""
    dt = timestep_fs * unit.femtoseconds
    if thermostat is None or thermostat.upper() == "NVE":
        integ = mm.VerletIntegrator(dt)
        return integ

    if thermostat.upper() in {"PILE_L", "PILE_G", "LANGEVIN", "ANDERSEN"}:
        if tau_au <= 0.0:
            raise ValueError("tau must be positive when using a thermostat.")
        tau_fs = tau_au * AU_TIME_TO_FS
        tau_ps = tau_fs / 1000.0
        gamma = 1.0 / tau_ps
        integ = mm.LangevinIntegrator(
            temperature * unit.kelvin,
            gamma / unit.picosecond,
            dt,
        )
        integ.setRandomNumberSeed(seed)
        return integ

    raise ValueError(f"Unsupported thermostat '{thermostat}'.")


def _total_energy_ha(state: mm.State) -> float:
    """Convert OpenMM energies (kJ/mol) to Hartree."""
    ek = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
    ep = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    return (ek + ep) * HARTREE_PER_KJMOL


def run_noneq_protocol(
    system: mm.System,
    topology: app.Topology,
    positions_nm: np.ndarray,
    temperature: float,
    timestep_fs: float,
    timestep_au: float,
    thermostat_parent: Optional[str],
    tau_parent_au: float,
    thermostat_child: Optional[str],
    tau_child_au: float,
    sample_steps: int,
    n_steps_child: int,
    n_samples: int,
    seed: int,
    platform_name: Optional[str] = None,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute the NE protocol and return (times_au, work_ha, E_i_ha, E_f_ha).
    """
    if sample_steps < 1:
        raise ValueError("sample_steps must be >= 1.")
    if n_steps_child < 1:
        raise ValueError("n_steps_child must be >= 1.")
    platform = None
    if platform_name:
        platform = mm.Platform.getPlatformByName(platform_name)

    parent_integrator = _build_integrator(
        thermostat_parent, temperature, tau_parent_au, timestep_fs, seed
    )
    parent_sim = app.Simulation(topology, system, parent_integrator, platform)
    parent_sim.context.setPositions(_to_quantity(positions_nm, unit.nanometer))
    parent_sim.context.setVelocitiesToTemperature(temperature * unit.kelvin, seed)
    parent_sim.context.setParameter("lambda", 0.0)

    child_integrator = _build_integrator(
        thermostat_child, temperature, tau_child_au, timestep_fs, seed + 1
    )
    child_sim = app.Simulation(topology, system, child_integrator, platform)
    child_sim.context.setParameter("lambda", 0.0)

    times: List[float] = []
    works = []
    energies_i = []
    energies_f = []

    counter = 0
    time_counter_au = np.float64(0.0)
    timestep_au = np.float64(timestep_au)

    max_parent = n_samples * sample_steps
    if debug:
        print(
            f"[debug] Starting NE loop: sample_steps={sample_steps}, "
            f"n_steps_child={n_steps_child}, parent_steps={max_parent}"
        )
    while len(works) < n_samples and counter < max_parent:
        parent_sim.step(1)
        if counter % sample_steps == 0:
            current_time = time_counter_au
            parent_state = parent_sim.context.getState(
                getPositions=True, getVelocities=True, getEnergy=True
            )
            child_sim.context.setPositions(parent_state.getPositions(asNumpy=True))
            child_sim.context.setVelocities(parent_state.getVelocities(asNumpy=True))
            child_sim.context.setParameter("lambda", 0.0)

            E_i = _total_energy_ha(parent_state)
            if debug:
                print(
                    f"[debug] Launching child #{len(works)+1}: "
                    f"parent_step={counter}, Ei={E_i:.6f} Ha"
                )
            for step_index in range(1, n_steps_child + 1):
                lam = step_index / float(n_steps_child)
                child_sim.context.setParameter("lambda", lam)
                child_sim.step(1)

            child_state = child_sim.context.getState(getEnergy=True)
            E_f = _total_energy_ha(child_state)
            works.append(E_f - E_i)
            times.append(current_time)
            energies_i.append(E_i)
            energies_f.append(E_f)
            if debug:
                print(
                    f"[debug] Child #{len(works)} complete: Ef={E_f:.6f} Ha, "
                    f"Work={works[-1]:.6f} Ha"
                )

        counter += 1
        time_counter_au += timestep_au

    if len(works) < n_samples:
        raise RuntimeError(
            f"Parent loop exited early: produced {len(works)} / {n_samples} child trajectories."
        )

    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(works, dtype=np.float64),
        np.asarray(energies_i, dtype=np.float64),
        np.asarray(energies_f, dtype=np.float64),
    )


def write_work_file(path: Path, times_au: np.ndarray, works_ha: np.ndarray):
    """Emit a Recorder-compatible text file."""
    with path.open("w") as fh:
        fh.write(f"{'#time(au)':>15s} {'Work(Ha)':>17s}\n")
        for t, w in zip(times_au, works_ha):
            fh.write(f"{t:15.8f} {w:17.8f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="OpenMM classical NE protocol for the 3D anisotropic HO."
    )
    parser.add_argument(
        "--input",
        default="input.txt",
        help="Input file compatible with the classical driver (default: input.txt)",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="Optional OpenMM platform name (CPU, Reference, CUDA, ...).",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Override prefix for output files (default: <input prefix>.openmm)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override the number of child trajectories (default: input n_steps).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed diagnostics (force constants, work samples, progress).",
    )
    parser.add_argument(
        "--parent-thermostat",
        default=None,
        help="Override parent thermostat ('NVE' to disable). Defaults to input file.",
    )
    parser.add_argument(
        "--child-thermostat",
        default=None,
        help="Override child thermostat ('NVE' to disable). Defaults to input file.",
    )
    args = parser.parse_args()

    params = read_input_file(args.input)
    if params["n_beads"] != 1:
        raise SystemExit("This OpenMM driver is restricted to classical runs (n_beads = 1).")

    n_samples = args.n_samples if args.n_samples is not None else params["n_steps"]
    sample_steps = params.get("sample_steps", 1000) or 1
    n_steps_child = params.get("n_steps_child", None)
    if n_steps_child is None:
        raise SystemExit("n_steps_child must be provided in the input file.")

    forces = params["forces"]
    if len(forces) < 1 or forces[0]["type"] != "anisotropic_ho":
        raise SystemExit("force1 must be 'anisotropic_ho' for this script.")
    freqA = [float(x) for x in forces[0]["args"]]
    if len(forces) < 2:
        freqB = freqA
    else:
        if forces[1]["type"] != "anisotropic_ho":
            raise SystemExit("force2 must also be 'anisotropic_ho'.")
        freqB = [float(x) for x in forces[1]["args"]]

    symbols, coords_bohr = _load_geometry(params["geometry_file"])
    coords_nm = coords_bohr * BOHR_TO_NM
    system, topology = _build_anisotropic_system(symbols, coords_nm, freqA, freqB, debug=args.debug)

    timestep_fs = params["timestep"] * AU_TIME_TO_FS
    prefix = args.prefix or (params.get("prefix", "simulation") + ".openmm")

    print("=== OpenMM NE protocol (anisotropic HO) ===")
    print(f"Geometry: {params['geometry_file']}")
    print(f"Symbols: {symbols}")
    print(f"Frequencies A (cm^-1): {freqA}")
    print(f"Frequencies B (cm^-1): {freqB}")
    print(f"Temperature (K): {params['temperature']}")
    print(f"Timestep (fs): {timestep_fs}")
    parent_thermo = args.parent_thermostat if args.parent_thermostat is not None else params.get("thermostat", "NVE")
    child_thermo = args.child_thermostat if args.child_thermostat is not None else params.get("thermostat_child", "NVE")
    print(f"Parent thermostat: {parent_thermo}")
    print(f"Child thermostat: {child_thermo}")
    print(f"sample_steps: {sample_steps}")
    print(f"n_steps_child: {n_steps_child}")
    print(f"n_samples: {n_samples}")

    times_au, works_ha, Ei_ha, Ef_ha = run_noneq_protocol(
        system=system,
        topology=topology,
        positions_nm=coords_nm,
        temperature=params["temperature"],
        timestep_fs=timestep_fs,
        timestep_au=params["timestep"],
        thermostat_parent=parent_thermo,
        tau_parent_au=params.get("tau", 1.0),
        thermostat_child=child_thermo,
        tau_child_au=params.get("tau_child", 1.0),
        sample_steps=sample_steps,
        n_steps_child=n_steps_child,
        n_samples=n_samples,
        seed=params.get("seed", 0),
        platform_name=args.platform,
        debug=args.debug,
    )

    work_path = Path(f"{prefix}.work.out")
    write_work_file(work_path, times_au, works_ha)

    npz_path = Path(f"{prefix}.work.npz")
    np.savez(
        npz_path,
        time_au=times_au,
        work_ha=works_ha,
        E_initial_ha=Ei_ha,
        E_final_ha=Ef_ha,
    )

    std = works_ha.std(ddof=1) if works_ha.size > 1 else 0.0

    print(f"Wrote {works_ha.size} work samples to {work_path}")
    print(f"Saved detailed arrays to {npz_path}")
    print(
        f"Work stats (Ha): mean={works_ha.mean():.6f} "
        f"std={std:.6f} min={works_ha.min():.6f} max={works_ha.max():.6f}"
    )


if __name__ == "__main__":
    main()
