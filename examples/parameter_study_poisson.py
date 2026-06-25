"""
Parameter Study: solve_poisson_bvp and solve_poisson_ivp
=========================================================

This script benchmarks the BVP and IVP Poisson solvers in grid across:
  - Grid sizes (number of radial points)
  - ODE solver tolerances
  - Radial transform parameters
  - remove_large_pts (BVP only)
  - r_interval bounds (IVP only)

Test density: normalized Gaussian with known exact analytical potential.
  rho(r) = (alpha / pi)^(3/2) * exp(-alpha * r^2)
  phi(r) = erf(sqrt(alpha) * r) / r

Results are printed as tables and saved for the report.

Run from the grid/ root directory:
    python examples/parameter_study_poisson.py
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.special import erf

from grid.atomgrid import AtomGrid
from grid.becke import BeckeWeights
from grid.molgrid import MolGrid
from grid.onedgrid import GaussLegendre, Trapezoidal
from grid.poisson import solve_poisson_bvp, solve_poisson_ivp
from grid.rtransform import BeckeRTransform, LinearFiniteRTransform

# ---------------------------------------------------------------------------
# Test density and exact potential
# ---------------------------------------------------------------------------
ALPHA = 0.5  # Gaussian exponent — moderate sharpness


def gaussian_density(pts, alpha=ALPHA):
    """Normalized s-type Gaussian density centered at origin."""
    r = np.linalg.norm(pts, axis=1)
    return (alpha / np.pi) ** 1.5 * np.exp(-alpha * r**2)


def gaussian_potential_exact(pts, alpha=ALPHA):
    """Exact analytical Coulomb potential of the Gaussian density."""
    r = np.linalg.norm(pts, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = erf(np.sqrt(alpha) * r) / r
        phi[r < 1e-12] = 2.0 * np.sqrt(alpha / np.pi)
    return phi


# ---------------------------------------------------------------------------
# Grid builder helper
# ---------------------------------------------------------------------------

def build_atomgrid(n_radial, small_r=1e-5, R=1.5, degree=29):
    """Build an AtomGrid at origin with n_radial radial points."""
    oned = GaussLegendre(n_radial)
    tf = BeckeRTransform(small_r, R=R)
    radial = tf.transform_1d_grid(oned)
    atgrid = AtomGrid(radial, center=np.zeros(3), degrees=[degree])
    return atgrid, tf


def build_molgrid(n_radial, small_r=1e-5, R=1.5, degree=29):
    """Wrap AtomGrid in a single-atom MolGrid for solve_poisson_bvp."""
    atgrid, tf = build_atomgrid(n_radial, small_r=small_r, R=R, degree=degree)
    molgrid = MolGrid(
        atnums=np.array([1]),
        atgrids=[atgrid],
        aim_weights=BeckeWeights(order=3),
        store=True,
    )
    return molgrid, tf


def relative_l2_error(computed, exact):
    """Relative L2 error between two arrays."""
    return np.sqrt(np.mean((computed - exact) ** 2)) / (np.sqrt(np.mean(exact**2)) + 1e-30)


def max_abs_error(computed, exact):
    """Maximum absolute error."""
    return np.max(np.abs(computed - exact))


# ===========================================================================
# STUDY 1: BVP — Effect of grid size (n_radial)
# ===========================================================================

def study_bvp_grid_size():
    print("\n" + "=" * 65)
    print("STUDY 1: BVP — Effect of radial grid size")
    print("=" * 65)
    print(f"{'n_radial':>10} {'rel_L2':>12} {'max_abs':>12} {'time_s':>10} {'status':>8}")
    print("-" * 65)

    grid_sizes = [30, 50, 80, 100, 150, 200]
    results = []

    for n in grid_sizes:
        molgrid, tf = build_molgrid(n)
        density = gaussian_density(molgrid.points)
        exact = gaussian_potential_exact(molgrid.points)

        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phi_fn = solve_poisson_bvp(
                    molgrid, density, tf,
                    remove_large_pts=10.0, include_origin=True
                )
            phi = phi_fn(molgrid.points)
            status = "OK"
            rel_err = relative_l2_error(phi, exact)
            abs_err = max_abs_error(phi, exact)
        except Exception as e:
            status = f"FAIL"
            rel_err = abs_err = float("nan")
        elapsed = time.time() - t0

        results.append((n, rel_err, abs_err, elapsed, status))
        print(f"{n:>10} {rel_err:>12.4e} {abs_err:>12.4e} {elapsed:>10.2f} {status:>8}")

    return results


# ===========================================================================
# STUDY 2: BVP — Effect of ODE tolerance (tol)
# ===========================================================================

def study_bvp_tolerance():
    print("\n" + "=" * 65)
    print("STUDY 2: BVP — Effect of ODE solver tolerance")
    print("=" * 65)
    print(f"{'tol':>12} {'rel_L2':>12} {'max_abs':>12} {'time_s':>10} {'status':>8}")
    print("-" * 65)

    tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    molgrid, tf = build_molgrid(100)
    density = gaussian_density(molgrid.points)
    exact = gaussian_potential_exact(molgrid.points)
    results = []

    for tol in tolerances:
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phi_fn = solve_poisson_bvp(
                    molgrid, density, tf,
                    remove_large_pts=10.0, include_origin=True,
                    ode_params={"tol": tol}
                )
            phi = phi_fn(molgrid.points)
            status = "OK"
            rel_err = relative_l2_error(phi, exact)
            abs_err = max_abs_error(phi, exact)
        except Exception as e:
            status = "FAIL"
            rel_err = abs_err = float("nan")
        elapsed = time.time() - t0

        results.append((tol, rel_err, abs_err, elapsed, status))
        print(f"{tol:>12.0e} {rel_err:>12.4e} {abs_err:>12.4e} {elapsed:>10.2f} {status:>8}")

    return results


# ===========================================================================
# STUDY 3: BVP — Effect of remove_large_pts
# ===========================================================================

def study_bvp_remove_large_pts():
    print("\n" + "=" * 65)
    print("STUDY 3: BVP — Effect of remove_large_pts threshold")
    print("=" * 65)
    print(f"{'remove_large_pts':>18} {'rel_L2':>12} {'max_abs':>12} {'time_s':>10} {'status':>8}")
    print("-" * 65)

    thresholds = [2.0, 5.0, 10.0, 50.0, 100.0, 1e6]
    molgrid, tf = build_molgrid(100)
    density = gaussian_density(molgrid.points)
    exact = gaussian_potential_exact(molgrid.points)
    results = []

    for rp in thresholds:
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phi_fn = solve_poisson_bvp(
                    molgrid, density, tf,
                    remove_large_pts=rp, include_origin=True
                )
            phi = phi_fn(molgrid.points)
            status = "OK"
            rel_err = relative_l2_error(phi, exact)
            abs_err = max_abs_error(phi, exact)
        except Exception as e:
            status = "FAIL"
            rel_err = abs_err = float("nan")
        elapsed = time.time() - t0

        results.append((rp, rel_err, abs_err, elapsed, status))
        print(f"{rp:>18.1e} {rel_err:>12.4e} {abs_err:>12.4e} {elapsed:>10.2f} {status:>8}")

    return results


# ===========================================================================
# STUDY 4: BVP — Effect of Gaussian sharpness (alpha)
# ===========================================================================

def study_bvp_density_sharpness():
    print("\n" + "=" * 65)
    print("STUDY 4: BVP — Robustness vs density sharpness (alpha)")
    print("  Sharper densities simulate nuclear cusps.")
    print("=" * 65)
    print(f"{'alpha':>10} {'rel_L2':>12} {'max_abs':>12} {'time_s':>10} {'status':>8}")
    print("-" * 65)

    alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    molgrid, tf = build_molgrid(150)
    results = []

    for alpha in alphas:
        density = gaussian_density(molgrid.points, alpha=alpha)
        exact = gaussian_potential_exact(molgrid.points, alpha=alpha)

        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phi_fn = solve_poisson_bvp(
                    molgrid, density, tf,
                    remove_large_pts=10.0, include_origin=True
                )
            phi = phi_fn(molgrid.points)
            status = "OK"
            rel_err = relative_l2_error(phi, exact)
            abs_err = max_abs_error(phi, exact)
        except Exception as e:
            status = "FAIL"
            rel_err = abs_err = float("nan")
        elapsed = time.time() - t0

        results.append((alpha, rel_err, abs_err, elapsed, status))
        print(f"{alpha:>10.1f} {rel_err:>12.4e} {abs_err:>12.4e} {elapsed:>10.2f} {status:>8}")

    return results


# ===========================================================================
# IVP grid builder — CORRECT setup matching test_poisson_ivp_on_unit_charge_distribution
# Uses LinearFiniteRTransform (finite domain) + Trapezoidal grid.
# r_interval MUST exactly match the transform bounds.
# ===========================================================================

def build_ivp_grid(n_radial, r_min=1e-3, r_max=1000.0, degree=11):
    """Build AtomGrid for IVP using LinearFiniteRTransform.

    The transform maps a finite interval [r_min, r_max] to the radial domain.
    r_interval must be (r_max, r_min) to match the transform exactly.
    Degree 11 matches the existing passing IVP test.
    """
    tf = LinearFiniteRTransform(r_min, r_max)
    oned = Trapezoidal(n_radial)
    radial = tf.transform_1d_grid(oned)
    atgrid = AtomGrid(radial, center=np.zeros(3), degrees=[degree])
    return atgrid, tf, r_max, r_min


# ===========================================================================
# STUDY 5: IVP — Effect of grid size (n_radial)
# ===========================================================================

def study_ivp_grid_size():
    print("\n" + "=" * 65)
    print("STUDY 5: IVP — Effect of radial grid size (n_radial)")
    print("  Transform: LinearFiniteRTransform(1e-3, 1000)")
    print("  r_interval: (1000.0, 1e-3) matching transform bounds exactly.")
    print("=" * 65)
    print(f"{'n_radial':>10} {'rel_L2':>12} {'max_abs':>12} {'time_s':>10} {'status':>8}")
    print("-" * 65)

    grid_sizes = [500, 1000, 2000, 5000, 10000]
    results = []

    for n in grid_sizes:
        atgrid, tf, r_max, r_min = build_ivp_grid(n)
        density = gaussian_density(atgrid.points)
        exact = gaussian_potential_exact(atgrid.points)

        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phi_fn = solve_poisson_ivp(
                    atgrid, density, tf,
                    r_interval=(r_max, r_min),
                )
            phi = phi_fn(atgrid.points)
            status = "OK"
            rel_err = relative_l2_error(phi, exact)
            abs_err = max_abs_error(phi, exact)
        except Exception as e:
            status = f"FAIL ({type(e).__name__})"
            rel_err = abs_err = float("nan")
        elapsed = time.time() - t0

        results.append((n, rel_err, abs_err, elapsed, status))
        print(f"{n:>10} {rel_err:>12.4e} {abs_err:>12.4e} {elapsed:>10.2f} {status:>8}")

    return results


# ===========================================================================
# STUDY 6: IVP — Robustness vs density sharpness (alpha)
# ===========================================================================

def study_ivp_density_sharpness():
    print("\n" + "=" * 65)
    print("STUDY 6: IVP — Robustness vs density sharpness (alpha)")
    print("  Transform: LinearFiniteRTransform(1e-3, 1000), n=10000.")
    print("  Direct comparison with BVP Study 4 (same alphas).")
    print("=" * 65)
    print(f"{'alpha':>10} {'rel_L2':>12} {'max_abs':>12} {'time_s':>10} {'status':>8}")
    print("-" * 65)

    alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    # Use the same n=10000 as the existing passing IVP test
    atgrid, tf, r_max, r_min = build_ivp_grid(10000)
    results = []

    for alpha in alphas:
        density = gaussian_density(atgrid.points, alpha=alpha)
        exact = gaussian_potential_exact(atgrid.points, alpha=alpha)

        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phi_fn = solve_poisson_ivp(
                    atgrid, density, tf,
                    r_interval=(r_max, r_min),
                )
            phi = phi_fn(atgrid.points)
            status = "OK"
            rel_err = relative_l2_error(phi, exact)
            abs_err = max_abs_error(phi, exact)
        except Exception as e:
            status = f"FAIL"
            rel_err = abs_err = float("nan")
        elapsed = time.time() - t0

        results.append((alpha, rel_err, abs_err, elapsed, status))
        print(f"{alpha:>10.1f} {rel_err:>12.4e} {abs_err:>12.4e} {elapsed:>10.2f} {status:>8}")

    return results


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  Poisson Solver Parameter Study — Issue #215")
    print("  Test density: Normalized Gaussian, phi = erf(sqrt(a)*r)/r")
    print("=" * 65)

    r1 = study_bvp_grid_size()
    r2 = study_bvp_tolerance()
    r3 = study_bvp_remove_large_pts()
    r4 = study_bvp_density_sharpness()
    r5 = study_ivp_grid_size()
    r6 = study_ivp_density_sharpness()

    print("\n" + "=" * 65)
    print("  SUMMARY OF FINDINGS")
    print("=" * 65)
    print("  See printed tables above.")
    print("  Key questions answered:")
    print("  1. Minimum n_radial for BVP convergence")
    print("  2. Optimal ODE tolerance (accuracy vs speed trade-off)")
    print("  3. Best remove_large_pts threshold")
    print("  4. Alpha (sharpness) at which BVP diverges")
    print("  5. Minimum n_radial for IVP convergence (and cost vs BVP)")
    print("  6. Alpha at which IVP accuracy degrades vs BVP")
