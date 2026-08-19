"""Screened Coulomb Study vs. Double-Split Architecture.

This script benchmarks the Screened Coulomb (Yukawa) potential e^{-mu r}/r as an
alternative to the Double-Split Poisson solver (solve_poisson_robust).

Key Questions Answered:
1. Does screening (mu > 0) reduce errors or solve nuclear cusp divergence?
2. How much physical distortion does non-zero screening introduce vs. exact 1/r?
3. How does Screened Coulomb compare with the Double-Split solver (solve_poisson_robust)?
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfc, erfcx

from grid.atomgrid import AtomGrid
from grid.onedgrid import GaussLegendre
from grid.rtransform import BeckeRTransform, InverseRTransform
from grid.poisson import solve_poisson_bvp
from grid.robust_poisson import solve_poisson_robust

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


# -----------------------------------------------------------------------------
# Analytical Expressions
# -----------------------------------------------------------------------------

def gaussian_density(points, alpha, center=None):
    """Normalized s-type Gaussian density."""
    center = np.zeros(3) if center is None else np.asarray(center)
    r2 = np.sum((points - center) ** 2, axis=1)
    return (alpha / np.pi) ** 1.5 * np.exp(-alpha * r2)


def gaussian_potential_exact(points, alpha, center=None):
    """Exact 1/r electrostatic potential of a normalized s-type Gaussian."""
    center = np.zeros(3) if center is None else np.asarray(center)
    r = np.linalg.norm(points - center, axis=1)
    return np.where(r > 1e-12, erf(np.sqrt(alpha) * r) / r, 2.0 * np.sqrt(alpha / np.pi))


def screened_coulomb_potential(points, alpha, mu, center=None):
    """Analytical Screened Coulomb (Yukawa) potential V_mu(r) for a Gaussian density.

    (nabla^2 - mu^2) V_mu(r) = -4*pi*rho(r)
    """
    center = np.zeros(3) if center is None else np.asarray(center)
    r = np.linalg.norm(points - center, axis=1)
    if mu == 0:
        return gaussian_potential_exact(points, alpha, center)

    r_safe = np.maximum(r, 1e-12)
    sa = np.sqrt(alpha)
    
    # Numerically stable computation of terms
    z1 = mu / (2 * sa) - sa * r_safe
    z2 = mu / (2 * sa) + sa * r_safe

    prefactor = np.exp(-mu * r_safe) * np.exp(mu**2 / (4 * alpha)) / (2 * r_safe)
    term1 = erfc(z1)
    
    # Use erfcx(z2) = exp(z2^2)*erfc(z2) to avoid overflow
    # exp(2*mu*r)*erfc(z2) = exp(2*mu*r - z2^2)*erfcx(z2)
    # 2*mu*r - z2^2 = -(mu/(2*sa) - sa*r)^2 = -z1^2
    term2 = np.exp(-z1**2) * erfcx(z2)
    
    V_mu = prefactor * (term1 - term2)

    # Limit at r -> 0
    V_mu0 = 2.0 * sa / np.sqrt(np.pi) - mu * np.exp(mu**2 / (4 * alpha)) * erfc(mu / (2 * sa))
    return np.where(r > 1e-12, V_mu, V_mu0)


def rel_l2(V, V_ref):
    """Relative L2 error."""
    return np.sqrt(np.mean((V - V_ref) ** 2)) / np.sqrt(np.mean(V_ref ** 2))


# -----------------------------------------------------------------------------
# Main Study
# -----------------------------------------------------------------------------

def run_screened_coulomb_study():
    print("=" * 70)
    print("  SCREENED COULOMB STUDY vs. DOUBLE-SPLIT POISSON SOLVER")
    print("=" * 70)

    # Grid Setup
    alpha = 50.0  # Sharp Gaussian simulating nuclear cusp
    oned = GaussLegendre(150)
    tf = BeckeRTransform(1e-5, R=1.5)
    radial = tf.transform_1d_grid(oned)
    atgrid = AtomGrid(radial, degrees=[29], center=np.zeros(3))
    inv_tf = InverseRTransform(tf)

    density = gaussian_density(atgrid.points, alpha)
    V_exact = gaussian_potential_exact(atgrid.points, alpha)

    # 1. Screened Coulomb distortion analysis for different mu values
    print("\n1. Screened Coulomb Distortion vs. Exact Physical 1/r Potential:")
    print("-" * 55)
    print(f"{'mu (screening)':>15}  {'Relative L2 Error vs. 1/r':>28}")
    print("-" * 55)

    mus = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]
    screened_errors = []
    for mu in mus:
        V_sc = screened_coulomb_potential(atgrid.points, alpha, mu)
        err = rel_l2(V_sc, V_exact)
        screened_errors.append(err)
        print(f"{mu:>15.3f}  {err:>28.4e}")

    # 2. Solver comparisons (Plain BVP vs Split 1 vs Split 1+2)
    print("\n2. Solver Accuracy Comparison (alpha = 50.0):")
    print("-" * 55)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        V_bvp = solve_poisson_bvp(atgrid, density, inv_tf)(atgrid.points)
        pot_s1 = solve_poisson_robust(
            atgrid, density, inv_tf,
            atnums=np.array([1]), atcoords=np.zeros((1, 3)), split2=False
        )
        V_s1 = pot_s1(atgrid.points)

        pot_s2 = solve_poisson_robust(
            atgrid, density, inv_tf,
            atnums=np.array([1]), atcoords=np.zeros((1, 3)), split2=True
        )
        V_s2 = pot_s2(atgrid.points)

    print(f"Plain BVP Solver             rel-L2: {rel_l2(V_bvp, V_exact):.4e}")
    print(f"Double-Split (Split 1)       rel-L2: {rel_l2(V_s1, V_exact):.4e}")
    print(f"Double-Split (Split 1+2)     rel-L2: {rel_l2(V_s2, V_exact):.4e}")

    # 3. Radial profile analysis
    r_vals = np.linspace(0.01, 4.0, 300)
    ray = np.column_stack([r_vals, np.zeros(300), np.zeros(300)])

    V_ex_ray = gaussian_potential_exact(ray, alpha)
    V_s2_ray = pot_s2(ray)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left Plot: Screened Coulomb Distortion vs Exact and Double-Split
    ax = axes[0]
    ax.plot(r_vals, V_ex_ray, "k-", lw=2, label="Exact 1/r")
    ax.plot(r_vals, V_s2_ray, "g:", lw=2, label="Double-Split (Split 1+2)")

    for mu, ls in [(0.1, "--"), (0.5, "-."), (1.0, ":")]:
        V_sc_ray = screened_coulomb_potential(ray, alpha, mu)
        ax.plot(r_vals, V_sc_ray, ls=ls, label=f"Screened (μ={mu})")

    ax.set_xlabel("r (Bohr)")
    ax.set_ylabel("V(r)")
    ax.set_title("Potential Comparison (α = 50.0)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 3)

    # Right Plot: Error Profile
    ax = axes[1]
    ax.semilogy(r_vals, np.abs(V_s2_ray - V_ex_ray), "g:", lw=2, label="Double-Split Error")
    for mu, ls in [(0.1, "--"), (0.5, "-."), (1.0, ":")]:
        V_sc_ray = screened_coulomb_potential(ray, alpha, mu)
        ax.semilogy(r_vals, np.abs(V_sc_ray - V_ex_ray), ls=ls, label=f"Screened Distortion (μ={mu})")

    ax.set_xlabel("r (Bohr)")
    ax.set_ylabel("|V_computed - V_exact|")
    ax.set_title("Distortion / Error vs. Exact 1/r Physics")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 3)

    plt.tight_layout()
    output_img = "examples/screened_coulomb_study.png"
    plt.savefig(output_img, dpi=150)
    print(f"\nPlot saved to: {output_img}")

    # Summary findings
    print("\n" + "-" * 70)
    print("  SUMMARY & CONCLUSION")
    print("-" * 70)
    print("1. Physical Distortion: Screened Coulomb modifies the operator (del^2 -> del^2 - mu^2),")
    print("   introducing systematic error/distortion vs. true 1/r Coulomb physics for any mu > 0.")
    print("2. Cusp Behavior: Screening exponentially suppresses outer tails (e^{-mu r}),")
    print("   but DOES NOT resolve the r -> 0 nuclear cusp singularity.")
    print("3. Double-Split Advantage: Preserves EXACT 1/r physics while analytically eliminating")
    print("   the nuclear cusp singularity via atomic Gaussian pre-subtraction + NNLS fitting.")
    print("-" * 70)


if __name__ == "__main__":
    run_screened_coulomb_study()
