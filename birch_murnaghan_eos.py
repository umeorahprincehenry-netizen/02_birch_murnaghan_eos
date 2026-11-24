#!/usr/bin/env python3
"""
Third-Order Birch-Murnaghan Equation of State (EOS) Fitter & Validator.

Performs nonlinear least-squares regression of total energy vs. volume E(V)
curves to extract equilibrium volume (V0), equilibrium energy (E0), isothermal
bulk modulus (B0), and its pressure derivative (B0') with asymptotic covariance
uncertainties, and validates analytical P(V) and B(V) derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

# Exact CODATA unit conversion: 1 eV / Angstrom^3 in GPa
EV_PER_A3_TO_GPA: float = 160.2176634


@dataclass
class EOSFitResult:
    """Container for fitted Birch-Murnaghan EOS parameters and diagnostics."""

    v0: float
    e0: float
    b0_gpa: float
    b0_prime: float
    v0_err: float
    e0_err: float
    b0_gpa_err: float
    b0_prime_err: float
    rmse_mev: float
    r_squared: float
    volumes: np.ndarray
    energies: np.ndarray
    fitted_energies: np.ndarray
    residuals_mev: np.ndarray


def _eulerian_strain(volume: np.ndarray, v0: float) -> np.ndarray:
    """Compute Eulerian finite volumetric strain f = 0.5 * [(V0 / V)^(2/3) - 1]."""
    return 0.5 * ((v0 / volume) ** (2.0 / 3.0) - 1.0)


def birch_murnaghan_energy(
    volume: np.ndarray,
    v0: float,
    e0: float,
    b0_gpa: float,
    b0_prime: float,
) -> np.ndarray:
    """
    Evaluate third-order Birch-Murnaghan internal energy E(V) in eV.

    Parameters
    ----------
    volume : np.ndarray
        Unit-cell volumes in Angstrom^3.
    v0 : float
        Equilibrium volume in Angstrom^3.
    e0 : float
        Equilibrium total energy in eV.
    b0_gpa : float
        Equilibrium isothermal bulk modulus in GPa.
    b0_prime : float
        First pressure derivative of the bulk modulus (dimensionless).
    """
    vol = np.asarray(volume, dtype=float)
    f = _eulerian_strain(vol, v0)
    b0_ev_a3 = b0_gpa / EV_PER_A3_TO_GPA
    return e0 + 4.5 * v0 * b0_ev_a3 * (f**2) * (1.0 + (b0_prime - 4.0) * f)


def birch_murnaghan_pressure(
    volume: np.ndarray,
    v0: float,
    b0_gpa: float,
    b0_prime: float,
) -> np.ndarray:
    """
    Evaluate analytical isothermal pressure P(V) = -dE/dV in GPa.
    """
    vol = np.asarray(volume, dtype=float)
    f = _eulerian_strain(vol, v0)
    return (
        3.0
        * b0_gpa
        * f
        * ((1.0 + 2.0 * f) ** 2.5)
        * (1.0 + 1.5 * (b0_prime - 4.0) * f)
    )


def birch_murnaghan_bulk_modulus(
    volume: np.ndarray,
    v0: float,
    b0_gpa: float,
    b0_prime: float,
) -> np.ndarray:
    """
    Evaluate analytical isothermal bulk modulus B(V) = -V * dP/dV in GPa.
    """
    vol = np.asarray(volume, dtype=float)
    f = _eulerian_strain(vol, v0)
    coeff = b0_prime - 4.0
    term_1 = 5.0 * f * (1.0 + 1.5 * coeff * f)
    term_2 = (1.0 + 2.0 * f) * (1.0 + 3.0 * coeff * f)
    return b0_gpa * ((1.0 + 2.0 * f) ** 2.5) * (term_1 + term_2)


def fit_birch_murnaghan(volumes: np.ndarray, energies: np.ndarray) -> EOSFitResult:
    """
    Fit E(V) data to the third-order Birch-Murnaghan equation of state.
    """
    vol = np.asarray(volumes, dtype=float)
    eng = np.asarray(energies, dtype=float)

    if vol.ndim != 1 or eng.ndim != 1 or vol.shape != eng.shape:
        raise ValueError("volumes and energies must be 1D arrays of identical length.")
    if vol.size < 5:
        raise ValueError("At least 5 (V, E) points are required for a 4-parameter EOS fit.")
    if np.any(vol <= 0.0):
        raise ValueError("All unit-cell volumes must be strictly positive.")

    # Estimate initial parameters via local quadratic curvature around E_min
    min_idx = int(np.argmin(eng))
    v0_guess = float(vol[min_idx])
    e0_guess = float(eng[min_idx])
    quad_coeffs = np.polyfit(vol, eng, deg=2)
    b0_guess = max(float(v0_guess * 2.0 * quad_coeffs[0] * EV_PER_A3_TO_GPA), 10.0)
    p0 = (v0_guess, e0_guess, b0_guess, 4.0)

    lower_bounds = (0.5 * float(np.min(vol)), -np.inf, 1e-3, 1.0)
    upper_bounds = (1.5 * float(np.max(vol)), np.inf, 2000.0, 15.0)

    popt, pcov = curve_fit(
        birch_murnaghan_energy,
        vol,
        eng,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=10000,
    )
    perr = np.sqrt(np.diag(pcov))

    fitted = birch_murnaghan_energy(vol, *popt)
    residuals = eng - fitted
    residuals_mev = residuals * 1000.0
    rmse_mev = float(np.sqrt(np.mean(residuals_mev**2)))

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((eng - np.mean(eng)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot)

    return EOSFitResult(
        v0=float(popt[0]),
        e0=float(popt[1]),
        b0_gpa=float(popt[2]),
        b0_prime=float(popt[3]),
        v0_err=float(perr[0]),
        e0_err=float(perr[1]),
        b0_gpa_err=float(perr[2]),
        b0_prime_err=float(perr[3]),
        rmse_mev=rmse_mev,
        r_squared=r_squared,
        volumes=vol,
        energies=eng,
        fitted_energies=fitted,
        residuals_mev=residuals_mev,
    )


def save_eos_plot(fit: EOSFitResult, output_path: Path) -> bool:
    """
    Plot fitted E(V) curve and derived P(V) compression curve if matplotlib is available.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    v_dense = np.linspace(float(np.min(fit.volumes)), float(np.max(fit.volumes)), 300)
    e_dense = birch_murnaghan_energy(v_dense, fit.v0, fit.e0, fit.b0_gpa, fit.b0_prime)
    p_dense = birch_murnaghan_pressure(v_dense, fit.v0, fit.b0_gpa, fit.b0_prime)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # Panel 1: Energy vs Volume
    ax1.scatter(fit.volumes, fit.energies, color="#1f77b4", zorder=3, label="DFT Data Points")
    ax1.plot(v_dense, e_dense, color="#d62728", linewidth=1.8, label="3rd-Order BM Fit")
    ax1.scatter([fit.v0], [fit.e0], color="black", marker="*", s=110, zorder=4, label=f"$V_0 = {fit.v0:.2f}\\ \\AA^3$")
    ax1.set_title("Equation of State: $E(V)$")
    ax1.set_xlabel(r"Primitive Cell Volume $V$ ($\mathrm{\AA}^3$)")
    ax1.set_ylabel("Total Energy $E$ (eV)")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, alpha=0.25)

    # Panel 2: Pressure vs Volume
    ax2.plot(v_dense, p_dense, color="#2ca02c", linewidth=1.8, label="$P(V) = -\\partial E / \\partial V$")
    ax2.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
    ax2.axvline(fit.v0, color="gray", linestyle=":", linewidth=0.9)
    ax2.set_title(f"Compression Curve ($B_0 = {fit.b0_gpa:.1f}$ GPa, $B_0' = {fit.b0_prime:.2f}$)")
    ax2.set_xlabel(r"Primitive Cell Volume $V$ ($\mathrm{\AA}^3$)")
    ax2.set_ylabel("Isothermal Pressure $P$ (GPa)")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def run_validation_suite() -> None:
    """Execute exact recovery, derivative consistency, and noisy DFT benchmark checks."""
    # Reference parameters for diamond-cubic Silicon (2-atom primitive cell)
    v0_true = 40.02
    e0_true = -10.8420
    b0_true = 97.80
    bp_true = 4.21

    volumes = np.linspace(35.2, 44.8, 15)
    exact_energies = birch_murnaghan_energy(volumes, v0_true, e0_true, b0_true, bp_true)

    # Check 1: Exact parameter recovery on noiseless E(V) data
    clean_fit = fit_birch_murnaghan(volumes, exact_energies)
    assert abs(clean_fit.v0 - v0_true) < 1e-10, f"V0 recovery failed: {clean_fit.v0}"
    assert abs(clean_fit.e0 - e0_true) < 1e-10, f"E0 recovery failed: {clean_fit.e0}"
    assert abs(clean_fit.b0_gpa - b0_true) < 1e-9, f"B0 recovery failed: {clean_fit.b0_gpa}"
    assert abs(clean_fit.b0_prime - bp_true) < 1e-9, f"B0' recovery failed: {clean_fit.b0_prime}"

    # Check 2: Analytical P(V) and B(V) vs. 4th-order central finite differences
    v_probe = np.linspace(36.0, 44.0, 9)
    dv = 1e-3
    e_m2 = birch_murnaghan_energy(v_probe - 2.0 * dv, v0_true, e0_true, b0_true, bp_true)
    e_m1 = birch_murnaghan_energy(v_probe - dv, v0_true, e0_true, b0_true, bp_true)
    e_0 = birch_murnaghan_energy(v_probe, v0_true, e0_true, b0_true, bp_true)
    e_p1 = birch_murnaghan_energy(v_probe + dv, v0_true, e0_true, b0_true, bp_true)
    e_p2 = birch_murnaghan_energy(v_probe + 2.0 * dv, v0_true, e0_true, b0_true, bp_true)

    de_dv = (-e_p2 + 8.0 * e_p1 - 8.0 * e_m1 + e_m2) / (12.0 * dv)
    d2e_dv2 = (-e_p2 + 16.0 * e_p1 - 30.0 * e_0 + 16.0 * e_m1 - e_m2) / (12.0 * dv**2)

    p_num = -de_dv * EV_PER_A3_TO_GPA
    b_num = v_probe * d2e_dv2 * EV_PER_A3_TO_GPA

    p_ana = birch_murnaghan_pressure(v_probe, v0_true, b0_true, bp_true)
    b_ana = birch_murnaghan_bulk_modulus(v_probe, v0_true, b0_true, bp_true)

    max_p_err = float(np.max(np.abs(p_ana - p_num)))
    max_b_err = float(np.max(np.abs(b_ana - b_num)))
    assert max_p_err < 1e-7, f"Pressure derivative mismatch: {max_p_err}"
    assert max_b_err < 1e-3, f"Bulk modulus derivative mismatch: {max_b_err}"

    # Check 3: Realistic DFT benchmark with 0.25 meV Gaussian numerical noise
    rng = np.random.default_rng(42)
    noisy_energies = exact_energies + rng.normal(loc=0.0, scale=2.5e-4, size=volumes.size)
    noisy_fit = fit_birch_murnaghan(volumes, noisy_energies)

    v0_rel_err = abs(noisy_fit.v0 - v0_true) / v0_true
    b0_rel_err = abs(noisy_fit.b0_gpa - b0_true) / b0_true
    assert v0_rel_err < 1e-3, f"Noisy V0 relative error too high: {v0_rel_err:.4%}"
    assert b0_rel_err < 1e-2, f"Noisy B0 relative error too high: {b0_rel_err:.4%}"

    print("=== Third-Order Birch-Murnaghan EOS Fit (Diamond-Cubic Silicon Benchmark) ===")
    print(f"{'Parameter':<14} {'Ground Truth':<16} {'Fitted Value':<16} {'1-Sigma Unc.':<14}")
    print("-" * 62)
    print(f"{'V0 (A^3)':<14} {v0_true:<16.6f} {noisy_fit.v0:<16.6f} +/- {noisy_fit.v0_err:<10.6f}")
    print(f"{'E0 (eV)':<14} {e0_true:<16.6f} {noisy_fit.e0:<16.6f} +/- {noisy_fit.e0_err:<10.6f}")
    print(f"{'B0 (GPa)':<14} {b0_true:<16.4f} {noisy_fit.b0_gpa:<16.4f} +/- {noisy_fit.b0_gpa_err:<10.4f}")
    print(f"{'B0_prime':<14} {bp_true:<16.4f} {noisy_fit.b0_prime:<16.4f} +/- {noisy_fit.b0_prime_err:<10.4f}")
    print("-" * 62)
    print(f"Fit Quality:   R^2 = {noisy_fit.r_squared:.8f} | RMSE = {noisy_fit.rmse_mev:.4f} meV")

    plot_file = Path(__file__).resolve().parent / "birch_murnaghan_eos.png"
    plot_saved = save_eos_plot(noisy_fit, plot_file)

    print("\n=== Validation Summary ===")
    print(f"[PASS] Noiseless B0 recovery error:     {abs(clean_fit.b0_gpa - b0_true):.3e} GPa")
    print(f"[PASS] Analytical vs. numerical dE/dV:  {max_p_err:.3e} GPa")
    print(f"[PASS] Analytical vs. numerical B(V):   {max_b_err:.3e} GPa")
    print(f"[PASS] Noisy DFT B0 relative accuracy:  {(1.0 - b0_rel_err) * 100.0:.3f}%")
    if plot_saved:
        print(f"[INFO] EOS figure saved to:             {plot_file.name}")
    else:
        print("[INFO] matplotlib not installed; skipped figure generation.")


if __name__ == "__main__":
    run_validation_suite()
