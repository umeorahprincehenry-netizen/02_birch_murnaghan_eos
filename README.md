**README 1 (for 01_quantum_harmonic_oscillator/README.md)**

1D Time-Independent Schrodinger Equation Solver and Convergence Benchmark

A self-contained numerical solver and verification suite for the one-dimensional quantum harmonic oscillator in dimensionless units (hbar = m = omega = 1).

Physical Model and Discretization

This module solves the stationary Schrodinger eigenvalue equation:
(-1/2 d²/dx² + 1/2 x²) psi_n(x) = E_n psi_n(x)

The equation is solved on a truncated symmetric domain [-L, L] with L = 8.0 and homogeneous Dirichlet boundary conditions psi(-L) = psi(L) = 0. Using a second-order central finite-difference stencil on N interior nodes separated by step size h = 2L / (N + 1), the Hamiltonian reduces to a real symmetric tridiagonal matrix with:

* Main diagonal: d_i = (1 / h²) + 0.5 x_i²
* Sub-diagonal and super-diagonal: e_i = -0.5 / h²

The lowest k bound eigenvalues and eigenstates are extracted in O(N k) operations using scipy.linalg.eigh_tridiagonal.

Analytical Verification Checks

Running schrodinger_1d_solver.py executes three automated scientific validation checks:

1. Eigenvalue Accuracy: Compares numerical eigenvalues for states n = 0 to 5 (at N = 800) against exact analytical values E_n = n + 0.5.
2. Eigenstate Orthonormality: Verifies that the spatial integral of psi_m(x) psi_n(x) dx equals the Kronecker delta to machine precision (maximum deviation below 10^-12).
3. Asymptotic Grid Convergence: Halves the grid spacing h across N = 200, 400, 800, and 1600 and verifies that the ground-state truncation error decays at the theoretical second-order rate O(h²), with an empirical convergence order near 2.00.

How to Run

Install dependencies:
pip install numpy scipy matplotlib

Run the solver and validation suite:
python schrodinger_1d_solver.py

---

**README 2 (for 02_birch_murnaghan_eos/README.md)**

Third-Order Birch-Murnaghan Equation of State (EOS) Fitter and Validator

A Python implementation for fitting total energy versus volume E(V) curves from first-principles solid-state calculations to the third-order Birch-Murnaghan isothermal equation of state, complete with analytical thermodynamic derivatives and covariance uncertainty estimation.

Physical Formulation

Defining the Eulerian finite volumetric strain as f = 0.5 ((V0 / V)^(2/3) - 1), the third-order Birch-Murnaghan energy-volume relation is:
E(V) = E0 + (9/2) V0 B0 f² (1 + (B0' - 4) f)

Here V0 is the equilibrium unit-cell volume, E0 is the equilibrium total energy, B0 is the equilibrium isothermal bulk modulus, and B0' is its first pressure derivative at zero pressure.

Differentiating E(V) with respect to volume yields the exact isothermal pressure P(V) and volume-dependent bulk modulus B(V):
P(V) = -dE/dV = 3 B0 f (1 + 2f)^(5/2) (1 + 1.5 (B0' - 4) f)
B(V) = -V dP/dV = B0 (1 + 2f)^(5/2) (5f (1 + 1.5 (B0' - 4) f) + (1 + 2f) (1 + 3 (B0' - 4) f))

Automated Verification Suite

Running birch_murnaghan_eos.py executes three validation tests on a diamond-cubic silicon primitive-cell benchmark:

1. Exact Parameter Recovery: Verifies nonlinear least-squares convergence to machine precision on noiseless E(V) curves.
2. Thermodynamic Derivative Consistency: Cross-checks analytical P(V) and B(V) expressions against fourth-order central finite-difference derivatives of E(V).
3. Noisy DFT Regression: Benchmarks parameter recovery and 1-sigma covariance uncertainties under realistic 0.25 meV plane-wave convergence noise.

How to Run

Install dependencies:
pip install numpy scipy matplotlib

Run the fitter and validation suite:
python birch_murnaghan_eos.py
