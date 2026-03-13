# Solar-Sail Transfer (Indirect Optimal Control)

> JAX-based reference implementation for **planar planetocentric circular-to-circular solar-sail transfers** solved via **indirect optimal control** (single shooting).
> Two formulations are provided:
>
> * `TransferProblem_TrueAnomaly` — integrate with **normalized true anomaly** as the independent variable.
> * `TransferProblem_Time` — integrate with **normalized time** as the independent variable.

Both classes expose:

* `optimize_fsolve(...)` and `optimize_jaxopt(...)` to solve for the optimal initial costates and multipliers.
* `integrate(...)` to sample the optimal state+costate along the trajectory.
* `eci_vectors(...)` to return **ECI position**, **ECI velocity**, and **ECI solar-sail acceleration** at requested samples.

---

## Contents

* [Features](#features)
* [Install](#install)
* [Quickstart](#quickstart)
* [Examples](#examples)
* [Command-line helper](#command-line-helper)
* [Project structure](#project-structure)
* [API reference](#api-reference)
* [Troubleshooting](#troubleshooting)
* [Performance tips](#performance-tips)
* [License](#license)

---

## Features

* Indirect OCP with **single shooting** and tight terminal constraints (circularization).
* Thrust direction from **primer vector**; sail law uses cone/clock angles in a **sun-line frame**.
* Clean frame handling (RTN → ORB → ECI → SLF) with robust numerics (clamps/epsilons).
* High-accuracy integration via **diffrax** (Dopri8 + PID controller).
* Pure JAX math for speed and jit-ability.
* Convenience function `eci_vectors(...)` to get **(x, y, z)**, **(vx, vy, vz)**, **(nx, ny, nz)** in **ECI**.

---

## Install

> Requires Python **3.10+**.

```bash
# optional: create a clean environment
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

python -m pip install -U pip

# clone and install in editable mode
git clone https://github.com/oskarmiller/OskarMiller_MScThesis.git
cd solar-sail-transfer
python -m pip install -e .
```

---

## Quickstart

```python
import numpy as np
from sailtransfer import TransferProblem_TrueAnomaly

# Earth-like parameters (SI)
mu = 3.986004418e14                 # gravitational parameter [m^3/s^2]
omega_body = 7.2921159e-5           # rotation rate [rad/s]
r0 = 42164e3                         # initial radius [m]
a0 = 1e-3                           # characteristic acceleration [m/s^2]
raan = 0.0
inc = 0.0
arglat = 0.0
theta_f = 2.0 * np.pi               # final physical true anomaly span [rad]

# Build problem (true-anomaly formulation)
prob = TransferProblem_TrueAnomaly(
    r_0=r0,
    mu_central_body=mu,
    rotational_speed_central_body=omega_body,
    a_0=a0,
    raan=raan,
    inclination=inc,
    arg_of_latitude=arglat,
    theta_f=theta_f
)

# Initial guess: [lambda_r0, lambda_t0, lambda_u0, lambda_v0, nu1, nu2]
s0 = np.full(6, -2.0)

# Solve (SciPy fsolve wrapped around a JAX residual)
sol_params, res_norm, nfev, ok = prob.optimize_fsolve(s0, xtol_fsolve=1e-10)
print(f"converged={ok}, ||res||={res_norm:.3e}, evals={nfev}")

# Sample trajectory and get ECI vectors
theta_bar = np.linspace(0.0, 1.0, 200)    # normalized samples
pos_eci, vel_eci, acc_eci = prob.eci_vectors(sol_params, theta_bar)
print(pos_eci.shape, vel_eci.shape, acc_eci.shape)  # (200, 3) each
```

For the **time formulation**, replace the class and pass `time_f`:

```python
from sailtransfer import TransferProblem_Time

time_f = 24.0 * 3600.0  # final physical time span [s], example
prob = TransferProblem_Time(r0, mu, omega_body, a0, raan, inc, arglat, time_f)
s0 = np.zeros(6)
sol_params, res_norm, nfev, ok = prob.optimize_fsolve(s0)
tbar = np.linspace(0.0, 1.0, 200)
pos_eci, vel_eci, acc_eci = prob.eci_vectors(sol_params, tbar)
```

---

## Examples

Run from repo root after install:

```bash
python examples/01_true_anomaly_demo.py
python examples/02_time_formulation_demo.py
```

Each script:

* builds a problem instance,
* solves with a basic initial guess,
* prints convergence info, and
* returns ECI vectors on a grid.

> The dummy guess may not converge for all parameters; adjust as needed.

---

## Command-line helper

A tiny CLI is included to solve and dump CSVs:

```bash
python scripts/solve_and_dump.py \
  --mode true_anomaly \
  --r0 42164e3 \
  --mu 3.986004418e14 \
  --omega 7.2921159e-5 \
  --a0 0.001 \
  --samples 200 \
  --outprefix eci
```

Outputs:

* `eci_pos.csv` with columns `x,y,z` (meters),
* `eci_vel.csv` with columns `vx,vy,vz` (meters per second),
* `eci_acc.csv` with columns `nx,ny,nz` (meters per second squared).

---

## Project structure

```
solar-sail-transfer/
├─ README.md
├─ LICENSE
├─ pyproject.toml
├─ .gitignore
├─ sailtransfer/
│  ├─ __init__.py
│  └─ problems.py          # TransferProblem_TrueAnomaly, TransferProblem_Time
├─ examples/
│  ├─ 01_true_anomaly_demo.py
│  └─ 02_time_formulation_demo.py
└─ scripts/
   └─ solve_and_dump.py
```

* **sailtransfer/problems.py**: your thesis code with both formulations and `eci_vectors(...)`.
* **examples/**: runnable scripts illustrating both modes.
* **scripts/solve_and_dump.py**: quick CSV export for plotting.
* **tests/**: smoke tests on shapes (does not require convergence).

---

## API reference

### `TransferProblem_TrueAnomaly(...)`

**Constructor arguments (SI unless stated):**

* `r_0` — initial radius [m]
* `mu_central_body` — gravitational parameter [m^3/s^2]
* `rotational_speed_central_body` — rotation rate [rad/s]
* `a_0` — characteristic acceleration [m/s^2]
* `raan` — right ascension of ascending node [rad]
* `inclination` — inclination [rad]
* `arg_of_latitude` — argument of latitude at start [rad]
* `theta_f` — final physical true anomaly span [rad] (trajectory integrates normalized `theta_bar` from 0 to 1)

**Methods:**

* `optimize_fsolve(s_initial_guess_np, xtol_fsolve=1e-10)`
  Returns `(solution_params, residual_norm, num_evals, success)`.
* `optimize_jaxopt(s_initial_guess_np, tol_jaxopt=1e-10)`
  JAX-only root finding. Returns `(params, residual_norm, num_evals, success)`.
* `integrate(s_solution, theta_bar_points)`
  Returns stacked state+costate at samples. Shape `(N, 8)`.
* `eci_vectors(s_solution, theta_bar_points)`
  Returns `(pos_eci, vel_eci, acc_eci)` each `(N, 3)`.

### `TransferProblem_Time(...)`

Same as above, but with:

* `time_f` — final physical time span [s] (trajectory integrates normalized `time_bar` from 0 to 1).

**Methods** mirror the true-anomaly class with `time_bar_points`.

---

## License

MIT — see [LICENSE](./LICENSE).

---

**Attribution**
This code originates from a master’s thesis implementation by Oskar Miller. If you use it in a publication, please cite the repository and the thesis accordingly.
