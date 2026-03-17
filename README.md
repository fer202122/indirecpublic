# Code for Characterization of Multi-Revolution Circular-to-Circular Solar-Sail Transfers Around Planets

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
* [Example](#example)
* [Command-line helper](#command-line-helper)
* [Project structure](#project-structure)
* [API reference](#api-reference)
* [Authors](#authors)
* [License](#license)
* [References](#references)
* [Cite this repository](#cite-this-repository)
* [Would you like to contribute?](#would-you-like-to-contribute?)

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

## Example

Run from repo root after install:

```bash
python examples/01_true_anomaly_demo.py
```

Each script:

* builds a problem instance,
* solves with a basic initial guess,
* prints convergence info, and
* returns ECI vectors on a grid.

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
│  └─ 01_true_anomaly_demo.py
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
* `direction` — Ascending (1) or descending (-1)

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

## Authors

This software was developed by:  
- *Oskar Miller* ([@oskarmiller](https://github.com/oskarmiller)), Oskar.Miller2001@gmail.com, Technische Universiteit Delft   
- *Fernando Gámez Losada* ([@fer202122](https://github.com/fer202122)), ![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0009-0007-6107-8484](https://orcid.org/0009-0007-6107-8484), F.GamezLosada@tudelft.nl, Technische Universiteit Delft   
  

---

## License

All source code files available in this repository are licensed under a MIT license (see `./LICENSE`).

Copyright notice:

Technische Universiteit Delft hereby disclaims all copyright interest in the program "Characterization of Multi-Revolution Circular-to-Circular
Solar-Sail Transfers Around Planets" written by the Author(s). 
Henri Werij, Faculty of Aerospace Engineering, Technische Universiteit Delft.

© 2026, O. Miller, F. Gámez Losada

---

## References  

- [Characterization of Multi-Revolution Circular-to-Circular
Solar-Sail Transfers Around Planets](TODO BLA: Paste DOI paper here)  

---

## Cite this repository

**How to cite this repository:** O. Miller, F. Gámez Losada, 2026, Code for Characterization of Multi-Revolution Circular-to-Circular
Solar-Sail Transfers Around Planets. 4TU.ResearchData. Software. TODO BLA: Paste software DOI here

---

## Would you like to contribute?

You are welcome to contribute! If you have any comments, feedback, or recommendations, feel free to reach out to the authors.  

If you want to contribute directly, you are welcome to open an issue and fork this repository.
