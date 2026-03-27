# SPDX-FileCopyrightText: 2026 Oskar Miller <Oskar.Miller2001@gmail.com>
# SPDX-FileCopyrightText: 2026 Fernando Gamez Losada <F.GamezLosada@tudelft.nl>
# SPDX-License-Identifier: MIT
import argparse
import numpy as np
from sailtransfer import TransferProblem_TrueAnomaly, TransferProblem_Time

def parse_args():
    p = argparse.ArgumentParser(description="Solve solar-sail transfer and dump ECI CSVs.")
    p.add_argument("--mode", choices=["true_anomaly", "time"], default="true_anomaly")
    p.add_argument("--r0", type=float, required=True, help="Initial radius [m]")
    p.add_argument("--mu", type=float, required=True, help="Gravitational parameter [m^3/s^2]")
    p.add_argument("--omega", type=float, required=True, help="Central body rotation rate [rad/s]")
    p.add_argument("--a0", type=float, required=True, help="Characteristic acceleration [m/s^2]")
    p.add_argument("--raan", type=float, required=True, help="Right ascension of ascending node [deg]")
    p.add_argument("--inc", type=float, required=True, help="Inclination [deg]")
    p.add_argument("--arglat", type=float, required=True, help="Argument of latitude [deg]")
    p.add_argument("--theta_f", type=float, default=1.0, help="Final true anomaly, number of revolutions [-]")
    p.add_argument("--time_f", type=float, default=1.0, help="Final time [days]")
    p.add_argument("--direction", type=float, default=1.0, help="Direction of transfer (1 for ascending, -1 for descending)")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--outprefix", type=str, default="eci")
    return p.parse_args()

def main():
    a = parse_args()
    if a.mode == "true_anomaly":
        prob = TransferProblem_TrueAnomaly(a.r0, a.mu, a.omega, a.a0, np.radians(a.raan), np.radians(a.inc), np.radians(a.arglat), a.theta_f * 2.0 * np.pi, a.direction)
        grid = np.linspace(0.0, 1.0, a.samples)
    else:
        prob = TransferProblem_Time(a.r0, a.mu, a.omega, a.a0, np.radians(a.raan), np.radians(a.inc), np.radians(a.arglat), a.time_f * 24.0 * 3600.0,a.direction)
        grid = np.linspace(0.0, 1.0, a.samples)

    s0 = np.full(6, -2.0)
    print(f"  Initial guess: {s0}")
    sol, res, nfev, ok = prob.optimize_fsolve(s0)
    print(f"solve ok={ok}, ||res||={res:.3e}, nfev={nfev}")

    pos, vel, acc = prob.eci_vectors(sol, grid)

    # make acc to unit vector
    acc /= np.linalg.norm(acc, axis=1, keepdims=True)

    # Dump to CSVs (see paper for frame definitions):
    # - x, y, z: Position in Ecliptic Frame (E) frame [m]
    # - vx, vy, vz: Velocity in Ecliptic Frame (E) frame [m/s]
    # - nx, ny, nz: Components of normal unit vector in Ecliptic Frame (E) frame [-]
    np.savetxt(f"./output/{a.outprefix}_pos.csv", np.asarray(pos), delimiter=",", header="x,y,z", comments="")
    np.savetxt(f"./output/{a.outprefix}_vel.csv", np.asarray(vel), delimiter=",", header="vx,vy,vz", comments="")
    np.savetxt(f"./output/{a.outprefix}_acc.csv", np.asarray(acc), delimiter=",", header="nx,ny,nz", comments="")
    print("Wrote CSVs with ECI vectors.")

if __name__ == "__main__":
    main()
