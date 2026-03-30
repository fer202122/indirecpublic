# SPDX-FileCopyrightText: 2026 Oskar Miller <Oskar.Miller2001@gmail.com>
# SPDX-FileCopyrightText: 2026 Fernando Gamez Losada <F.GamezLosada@tudelft.nl>
# SPDX-License-Identifier: MIT
import argparse
import numpy as np
import time
from sailtransfer import TransferProblem_TrueAnomaly, TransferProblem_Time, TransferProblem_TrueAnomaly_Optical

def parse_args():
    p = argparse.ArgumentParser(description="Solve solar-sail transfer and dump ECI CSVs.")
    p.add_argument("--mode", choices=["true_anomaly", "true_anomaly_optical", "time"], default="true_anomaly")
    p.add_argument("--r0", type=float, required=True, help="Initial radius [m]")
    p.add_argument("--mu", type=float, required=True, help="Gravitational parameter [m^3/s^2]")
    p.add_argument("--omega", type=float, required=True, help="Central body rotation rate [rad/s]")
    p.add_argument("--a0", type=float, required=True, help="Characteristic acceleration [m/s^2]")
    p.add_argument("--raan", type=float, required=True, help="Right ascension of ascending node [deg]")
    p.add_argument("--inc", type=float, required=True, help="Inclination [deg]")
    p.add_argument("--arglat", type=float, required=True, help="Argument of latitude [deg]")
    p.add_argument("--revs", type=float, default=1.0, help="Number of revolutions [-]")
    p.add_argument("--time_f", type=float, default=1.0, help="Final time [days]")
    p.add_argument("--direction", type=float, default=1.0, help="Direction of transfer (1 for ascending, -1 for descending)")
    p.add_argument("--C1", type=float, default=2.0, help="C1 parameter for the optical model (only used in true_anomaly_optical mode)")
    p.add_argument("--C2", type=float, default=0.0, help="C2 parameter for the optical model (only used in true_anomaly_optical mode)")
    p.add_argument("--C3", type=float, default=0.0, help="C3 parameter for the optical model (only used in true_anomaly_optical mode)")
    p.add_argument("--samples", type=int, default=40, help="Number of samples per revolution for output CSVs")
    p.add_argument("--frame", type=str, default="ecliptic", help="Output reference frame for the vectors in the CSVs")
    return p.parse_args()

def main():
    a = parse_args()
    if a.mode == "true_anomaly":
        prob = TransferProblem_TrueAnomaly(a.r0, a.mu, a.omega, a.a0, np.radians(a.raan), np.radians(a.inc), np.radians(a.arglat), a.revs * 2.0 * np.pi, a.direction)
        grid = np.linspace(0.0, 1.0, a.samples*int(a.revs) + 1)
    elif a.mode == "true_anomaly_optical":
        prob = TransferProblem_TrueAnomaly_Optical(a.r0, a.mu, a.omega, a.a0, np.radians(a.raan), np.radians(a.inc), np.radians(a.arglat), a.revs * 2.0 * np.pi, a.direction, a.C1, a.C2, a.C3)
        grid = np.linspace(0.0, 1.0, a.samples*int(a.revs) + 1)
    elif a.mode == "time":
        prob = TransferProblem_Time(a.r0, a.mu, a.omega, a.a0, np.radians(a.raan), np.radians(a.inc), np.radians(a.arglat), a.time_f * 24.0 * 3600.0,a.direction)
        grid = np.linspace(0.0, 1.0, a.samples*int(a.revs) + 1)

    # Initial guess: [lambda_r0, lambda_t0, lambda_u0, lambda_v0, nu1, nu2]
    s0 = np.full((6,), -2.0)

    print(f"  Initial guess: {s0}")

    # Optimize with fsolve
    start_time = time.time()
    sol, res_norm, nfev, ok = prob.optimize_fsolve(s0, xtol_fsolve=1e-12)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"  fsolve success={ok}, ||res||={res_norm:.3e}, evals={nfev}, time={runtime:.3e}")

    # Check for convergence
    if not ok and res_norm > 1e-9:
        raise RuntimeError(f"fsolve failed with res_norm={res_norm:.3e}, evals={nfev}")

    # Dump to CSVs (see paper for frame definitions):
    if a.frame == "ecliptic":
        print("Output frame: Ecliptic")
        # Get position, velocity, and acceleration vectors in Ecliptic Frame (E) from solution
        pos, vel, acc = prob.ecliptic_vectors(sol, grid)

        # make acc to unit vector
        acc /= np.linalg.norm(acc, axis=1, keepdims=True)

        # CSV columns: 
        # - x, y, z: Position in Ecliptic Frame (E) frame [m]
        # - vx, vy, vz: Velocity in Ecliptic Frame (E) frame [m/s]
        # - nx, ny, nz: Components of normal unit vector in Ecliptic Frame (E) frame [-]
        np.savetxt(f"./output/{a.frame}_pos.csv", np.asarray(pos), delimiter=",", header="x,y,z", comments="")
        np.savetxt(f"./output/{a.frame}_vel.csv", np.asarray(vel), delimiter=",", header="vx,vy,vz", comments="")
        np.savetxt(f"./output/{a.frame}_normal.csv", np.asarray(acc), delimiter=",", header="nx,ny,nz", comments="")

    elif a.frame == "orbital":
        print("Output frame: Orbital")
        # Retrieve scaled time
        sol_integrated = np.array(prob.integrate(sol, grid))
        t_bar = sol_integrated[:, 1]

        # Obtain position, velocity, control profiles from solution
        pos_c, vel_c = np.array(prob.orbital_plane_vectors(sol, grid))
        n_s = np.array(prob.control_unit_slf(sol, grid))

        # Rotate velocity costate to Orbital Frame
        lam_c = np.zeros((len(grid), 5))
        lam_u = sol_integrated[:, 6]
        lam_v = sol_integrated[:, 7]
        theta = grid * 2 * np.pi * a.revs + a.arglat
        lam_vx = lam_u * np.cos(theta) - lam_v * np.sin(theta)
        lam_vy = lam_u * np.sin(theta) + lam_v * np.cos(theta)
        lam_c[:, 2:4] = np.hstack((lam_vx[:, None], lam_vy[:, None]))

        # Dump to csv file for plotting:
        # - grid: Normalized true anomaly (0 to 1) [-]
        # - x_c, y_c: Normalized position in Orbital Frame (I) [-]
        # - vx_c, vy_c: Normalized velocity in Orbital Frame (I) [-]
        # - nx_s, ny_s, nz_s: Components of normal unit vector in Sunlight Frame (S) [-]
        # - lam_x, lam_y, lam_vx, lam_vy, lam_t_bar: Costate components in Orbital Frame (I) (I) [-]
        # - t_bar: Normalized time (order of magnitude of one)
        data = np.hstack((grid[:, None], pos_c, vel_c, t_bar[:, None], n_s, lam_c))
        header = "grid,x_c,y_c,vx_c,vy_c,t_bar,nx_s,ny_s,nz_s,lam_x,lam_y,lam_vx,lam_vy,lam_t_bar"
        np.savetxt("./output/indirect_output.csv", data, delimiter=",", header=header, comments='')

        data = np.hstack((res_norm, nfev, ok, runtime))
        header = "res_norm,nfev,ok,runtime"
        np.savetxt("./output/indirect_output_conv.csv", data, delimiter=",", header=header, comments='')

        data = np.atleast_2d(sol)
        header = "lam_r0,lam_t0,lam_u0,lam_v0,nu1,nu2"
        np.savetxt("./output/indirect_output_guess.csv", data, delimiter=",", header=header, comments='')


if __name__ == "__main__":
    main()
