# SPDX-FileCopyrightText: 2026 Oskar Miller <Oskar.Miller2001@gmail.com>
# SPDX-FileCopyrightText: 2026 Fernando Gamez Losada <F.GamezLosada@tudelft.nl>
# SPDX-License-Identifier: MIT
import numpy as np
import jax.numpy as jnp
from sailtransfer import TransferProblem_TrueAnomaly_Optical
import time

def main():
    # Read problem parameters from the matlab-generated csv file
    params = np.loadtxt("indirect_input.csv", delimiter=",")
    # params = np.loadtxt("/Users/fernando/workspace/gpops2cases/solarSail2D/cranked_cart_maxr/normal_comps_sm_suns_opt/work/indirect_input.csv", delimiter=",")
    mu = params[0]
    omega_body = params[1]
    r0 = params[2]
    a0 = params[3]
    raan = params[4]
    inc = params[5]
    arglat = params[6]
    theta_f = params[7]
    C1 = params[8]
    C2 = params[9]
    C3 = params[10]

    prob = TransferProblem_TrueAnomaly_Optical(
        r_0=r0,
        mu_central_body=mu,
        rotational_speed_central_body=omega_body,
        a_0=a0,
        raan=raan,
        inclination=inc,
        arg_of_latitude=arglat,
        theta_f=theta_f,
        C1=C1,
        C2=C2,
        C3=C3
    )

    # Initial guess: [lambda_r0, lambda_t0, lambda_u0, lambda_v0, nu1, nu2]
    s0 = np.full((6,), -2.0)

    # Optimize with fsolve
    start_time = time.time()
    sol_params, res_norm, nfev, ok = prob.optimize_fsolve(s0, xtol_fsolve=1e-12)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"fsolve success={ok}, ||res||={res_norm:.3e}, evals={nfev}, time={runtime:.3e}")

    # Retrieve scaled time
    revs = int(theta_f/(2*np.pi))
    N = 40*revs + 1
    theta_bar = np.linspace(0.0, 1.0, N)
    sol_integrated = np.array(prob.integrate(sol_params, theta_bar))
    t_bar = sol_integrated[:, 1]

    # Obtain position, velocity, control profiles from solution
    pos_c, vel_c = np.array(prob.orbital_plane_vectors(sol_params, theta_bar))
    n_slf = np.array(prob.control_unit_slf(sol_params, theta_bar))

    # Re-order n_slf to match MATLAB
    n_s = np.zeros_like(n_slf)
    n_s[:, 0] = n_slf[:, 1]
    n_s[:, 1] = n_slf[:, 2]
    n_s[:, 2] = n_slf[:, 0]

    # Rotate velocity costate to Orbital Frame
    lam_c = np.zeros((len(theta_bar), 5))
    lam_u = sol_integrated[:, 6]
    lam_v = sol_integrated[:, 7]
    theta = theta_bar * theta_f + arglat
    lam_vx = lam_u * np.cos(theta) - lam_v * np.sin(theta)
    lam_vy = lam_u * np.sin(theta) + lam_v * np.cos(theta)
    lam_c[:, 2:4] = np.hstack((lam_vx[:, None], lam_vy[:, None]))

    # Adjust scaled theta to match my deinition in MATLAB
    theta_bar = theta_bar + arglat/theta_f

    # Dump to csv file for MATLAB plotting
    data = np.hstack((theta_bar[:, None], pos_c, vel_c, t_bar[:, None], n_s, lam_c))
    header = "theta_bar,x_c,y_c,vx_c,vy_c,t_bar,nx_s,ny_s,nz_s,lam_x,lam_y,lam_vx,lam_vy,lam_t_bar"
    np.savetxt("indirect_output.csv", data, delimiter=",", header=header, comments='')

    data = np.hstack((res_norm, nfev, ok, runtime))
    header = "res_norm,nfev,ok,runtime"
    np.savetxt("indirect_output_conv.csv", data, delimiter=",", header=header, comments='')

if __name__ == "__main__":
    main()
