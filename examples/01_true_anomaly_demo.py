import numpy as np
import jax.numpy as jnp
from sailtransfer import TransferProblem_TrueAnomaly
import time

def main():
    # Input parameters
    mu         = 3.986004418e14    # Gravitational parameter (Earth) [m^3/s^2]
    omega_body = 7.2921159e-5      # Central body rotation rate (Earth) [rad/s]
    r0         = 20000e3           # Initial radius (MEO) [m]
    a0         = 0.05e-3           # Characteristic acceleration [m/s^2]
    raan       = np.radians(30.0)  # Right ascension of the ascending node in Ecliptic frame [rad] 
    inc        = np.radians(60.0)  # Inclination in Ecliptic frame [rad] 
    arglat     = np.radians(270.0) # Initial Argument of Latitude [rad]
    theta_f    = 2.0*np.pi*100     # Total transfer angle [rad]
    direction  = 1                 # Ascending (1) or descending (-1)

    # Create the transfer problem instance
    prob = TransferProblem_TrueAnomaly(
        r_0=r0,
        mu_central_body=mu,
        rotational_speed_central_body=omega_body,
        a_0=a0,
        raan=raan,
        inclination=inc,
        arg_of_latitude=arglat,
        theta_f=theta_f,
        direction=direction
    )

    # Initial guess: [lambda_r0, lambda_t0, lambda_u0, lambda_v0, nu1, nu2]
    s0 = np.full((6,), -2.0)

    print(f"  Initial guess: {s0}")

    # Optimize with fsolve
    start_time = time.time()
    sol_params, res_norm, nfev, ok = prob.optimize_fsolve(s0, xtol_fsolve=1e-12)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"  fsolve success={ok}, ||res||={res_norm:.3e}, evals={nfev}, time={runtime:.3e}")

    # Check for convergence
    if not ok and res_norm > 1e-9:
        raise RuntimeError(f"fsolve failed with res_norm={res_norm:.3e}, evals={nfev}")

    # Retrieve scaled time
    revs = int(theta_f/(2*np.pi))
    N = 40*revs + 1
    theta_bar = np.linspace(0.0, 1.0, N)
    sol_integrated = np.array(prob.integrate(sol_params, theta_bar))
    t_bar = sol_integrated[:, 1]

    # Obtain position, velocity, control profiles from solution
    pos_c, vel_c = np.array(prob.orbital_plane_vectors(sol_params, theta_bar))
    n_s = np.array(prob.control_unit_slf(sol_params, theta_bar))

    # Rotate velocity costate to Orbital Frame
    lam_c = np.zeros((len(theta_bar), 5))
    lam_u = sol_integrated[:, 6]
    lam_v = sol_integrated[:, 7]
    theta = theta_bar * theta_f + arglat
    lam_vx = lam_u * np.cos(theta) - lam_v * np.sin(theta)
    lam_vy = lam_u * np.sin(theta) + lam_v * np.cos(theta)
    lam_c[:, 2:4] = np.hstack((lam_vx[:, None], lam_vy[:, None]))

    # Dump to csv file for plotting
    data = np.hstack((theta_bar[:, None], pos_c, vel_c, t_bar[:, None], n_s, lam_c))
    header = "theta_bar,x_c,y_c,vx_c,vy_c,t_bar,nx_s,ny_s,nz_s,lam_x,lam_y,lam_vx,lam_vy,lam_t_bar"
    np.savetxt("indirect_output.csv", data, delimiter=",", header=header, comments='')

    data = np.hstack((res_norm, nfev, ok, runtime))
    header = "res_norm,nfev,ok,runtime"
    np.savetxt("indirect_output_conv.csv", data, delimiter=",", header=header, comments='')

    data = np.atleast_2d(sol_params)
    header = "lam_r0,lam_t0,lam_u0,lam_v0,nu1,nu2"
    np.savetxt("indirect_output_guess.csv", data, delimiter=",", header=header, comments='')

if __name__ == "__main__":
    main()
