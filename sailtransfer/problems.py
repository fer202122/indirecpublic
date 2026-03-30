# SPDX-FileCopyrightText: 2026 Oskar Miller <Oskar.Miller2001@gmail.com>
# SPDX-FileCopyrightText: 2026 Fernando Gamez Losada <F.GamezLosada@tudelft.nl>
# SPDX-License-Identifier: MIT
import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import jaxopt
from scipy.optimize import fsolve
from typing import Tuple, Any

jax.config.update("jax_enable_x64", True)

class TransferProblem_TrueAnomaly:
    """Solve the indirect optimal control problem in normalized true anomaly.

    Parameters
    ----------
    r_0 : float
        Initial radius (meters).
    mu_central_body : float
        Gravitational parameter of the central body.
    rotational_speed_central_body : float
        Rotation rate of the central body (rad/s).
    a_0 : float
        Characteristic acceleration (m/s^2).
    raan : float
        Right ascension of the ascending node (radians).
    inclination : float
        Orbit inclination (radians).
    arg_of_latitude : float
        Argument of latitude of the starting point (radians).
    theta_f : float
        Final normalized true anomaly value.
    direction : int
        Direction of transfer (1 for ascending, -1 for descending).

    Usage
    -----
    Instantiate the class with orbital parameters, then use
    :meth:`optimize_jaxopt` or :meth:`optimize_fsolve` to solve for the
    optimal initial costates.  The :meth:`integrate` method can be used to
    propagate the full state/costate history once a solution is obtained.
    """
    def __init__(self,
                 r_0: float,
                 mu_central_body: float,
                 rotational_speed_central_body: float,
                 a_0: float,
                 raan: float,
                 inclination: float,
                 arg_of_latitude: float,
                 theta_f: float,
                 direction: int
                ):

        self.mu_central_body = mu_central_body
        self.rotational_speed_central_body = rotational_speed_central_body
        self.a_0 = a_0
        self.raan = raan
        self.inclination = inclination
        self.arg_of_latitude = arg_of_latitude
        self.theta_f = theta_f
        self.r_0 = r_0
        self.direction = direction

        v_0 = jnp.sqrt(self.mu_central_body / self.r_0)
        initial_period_approx = 2 * np.pi * jnp.sqrt(self.r_0**3 / self.mu_central_body)
        time_f = self.theta_f / (2 * np.pi) * initial_period_approx

        self.r0_jax = jnp.array(self.r_0, dtype=jnp.float64)
        self.v0_jax = jnp.array(v_0, dtype=jnp.float64)
        self.time_f_jax = jnp.array(time_f, dtype=jnp.float64)
        self.theta_f_jax = jnp.array(self.theta_f, dtype=jnp.float64)
        self.a0_jax = jnp.array(self.a_0, dtype=jnp.float64)
        
        self.omega_body_jax = jnp.array(-self.rotational_speed_central_body, dtype=jnp.float64) # negative due to the rotation convention
        self.inclination_jax = jnp.array(self.inclination, dtype=jnp.float64)
        self.raan_jax = jnp.array(self.raan, dtype=jnp.float64)
        self.arg_of_latitude_jax = jnp.array(self.arg_of_latitude, dtype=jnp.float64)

        @jax.jit
        def _calculate_rotation_matrices_jit(theta_bar: jnp.ndarray, time_bar: jnp.ndarray) -> jnp.ndarray:
            theta = theta_bar * self.theta_f_jax + self.arg_of_latitude_jax
            time = time_bar * self.time_f_jax

            # SLF to ECL
            cos_om_t = jnp.cos(self.omega_body_jax * time)
            sin_om_t = jnp.sin(self.omega_body_jax * time)
            R_ECL_to_SLF = jnp.array([[cos_om_t,-sin_om_t,0.],[sin_om_t,cos_om_t,0.],[0.,0.,1.]], dtype=jnp.float64)
            R_SLF_to_ECL = R_ECL_to_SLF.T

            # RAAN rotation
            cos_Om = jnp.cos(self.raan_jax)
            sin_Om = jnp.sin(self.raan_jax)
            Rz_Omega = jnp.array([[cos_Om,-sin_Om,0.],[sin_Om,cos_Om,0.],[0.,0.,1.]], dtype=jnp.float64)

            # inclination rotation
            cos_inc = jnp.cos(self.inclination_jax); sin_inc = jnp.sin(self.inclination_jax)
            Rx_incl = jnp.array([[1.,0.,0.],[0.,cos_inc,-sin_inc],[0.,sin_inc,cos_inc]], dtype=jnp.float64)

            # ORB to ECL
            R_ORB_to_ECL = jnp.matmul(Rz_Omega, Rx_incl)

            # RTN to ORB
            cos_th = jnp.cos(theta)
            sin_th = jnp.sin(theta)
            R_RTN_to_ORB = jnp.array([[cos_th,-sin_th,0.],[sin_th,cos_th,0.],[0.,0.,1.]], dtype=jnp.float64)
            R_ORB_to_RTN = R_RTN_to_ORB.T

            # Combined
            R_RTN_to_ECL = jnp.matmul(R_ORB_to_ECL, R_RTN_to_ORB)
            R_RTN_to_SLF = jnp.matmul(R_ECL_to_SLF, R_RTN_to_ECL)
            
            return R_RTN_to_SLF
        self._rotation_matrices_func = _calculate_rotation_matrices_jit

        @jax.jit
        def _calculate_acceleration_from_primer_vector_jit(lambda_u: jnp.ndarray, lambda_v: jnp.ndarray, R_RTN_to_SLF: jnp.ndarray) -> jnp.ndarray:
            # define primer vector
            primer_vector_rtn = jnp.array([-lambda_u, -lambda_v, 0.0], dtype=jnp.float64)

            # convert primer vector to SLF
            primer_vector_slf = jnp.matmul(R_RTN_to_SLF, primer_vector_rtn)
            norm_pv_slf = jnp.linalg.norm(primer_vector_slf)
            safe_norm_pv_slf = jnp.where(norm_pv_slf < 1e-12, 1e-12, norm_pv_slf)
            primer_vector_slf_unit = primer_vector_slf / safe_norm_pv_slf

            # calculate angle between the sun line and the primer vector (phi)
            ref_vec=jnp.array([1.,0.,0.], dtype=jnp.float64)
            norm_diff = jnp.linalg.norm(primer_vector_slf_unit - ref_vec)
            norm_sum = jnp.linalg.norm(primer_vector_slf_unit + ref_vec)
            phi = 2 * jnp.arctan2(norm_diff, norm_sum + 1e-15)

            # calculate cone angle (alpha)
            sin_phi = jnp.sin(phi)
            safe_asin_arg = jnp.clip(sin_phi/3., -1.+1e-12, 1.-1e-12)
            alpha = 0.5 * (phi - jnp.arcsin(safe_asin_arg))

            # calculate clock angle (delta)
            is_zero = jnp.abs(primer_vector_slf[2]) < 1e-15
            divisor = jnp.where(is_zero, 1e-15, primer_vector_slf[2])
            delta = jnp.arctan2(primer_vector_slf[1], divisor)

            # calculate acceleration in the SLF frame
            cos_alpha = jnp.cos(alpha); sin_alpha = jnp.sin(alpha)
            a_SLF = self.a0_jax * (cos_alpha**2) * jnp.array([cos_alpha, sin_alpha*jnp.sin(delta), sin_alpha*jnp.cos(delta)], dtype=jnp.float64)

            # rotate acceleration to RTN frame to use in the dynamics
            a_RTN = jnp.matmul(R_RTN_to_SLF.T, a_SLF)

            return a_RTN
        self._acceleration_func = _calculate_acceleration_from_primer_vector_jit

        @jax.jit
        def _calculate_state_dynamics_jax_jit(theta_bar: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
            """Compute scaled dynamics for true anomaly integration.

            Parameters
            ----------
            theta_bar : jnp.ndarray
                Normalized true anomaly variable.
            x : jnp.ndarray
                Concatenated state and costate vector.

            Returns
            -------
            jnp.ndarray
                Derivatives of the state variables with respect to ``theta_bar``.
            """
            # extract the necessary states and costates
            r_bar, time_bar, u_bar, v_bar = x[0], x[1], x[2], x[3]
            lambda_u, lambda_v = x[6], x[7]

            # calculate the rotation matrices
            R_RTN_to_SLF = self._rotation_matrices_func(theta_bar, time_bar)

            # calculate the acceleration in RTN frame
            a_RTN = self._acceleration_func(lambda_u, lambda_v, R_RTN_to_SLF)
            a_R, a_T = a_RTN[0], a_RTN[1]

            # calculate the state dynamics
            accel_factor = (self.r0_jax * self.theta_f_jax) / (self.v0_jax**2)
            v_bar_near_zero = jnp.abs(v_bar) < 1e-12
            v_bar_sign = jnp.sign(jnp.where(v_bar_near_zero, 1.0, v_bar))
            v_bar_safe = jnp.where(v_bar_near_zero, v_bar_sign * 1e-12, v_bar)
            r_bar_near_zero = jnp.abs(r_bar) < 1e-12
            r_bar_sign = jnp.sign(jnp.where(r_bar_near_zero, 1.0, r_bar))
            r_bar_safe = jnp.where(r_bar_near_zero, r_bar_sign * 1e-12, r_bar)

            dr_bar_dtheta_bar = u_bar * r_bar_safe * self.theta_f_jax / v_bar_safe
            dtime_bar_dtheta_bar = r_bar_safe / v_bar_safe * (self.theta_f_jax * self.r0_jax) / (self.v0_jax * self.time_f_jax)
            du_bar_dtheta_bar = (v_bar_safe * self.theta_f_jax - self.theta_f_jax / (r_bar_safe * v_bar_safe) + a_R * r_bar_safe / v_bar_safe * accel_factor)
            dv_bar_dtheta_bar = (-u_bar * self.theta_f_jax + a_T * r_bar_safe / v_bar_safe * accel_factor)

            return jnp.array([dr_bar_dtheta_bar, dtime_bar_dtheta_bar, du_bar_dtheta_bar, dv_bar_dtheta_bar])
        self._state_dynamics_func = _calculate_state_dynamics_jax_jit

        @jax.jit
        def _calculate_hamiltonian_jax_jit(state_vars: jnp.ndarray, costate_vars: jnp.ndarray, theta_bar: jnp.ndarray) -> jnp.ndarray:
            x_for_dynamics = jnp.concatenate([state_vars, costate_vars])
            dxdtheta_states = self._state_dynamics_func(theta_bar, x_for_dynamics)
            H = jnp.dot(costate_vars, dxdtheta_states)
            return H
        self._hamiltonian_func = _calculate_hamiltonian_jax_jit
        self._grad_H_wrt_state_func = jax.jit(jax.grad(self._hamiltonian_func, argnums=0))

        def ode_system_for_diffrax_internal(theta_bar: float, x: jnp.ndarray, args: Any) -> jnp.ndarray:
            state_vars = x[0:4]; costate_vars = x[4:8]
            dxdtheta_states = self._state_dynamics_func(theta_bar, x)
            dH_dstate = self._grad_H_wrt_state_func(state_vars, costate_vars, theta_bar)
            return jnp.concatenate([dxdtheta_states, -dH_dstate])
        self.ode_system_for_diffrax = ode_system_for_diffrax_internal

        @jax.jit
        def _calculate_terminal_constraint_matrix_jit(X_final_state_vars: jnp.ndarray) -> jnp.ndarray:
            r_bar, u_bar, v_bar = X_final_state_vars[0], X_final_state_vars[2], X_final_state_vars[3]
            term1 = u_bar
            safe_r_bar = jnp.where(r_bar > 1e-12, r_bar, 1e-12)
            v_target = 1.0 / jnp.sqrt(safe_r_bar)
            term2 = v_bar - v_target
            return jnp.array([term1, term2])
        self._terminal_constraint_func = _calculate_terminal_constraint_matrix_jit

        @jax.jit
        def _terminal_function_phi_jit(state_final_vars: jnp.ndarray, p_free_params: jnp.ndarray) -> jnp.ndarray:
            r_bar = state_final_vars[0]
            Psi = self._terminal_constraint_func(state_final_vars)
            Phi = -1 * self.direction * r_bar + jnp.dot(p_free_params, Psi)
            return Phi
        self._terminal_phi_func = _terminal_function_phi_jit
        self._grad_Phi_wrt_state_func = jax.jit(jax.grad(self._terminal_phi_func, argnums=0))

        def shooting_residuals(s_jax: jnp.ndarray) -> jnp.ndarray:
            # s_jax = [lambda_r0, lambda_t0, lambda_u0, lambda_v0, nu1, nu2] (all JAX arrays)
            X0_state = jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.float64) # Initial states (r,t,u,v)
            X0_costate = s_jax[0:4] # Initial costates from s_jax
            X0_full_jax = jnp.concatenate([X0_state, X0_costate])
            
            p_free_params_jax = s_jax[4:6] # Lagrange multipliers nu from s_jax

            theta_bar_span = (0.0, 1.0)
            term = diffrax.ODETerm(self.ode_system_for_diffrax)
            solver = diffrax.Dopri8()
            stepsize_controller = diffrax.PIDController(rtol=1e-12, atol=1e-12) # Tolerances for ODE solve
            saveat = diffrax.SaveAt(t1=True, t0=False)
            
            sol_jax = diffrax.diffeqsolve(
                term, solver, theta_bar_span[0], theta_bar_span[1], dt0=None, y0=X0_full_jax,
                args=None, stepsize_controller=stepsize_controller, saveat=saveat,
                max_steps=16**5, adjoint=diffrax.ForwardMode()
            )
                        
            X_final_jax = sol_jax.ys[0]
            state_final_jax = X_final_jax[:4]
            costate_final_jax = X_final_jax[4:]

            res_state_constraints = self._terminal_constraint_func(state_final_jax)
            lambda_terminal_target_jax = self._grad_Phi_wrt_state_func(state_final_jax, p_free_params_jax)
            res_costate_constraits = costate_final_jax - lambda_terminal_target_jax
            
            return jnp.concatenate([res_state_constraints, res_costate_constraits])

        self.shooting_residuals = shooting_residuals

    def optimize_jaxopt(self, s_initial_guess_np: np.ndarray, tol_jaxopt: float = 1e-10) -> Tuple[jnp.ndarray, float, int, bool]:
        """
        Optimizes the problem using jaxopt.ScipyRootFinding.
        """
        s_initial_guess_jax = jnp.array(s_initial_guess_np)

        solver = jaxopt.ScipyRootFinding(
            optimality_fun=self.shooting_residuals,
            method="hybr", # hybr or lm
            tol=tol_jaxopt,
            jit=True, 
            use_jacrev=False 
        )

        sol = solver.run(s_initial_guess_jax)
        
        return sol.params, jnp.linalg.norm(sol.state.fun_val), sol.state.num_fun_eval, sol.state.success
    
    def optimize_fsolve(self, s_initial_guess_np: np.ndarray, xtol_fsolve: float = 1e-10) -> Tuple[np.ndarray, float, int, bool]:
        
        def residual_np(s_np: np.ndarray) -> np.ndarray:
            # 1. to JAX
            s_jax = jnp.array(s_np, dtype=jnp.float64)
            # 2. compute JAX residual
            res_jax = self.shooting_residuals(s_jax)
            # 3. back to NumPy
            return np.asarray(res_jax)


        # `full_output=True` provides more info from fsolve
        solution_np, infodict, ier, mesg = fsolve(
            residual_np,
            s_initial_guess_np,
            xtol=xtol_fsolve,
            full_output=True
        )
        
        final_residuals_np = residual_np(solution_np)
        final_norm_np = np.linalg.norm(final_residuals_np)
        
        success = (ier == 1) # ier=1 means solution converged
        num_evals = infodict['nfev'] # Number of function evaluations
        
        return solution_np, final_norm_np, num_evals, success
    
    def integrate(
        self,
        s_solution: jnp.ndarray,
        theta_bar_points: np.ndarray
    ) -> jnp.ndarray:
        """Integrate the optimal state and costate over specified points.

        Parameters
        ----------
        s_solution : jnp.ndarray
            Vector of optimal initial costates and multipliers.
        theta_bar_points : np.ndarray
            Array of normalized true anomaly values where the solution is saved.

        Returns
        -------
        jnp.ndarray
            Integrated state and costate vectors at ``theta_bar_points``.
        """
        # 1) Cast inputs to JAX
        theta_pts = jnp.array(theta_bar_points, dtype=jnp.float64)
        s_jax     = jnp.array(s_solution,    dtype=jnp.float64)

        # 2) Build the initial full vector [state0, costate0]
        x0 = jnp.concatenate([
            jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.float64),
            s_jax[:4]
        ])

        # 3) Set up and solve
        term       = diffrax.ODETerm(self.ode_system_for_diffrax)
        solver     = diffrax.Dopri8()
        controller = diffrax.PIDController(rtol=1e-12, atol=1e-12)
        saveat     = diffrax.SaveAt(ts=theta_pts)

        sol = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=1.0, dt0=None,
            y0=x0,
            args=None,
            stepsize_controller=controller,
            saveat=saveat,
            max_steps=10**6
        )

        return sol.ys
    
    def ecliptic_vectors(
        self,
        s_solution: jnp.ndarray,
        theta_bar_points: np.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Given an optimal solution, return Ecliptic position, velocity, and sail acceleration
        at the specified normalized true anomaly samples.

        Parameters
        ----------
        s_solution : jnp.ndarray
            Optimal initial costates and multipliers (same vector you pass to integrate).
        theta_bar_points : np.ndarray
            Normalized true anomaly samples in [0, 1] at which to output vectors.

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            Tuple of (pos_ecl, vel_ecl, acc_ecl), each shaped (N, 3) with
            columns (x, y, z), (vx, vy, vz), (nx, ny, nz) in SI units.
        """
        # 1) Integrate to get state+costate history at these samples
        ys = self.integrate(s_solution, theta_bar_points)  # shape (N, 8)

        theta_bar_arr = jnp.array(theta_bar_points, dtype=jnp.float64)

        def one_sample(theta_bar: jnp.ndarray, x: jnp.ndarray):
            # Unpack states and costates
            r_bar, time_bar, u_bar, v_bar = x[0], x[1], x[2], x[3]
            lambda_u, lambda_v = x[6], x[7]

            # Physical RTN position/velocity
            r = r_bar * self.r0_jax
            u = u_bar * self.v0_jax
            v = v_bar * self.v0_jax
            pos_rtn = jnp.array([r, 0.0, 0.0], dtype=jnp.float64)
            vel_rtn = jnp.array([u, v, 0.0], dtype=jnp.float64)

            # Rotation matrices
            R_rtn_to_slf = self._rotation_matrices_func(theta_bar, time_bar)

            # Control acceleration in RTN (physical units), then rotate to ECL
            a_rtn = self._acceleration_func(lambda_u, lambda_v, R_rtn_to_slf)

            # Build ECL <-> SLF block for the current time, then recover RTN->ECL
            time_phys = time_bar * self.time_f_jax
            cos_om_t = jnp.cos(self.omega_body_jax * time_phys)
            sin_om_t = jnp.sin(self.omega_body_jax * time_phys)
            R_ecl_to_slf = jnp.array(
                [[cos_om_t, -sin_om_t, 0.0],
                [sin_om_t,  cos_om_t, 0.0],
                [0.0,       0.0,      1.0]],
                dtype=jnp.float64
            )
            R_slf_to_ecl = R_ecl_to_slf.T
            R_rtn_to_ecl = jnp.matmul(R_slf_to_ecl, R_rtn_to_slf)

            # Rotate RTN vectors into ECL
            pos_ecl = jnp.matmul(R_rtn_to_ecl, pos_rtn)
            vel_ecl = jnp.matmul(R_rtn_to_ecl, vel_rtn)
            acc_ecl = jnp.matmul(R_rtn_to_ecl, a_rtn)

            return pos_ecl, vel_ecl, acc_ecl

        # Vectorize over samples
        pos, vel, acc = jax.vmap(one_sample, in_axes=(0, 0))(theta_bar_arr, ys)
        return pos, vel, acc

    def orbital_plane_vectors(
        self,
        s_solution: jnp.ndarray,
        theta_bar_points: np.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Return 2D position and 2D velocity in the fixed orbital plane (ORB frame)
        at the specified normalized true-anomaly samples.

        Parameters
        ----------
        s_solution : jnp.ndarray
            Initial costates + multipliers (same vector you pass to integrate).
        theta_bar_points : np.ndarray
            Normalized true anomaly samples in [0, 1].

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            (pos_orb, vel_orb) each shaped (N, 2):
            pos_orb columns: (x_orb, y_orb) in meters
            vel_orb columns: (vx_orb, vy_orb) in m/s
        """
        ys = self.integrate(s_solution, theta_bar_points)  # (N, 8)
        theta_bar_arr = jnp.array(theta_bar_points, dtype=jnp.float64)

        def one_sample(theta_bar: jnp.ndarray, x: jnp.ndarray):
            r_bar, time_bar, u_bar, v_bar = x[0], x[1], x[2], x[3]

            # physical radial/tangential components
            r = r_bar * self.r0_jax
            u = u_bar * self.v0_jax
            v = v_bar * self.v0_jax

            # angle of the radius in the ORB frame
            theta = theta_bar * self.theta_f_jax + self.arg_of_latitude_jax
            cos_theta = jnp.cos(theta)
            sin_theta = jnp.sin(theta)

            # RTN -> ORB rotation applied to [r, 0] and [u, v]
            pos_orb_x = r * cos_theta
            pos_orb_y = r * sin_theta
            vel_orb_x = u * cos_theta - v * sin_theta
            vel_orb_y = u * sin_theta + v * cos_theta

            return jnp.array([pos_orb_x, pos_orb_y]), jnp.array([vel_orb_x, vel_orb_y])

        pos2, vel2 = jax.vmap(one_sample, in_axes=(0, 0))(theta_bar_arr, ys)

        # normalize position to unit radius
        pos2 = pos2 / self.r0_jax
        vel2 = vel2 / self.v0_jax

        return pos2, vel2


    def control_unit_slf(
        self,
        s_solution: jnp.ndarray,
        theta_bar_points: np.ndarray
    ) -> jnp.ndarray:
        """
        Return the 3D unit control direction in the SLF frame at the given samples.

        Parameters
        ----------
        s_solution : jnp.ndarray
            Initial costates + multipliers (same vector you pass to integrate).
        theta_bar_points : np.ndarray
            Normalized true anomaly samples in [0, 1].

        Returns
        -------
        jnp.ndarray
            Array of shape (N, 3) with unit vectors (nx, ny, nz) in the SLF frame.
        """
        ys = self.integrate(s_solution, theta_bar_points)  # (N, 8)
        theta_bar_arr = jnp.array(theta_bar_points, dtype=jnp.float64)

        def one_sample(theta_bar: jnp.ndarray, x: jnp.ndarray):
            # unpack states/costates
            r_bar, time_bar, u_bar, v_bar = x[0], x[1], x[2], x[3]
            lambda_u, lambda_v = x[6], x[7]

            # RTN->SLF for this (theta_bar, time_bar)
            R_rtn_to_slf = self._rotation_matrices_func(theta_bar, time_bar)

            # acceleration in RTN (includes magnitude factor)
            a_rtn = self._acceleration_func(lambda_u, lambda_v, R_rtn_to_slf)

            # rotate to SLF, then normalize to unit vector
            a_slf = jnp.matmul(R_rtn_to_slf, a_rtn)
            norm = jnp.linalg.norm(a_slf)
            safe_norm = jnp.where(norm < 1e-15, 1e-15, norm)
            return a_slf / safe_norm

        n_slf = jax.vmap(one_sample, in_axes=(0, 0))(theta_bar_arr, ys)  # (N, 3)
        return n_slf

class TransferProblem_Time:
    """Solve the indirect optimal control problem in normalized time.

    Parameters
    ----------
    r_0 : float
        Initial radius (meters).
    mu_central_body : float
        Gravitational parameter of the central body.
    rotational_speed_central_body : float
        Rotation rate of the central body (rad/s).
    a_0 : float
        Characteristic acceleration (m/s^2).
    raan : float
        Right ascension of the ascending node (radians).
    inclination : float
        Orbit inclination (radians).
    arg_of_latitude : float
        Argument of latitude of the starting point (radians).
    time_f : float
        Final normalized time value.

    Usage
    -----
    Instantiate the class with orbital parameters, then call
    :meth:`optimize_jaxopt` or :meth:`optimize_fsolve` to obtain the optimal
    initial costates.  The resulting trajectory can be retrieved with
    :meth:`integrate`.
    """
    def __init__(self,
                 r_0: float,
                 mu_central_body: float,
                 rotational_speed_central_body: float,
                 a_0: float,
                 raan: float,
                 inclination: float,
                 arg_of_latitude: float,
                 time_f: float,
                 direction: int
                ):

        self.mu_central_body = mu_central_body
        self.rotational_speed_central_body = -rotational_speed_central_body # take the negative due to direction of rotation convention
        self.a_0 = a_0
        self.raan = raan
        self.inclination = inclination
        self.arg_of_latitude = arg_of_latitude
        self.time_f = time_f
        self.r_0 = r_0
        self.direction = direction

        v_0 = jnp.sqrt(self.mu_central_body / self.r_0)
        initial_period_approx = 2 * np.pi * jnp.sqrt(self.r_0**3 / self.mu_central_body)
        theta_f = self.time_f / initial_period_approx * 2 * np.pi

        self.r0_jax = jnp.array(self.r_0, dtype=jnp.float64)
        self.v0_jax = jnp.array(v_0, dtype=jnp.float64)
        self.time_f_jax = jnp.array(self.time_f, dtype=jnp.float64)
        self.theta_f_jax = jnp.array(theta_f, dtype=jnp.float64)
        self.a0_jax = jnp.array(self.a_0, dtype=jnp.float64)
        
        self.omega_body_jax = jnp.array(self.rotational_speed_central_body, dtype=jnp.float64)
        self.inclination_jax = jnp.array(self.inclination, dtype=jnp.float64)
        self.raan_jax = jnp.array(self.raan, dtype=jnp.float64)
        self.arg_of_latitude_jax = jnp.array(self.arg_of_latitude, dtype=jnp.float64)

        @jax.jit
        def _calculate_rotation_matrices_jit(theta_bar: jnp.ndarray, time_bar: jnp.ndarray) -> jnp.ndarray:
            theta = theta_bar * self.theta_f_jax + self.arg_of_latitude_jax
            time = time_bar * self.time_f_jax

            # SLF to ECL
            cos_om_t = jnp.cos(self.omega_body_jax * time)
            sin_om_t = jnp.sin(self.omega_body_jax * time)
            R_ECL_to_SLF = jnp.array([[cos_om_t,-sin_om_t,0.],[sin_om_t,cos_om_t,0.],[0.,0.,1.]], dtype=jnp.float64)
            R_SLF_to_ECL = R_ECL_to_SLF.T

            # RAAN rotation
            cos_Om = jnp.cos(self.raan_jax)
            sin_Om = jnp.sin(self.raan_jax)
            Rz_Omega = jnp.array([[cos_Om,-sin_Om,0.],[sin_Om,cos_Om,0.],[0.,0.,1.]], dtype=jnp.float64)

            # inclination rotation
            cos_inc = jnp.cos(self.inclination_jax); sin_inc = jnp.sin(self.inclination_jax)
            Rx_incl = jnp.array([[1.,0.,0.],[0.,cos_inc,-sin_inc],[0.,sin_inc,cos_inc]], dtype=jnp.float64)

            # ORB to ECL
            R_ORB_to_ECL = jnp.matmul(Rz_Omega, Rx_incl)

            # RTN to ORB
            cos_th = jnp.cos(theta)
            sin_th = jnp.sin(theta)
            R_RTN_to_ORB = jnp.array([[cos_th,-sin_th,0.],[sin_th,cos_th,0.],[0.,0.,1.]], dtype=jnp.float64)
            R_ORB_to_RTN = R_RTN_to_ORB.T

            # Combined
            R_RTN_to_ECL = jnp.matmul(R_ORB_to_ECL, R_RTN_to_ORB)
            R_RTN_to_SLF = jnp.matmul(R_ECL_to_SLF, R_RTN_to_ECL)
            
            return R_RTN_to_SLF
        self._rotation_matrices_func = _calculate_rotation_matrices_jit

        @jax.jit
        def _calculate_acceleration_from_primer_vector_jit(lambda_u: jnp.ndarray, lambda_v: jnp.ndarray, R_RTN_to_SLF: jnp.ndarray) -> jnp.ndarray:
            # define primer vector
            primer_vector_rtn = jnp.array([-lambda_u, -lambda_v, 0.0], dtype=jnp.float64)

            # convert primer vector to SLF
            primer_vector_slf = jnp.matmul(R_RTN_to_SLF, primer_vector_rtn)
            norm_pv_slf = jnp.linalg.norm(primer_vector_slf)
            safe_norm_pv_slf = jnp.where(norm_pv_slf < 1e-12, 1e-12, norm_pv_slf)
            primer_vector_slf_unit = primer_vector_slf / safe_norm_pv_slf

            # calculate angle between the sun line and the primer vector (phi)
            ref_vec=jnp.array([1.,0.,0.], dtype=jnp.float64)
            norm_diff = jnp.linalg.norm(primer_vector_slf_unit - ref_vec)
            norm_sum = jnp.linalg.norm(primer_vector_slf_unit + ref_vec)
            phi = 2 * jnp.arctan2(norm_diff, norm_sum + 1e-15)

            # calculate cone angle (alpha)
            sin_phi = jnp.sin(phi)
            safe_asin_arg = jnp.clip(sin_phi/3., -1.+1e-12, 1.-1e-12)
            alpha = 0.5 * (phi - jnp.arcsin(safe_asin_arg))

            # calculate clock angle (delta)
            is_zero = jnp.abs(primer_vector_slf[2]) < 1e-15
            divisor = jnp.where(is_zero, 1e-15, primer_vector_slf[2])
            delta = jnp.arctan2(primer_vector_slf[1], divisor)

            # calculate acceleration in the SLF frame
            cos_alpha = jnp.cos(alpha); sin_alpha = jnp.sin(alpha)
            a_SLF = self.a0_jax * (cos_alpha**2) * jnp.array([cos_alpha, sin_alpha*jnp.sin(delta), sin_alpha*jnp.cos(delta)], dtype=jnp.float64)

            # rotate acceleration to RTN frame to use in the dynamics
            a_RTN = jnp.matmul(R_RTN_to_SLF.T, a_SLF)

            return a_RTN
        self._acceleration_func = _calculate_acceleration_from_primer_vector_jit

        @jax.jit
        def _calculate_state_dynamics_jax_jit(time_bar: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
            """Compute scaled dynamics for normalized time integration.

            Parameters
            ----------
            time_bar : jnp.ndarray
                Normalized time variable.
            x : jnp.ndarray
                Concatenated state and costate vector.

            Returns
            -------
            jnp.ndarray
                Derivatives of the state variables with respect to ``time_bar``.
            """
            # extract the necessary states and costates
            r_bar, theta_bar, u_bar, v_bar = x[0], x[1], x[2], x[3]
            lambda_u, lambda_v = x[6], x[7]

            # calculate the rotation matrices
            R_RTN_to_SLF = self._rotation_matrices_func(theta_bar, time_bar)

            # calculate the acceleration in RTN frame
            a_RTN = self._acceleration_func(lambda_u, lambda_v, R_RTN_to_SLF)
            a_R, a_T = a_RTN[0], a_RTN[1]

            # calculate the state dynamics
            eta = v_0 * time_f / r_0
            v_bar_near_zero = jnp.abs(v_bar) < 1e-12
            v_bar_sign = jnp.sign(jnp.where(v_bar_near_zero, 1.0, v_bar))
            v_bar_safe = jnp.where(v_bar_near_zero, v_bar_sign * 1e-12, v_bar)
            r_bar_near_zero = jnp.abs(r_bar) < 1e-12
            r_bar_sign = jnp.sign(jnp.where(r_bar_near_zero, 1.0, r_bar))
            r_bar_safe = jnp.where(r_bar_near_zero, r_bar_sign * 1e-12, r_bar)

            dr_bar_dtheta_bar = u_bar * eta
            dtheta_bar_dtime_bar = v_bar / r_bar_safe * eta / theta_f
            du_bar_dtheta_bar = (v_bar ** 2 / r_bar_safe - 1 / r_bar_safe ** 2) * eta + a_R * time_f / v_0
            dv_bar_dtheta_bar = - u_bar * v_bar / r_bar_safe * eta + a_T * time_f / v_0

            return jnp.array([dr_bar_dtheta_bar, dtheta_bar_dtime_bar, du_bar_dtheta_bar, dv_bar_dtheta_bar])
        self._state_dynamics_func = _calculate_state_dynamics_jax_jit

        @jax.jit
        def _calculate_hamiltonian_jax_jit(state_vars: jnp.ndarray, costate_vars: jnp.ndarray, time_bar: jnp.ndarray) -> jnp.ndarray:
            x_for_dynamics = jnp.concatenate([state_vars, costate_vars])
            dxdtheta_states = self._state_dynamics_func(time_bar, x_for_dynamics)
            H = jnp.dot(costate_vars, dxdtheta_states)
            return H
        self._hamiltonian_func = _calculate_hamiltonian_jax_jit
        self._grad_H_wrt_state_func = jax.jit(jax.grad(self._hamiltonian_func, argnums=0))

        def ode_system_for_diffrax_internal(time_bar: float, x: jnp.ndarray, args: Any) -> jnp.ndarray:
            state_vars = x[0:4]; costate_vars = x[4:8]
            dxdtheta_states = self._state_dynamics_func(time_bar, x)
            dH_dstate = self._grad_H_wrt_state_func(state_vars, costate_vars, time_bar)
            return jnp.concatenate([dxdtheta_states, -dH_dstate])
        self.ode_system_for_diffrax = ode_system_for_diffrax_internal

        @jax.jit
        def _calculate_terminal_constraint_matrix_jit(X_final_state_vars: jnp.ndarray) -> jnp.ndarray:
            r_bar, u_bar, v_bar = X_final_state_vars[0], X_final_state_vars[2], X_final_state_vars[3]
            term1 = u_bar
            safe_r_bar = jnp.where(r_bar > 1e-12, r_bar, 1e-12)
            v_target = 1.0 / jnp.sqrt(safe_r_bar)
            term2 = v_bar - v_target
            return jnp.array([term1, term2])
        self._terminal_constraint_func = _calculate_terminal_constraint_matrix_jit

        @jax.jit
        def _terminal_function_phi_jit(state_final_vars: jnp.ndarray, p_free_params: jnp.ndarray) -> jnp.ndarray:
            r_bar = state_final_vars[0]
            Psi = self._terminal_constraint_func(state_final_vars)
            Phi = -1 * self.direction * r_bar + jnp.dot(p_free_params, Psi)
            return Phi
        self._terminal_phi_func = _terminal_function_phi_jit
        self._grad_Phi_wrt_state_func = jax.jit(jax.grad(self._terminal_phi_func, argnums=0))

        def shooting_residuals(s_jax: jnp.ndarray) -> jnp.ndarray:
            # s_jax = [lambda_r0, lambda_t0, lambda_u0, lambda_v0, nu1, nu2] (all JAX arrays)
            X0_state = jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.float64) # Initial states (r,t,u,v)
            X0_costate = s_jax[0:4] # Initial costates from s_jax
            X0_full_jax = jnp.concatenate([X0_state, X0_costate])
            
            p_free_params_jax = s_jax[4:6] # Lagrange multipliers nu from s_jax

            time_bar_span = (0.0, 1.0)
            term = diffrax.ODETerm(self.ode_system_for_diffrax)
            solver = diffrax.Dopri8()
            stepsize_controller = diffrax.PIDController(rtol=1e-12, atol=1e-12) # Tolerances for ODE solve
            saveat = diffrax.SaveAt(t1=True, t0=False)
            
            sol_jax = diffrax.diffeqsolve(
                term, solver, time_bar_span[0], time_bar_span[1], dt0=None, y0=X0_full_jax,
                args=None, stepsize_controller=stepsize_controller, saveat=saveat,
                max_steps=16**5, adjoint=diffrax.ForwardMode()
            )
                        
            X_final_jax = sol_jax.ys[0]
            state_final_jax = X_final_jax[:4]
            costate_final_jax = X_final_jax[4:]

            res_state_constraints = self._terminal_constraint_func(state_final_jax)
            lambda_terminal_target_jax = self._grad_Phi_wrt_state_func(state_final_jax, p_free_params_jax)
            res_costate_constraits = costate_final_jax - lambda_terminal_target_jax
            
            return jnp.concatenate([res_state_constraints, res_costate_constraits])

        self.shooting_residuals = shooting_residuals

    def optimize_jaxopt(self, s_initial_guess_np: np.ndarray, tol_jaxopt: float = 1e-10) -> Tuple[jnp.ndarray, float, int, bool]:
        """
        Optimizes the problem using jaxopt.ScipyRootFinding.
        """
        s_initial_guess_jax = jnp.array(s_initial_guess_np)

        solver = jaxopt.ScipyRootFinding(
            optimality_fun=self.shooting_residuals,
            method="hybr", # hybr or lm
            tol=tol_jaxopt,
            jit=True, 
            use_jacrev=False
        )

        sol = solver.run(s_initial_guess_jax)
        
        return sol.params, jnp.linalg.norm(sol.state.fun_val), sol.state.num_fun_eval, sol.state.success
    
    def optimize_fsolve(self, s_initial_guess_np: np.ndarray, xtol_fsolve: float = 1e-10) -> Tuple[np.ndarray, float, int, bool]:
        
        def residual_np(s_np: np.ndarray) -> np.ndarray:
            # 1. to JAX
            s_jax = jnp.array(s_np, dtype=jnp.float64)
            # 2. compute JAX residual
            res_jax = self.shooting_residuals(s_jax)
            # 3. back to NumPy
            return np.asarray(res_jax)


        # `full_output=True` provides more info from fsolve
        solution_np, infodict, ier, mesg = fsolve(
            residual_np,
            s_initial_guess_np,
            xtol=xtol_fsolve,
            full_output=True,
            # maxfev=20
        )
        
        final_residuals_np = residual_np(solution_np)
        final_norm_np = np.linalg.norm(final_residuals_np)
        
        success = (ier == 1) # ier=1 means solution converged
        num_evals = infodict['nfev'] # Number of function evaluations
        
        return solution_np, final_norm_np, num_evals, success
    
    def integrate(
        self,
        s_solution: jnp.ndarray,
        time_bar_points: np.ndarray
    ) -> jnp.ndarray:
        """Integrate the optimal state and costate over specified points.

        Parameters
        ----------
        s_solution : jnp.ndarray
            Vector of optimal initial costates and multipliers.
        time_bar_points : np.ndarray
            Normalized time values at which the solution is sampled.

        Returns
        -------
        jnp.ndarray
            Integrated state and costate vectors at ``time_bar_points``.
        """
        # 1) Cast inputs to JAX
        time_pts = jnp.array(time_bar_points, dtype=jnp.float64)
        s_jax     = jnp.array(s_solution,    dtype=jnp.float64)

        # 2) Build the initial full vector [state0, costate0]
        x0 = jnp.concatenate([
            jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.float64),
            s_jax[:4]
        ])

        # 3) Set up and solve
        term       = diffrax.ODETerm(self.ode_system_for_diffrax)
        solver     = diffrax.Dopri8()
        controller = diffrax.PIDController(rtol=1e-12, atol=1e-12)
        saveat     = diffrax.SaveAt(ts=time_pts)

        sol = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=1.0, dt0=None,
            y0=x0,
            args=None,
            stepsize_controller=controller,
            saveat=saveat,
            max_steps=10**6
        )

        return sol.ys
    
    def ecliptic_vectors(
        self,
        s_solution: jnp.ndarray,
        time_bar_points: np.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Given an optimal solution, return ECL position, velocity, and sail acceleration
        at the specified normalized time samples.

        Parameters
        ----------
        s_solution : jnp.ndarray
            Optimal initial costates and multipliers (same vector you pass to integrate).
        time_bar_points : np.ndarray
            Normalized time samples in [0, 1] at which to output vectors.

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            Tuple of (pos_ecl, vel_ecl, acc_ecl), each shaped (N, 3) with
            columns (x, y, z), (vx, vy, vz), (nx, ny, nz) in SI units.
        """
        # 1) Integrate to get state+costate history at these samples
        ys = self.integrate(s_solution, time_bar_points)  # shape (N, 8)

        time_bar_arr = jnp.array(time_bar_points, dtype=jnp.float64)

        def one_sample(time_bar: jnp.ndarray, x: jnp.ndarray):
            # Unpack states and costates
            r_bar, theta_bar, u_bar, v_bar = x[0], x[1], x[2], x[3]
            lambda_u, lambda_v = x[6], x[7]

            # Physical RTN position/velocity
            r = r_bar * self.r0_jax
            u = u_bar * self.v0_jax
            v = v_bar * self.v0_jax
            pos_rtn = jnp.array([r, 0.0, 0.0], dtype=jnp.float64)
            vel_rtn = jnp.array([u, v, 0.0], dtype=jnp.float64)

            # Rotation matrices
            R_rtn_to_slf = self._rotation_matrices_func(theta_bar, time_bar)

            # Control acceleration in RTN (physical units), then rotate to ECL
            a_rtn = self._acceleration_func(lambda_u, lambda_v, R_rtn_to_slf)

            # Build ECL <-> SLF block for the current time, then recover RTN->ECL
            time_phys = time_bar * self.time_f_jax
            cos_om_t = jnp.cos(self.omega_body_jax * time_phys)
            sin_om_t = jnp.sin(self.omega_body_jax * time_phys)
            R_ecl_to_slf = jnp.array(
                [[cos_om_t, -sin_om_t, 0.0],
                [sin_om_t,  cos_om_t, 0.0],
                [0.0,       0.0,      1.0]],
                dtype=jnp.float64
            )
            R_slf_to_ecl = R_ecl_to_slf.T
            R_rtn_to_ecl = jnp.matmul(R_slf_to_ecl, R_rtn_to_slf)

            # Rotate RTN vectors into ECL
            pos_ecl = jnp.matmul(R_rtn_to_ecl, pos_rtn)
            vel_ecl = jnp.matmul(R_rtn_to_ecl, vel_rtn)
            acc_ecl = jnp.matmul(R_rtn_to_ecl, a_rtn)

            return pos_ecl, vel_ecl, acc_ecl

        # Vectorize over samples
        pos, vel, acc = jax.vmap(one_sample, in_axes=(0, 0))(time_bar_arr, ys)
        return pos, vel, acc

    def orbital_plane_vectors(
        self,
        s_solution: jnp.ndarray,
        time_bar_points: np.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Return 2D position and 2D velocity in the fixed orbital plane (ORB frame)
        at the specified normalized true-anomaly samples.

        Parameters
        ----------
        s_solution : jnp.ndarray
            Initial costates + multipliers (same vector you pass to integrate).
        time_bar_points : np.ndarray
            Normalized time samples in [0, 1].

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            (pos_orb, vel_orb) each shaped (N, 2):
            pos_orb columns: (x_orb, y_orb) in meters
            vel_orb columns: (vx_orb, vy_orb) in m/s
        """
        ys = self.integrate(s_solution, time_bar_points)  # (N, 8)
        time_bar_arr = jnp.array(time_bar_points, dtype=jnp.float64)

        def one_sample(time_bar: jnp.ndarray, x: jnp.ndarray):
            r_bar, theta_bar, u_bar, v_bar = x[0], x[1], x[2], x[3]

            # physical radial/tangential components
            r = r_bar * self.r0_jax
            u = u_bar * self.v0_jax
            v = v_bar * self.v0_jax

            # angle of the radius in the ORB frame
            theta = theta_bar * self.theta_f_jax + self.arg_of_latitude_jax
            cos_theta = jnp.cos(theta)
            sin_theta = jnp.sin(theta)

            # RTN -> ORB rotation applied to [r, 0] and [u, v]
            pos_orb_x = r * cos_theta
            pos_orb_y = r * sin_theta
            vel_orb_x = u * cos_theta - v * sin_theta
            vel_orb_y = u * sin_theta + v * cos_theta

            return jnp.array([pos_orb_x, pos_orb_y]), jnp.array([vel_orb_x, vel_orb_y])

        pos2, vel2 = jax.vmap(one_sample, in_axes=(0, 0))(time_bar_arr, ys)

        # normalize position to unit radius
        pos2 = pos2 / self.r0_jax
        vel2 = vel2 / self.v0_jax

        return pos2, vel2


    def control_unit_slf(
        self,
        s_solution: jnp.ndarray,
        time_bar_points: np.ndarray
    ) -> jnp.ndarray:
        """
        Return the 3D unit control direction in the SLF frame at the given samples.

        Parameters
        ----------
        s_solution : jnp.ndarray
            Initial costates + multipliers (same vector you pass to integrate).
        time_bar_points : np.ndarray
            Normalized time samples in [0, 1].

        Returns
        -------
        jnp.ndarray
            Array of shape (N, 3) with unit vectors (nx, ny, nz) in the SLF frame.
        """
        ys = self.integrate(s_solution, time_bar_points)  # (N, 8)
        time_bar_arr = jnp.array(time_bar_points, dtype=jnp.float64)

        def one_sample(time_bar: jnp.ndarray, x: jnp.ndarray):
            # unpack states/costates
            r_bar, theta_bar, u_bar, v_bar = x[0], x[1], x[2], x[3]
            lambda_u, lambda_v = x[6], x[7]

            # RTN->SLF for this (theta_bar, time_bar)
            R_rtn_to_slf = self._rotation_matrices_func(theta_bar, time_bar)

            # acceleration in RTN (includes magnitude factor)
            a_rtn = self._acceleration_func(lambda_u, lambda_v, R_rtn_to_slf)

            # rotate to SLF, then normalize to unit vector
            a_slf = jnp.matmul(R_rtn_to_slf, a_rtn)
            norm = jnp.linalg.norm(a_slf)
            safe_norm = jnp.where(norm < 1e-15, 1e-15, norm)
            return a_slf / safe_norm

        n_slf = jax.vmap(one_sample, in_axes=(0, 0))(time_bar_arr, ys)  # (N, 3)
        return n_slf
        
class TransferProblem_TrueAnomaly_Optical:
    """Solve the indirect optimal control problem in normalized true anomaly.

    Parameters
    ----------
    r_0 : float
        Initial radius (meters).
    mu_central_body : float
        Gravitational parameter of the central body.
    rotational_speed_central_body : float
        Rotation rate of the central body (rad/s).
    a_0 : float
        Characteristic acceleration (m/s^2).
    raan : float
        Right ascension of the ascending node (radians).
    inclination : float
        Orbit inclination (radians).
    arg_of_latitude : float
        Argument of latitude of the starting point (radians).
    theta_f : float
        Final normalized true anomaly value.
    direction : int
        Direction of transfer (1 for ascending, -1 for descending).
    C1, C2, C3 : float
        Coefficients of the solar-sail aceeleration optical model.

    Usage
    -----
    Instantiate the class with orbital parameters, then use
    :meth:`optimize_jaxopt` or :meth:`optimize_fsolve` to solve for the
    optimal initial costates.  The :meth:`integrate` method can be used to
    propagate the full state/costate history once a solution is obtained.
    """
    def __init__(self,
                 r_0: float,
                 mu_central_body: float,
                 rotational_speed_central_body: float,
                 a_0: float,
                 raan: float,
                 inclination: float,
                 arg_of_latitude: float,
                 theta_f: float,
                 direction: int,
                 C1: float,
                 C2: float,
                 C3: float
                ):

        self.r_0 = r_0
        self.mu_central_body = mu_central_body
        self.rotational_speed_central_body = rotational_speed_central_body
        self.a_0 = a_0
        self.raan = raan
        self.inclination = inclination
        self.arg_of_latitude = arg_of_latitude
        self.theta_f = theta_f
        self.direction = direction
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3

        v_0 = jnp.sqrt(self.mu_central_body / self.r_0)
        initial_period_approx = 2 * np.pi * jnp.sqrt(self.r_0**3 / self.mu_central_body)
        time_f = self.theta_f / (2 * np.pi) * initial_period_approx

        self.r0_jax = jnp.array(self.r_0, dtype=jnp.float64)
        self.v0_jax = jnp.array(v_0, dtype=jnp.float64)
        self.time_f_jax = jnp.array(time_f, dtype=jnp.float64)
        self.theta_f_jax = jnp.array(self.theta_f, dtype=jnp.float64)
        self.a0_jax = jnp.array(self.a_0, dtype=jnp.float64)
        self.C1_jax = jnp.array(self.C1, dtype=jnp.float64)
        self.C2_jax = jnp.array(self.C2, dtype=jnp.float64)
        self.C3_jax = jnp.array(self.C3, dtype=jnp.float64)

        self.omega_body_jax = jnp.array(-self.rotational_speed_central_body, dtype=jnp.float64) # negative due to the rotation convention
        self.inclination_jax = jnp.array(self.inclination, dtype=jnp.float64)
        self.raan_jax = jnp.array(self.raan, dtype=jnp.float64)
        self.arg_of_latitude_jax = jnp.array(self.arg_of_latitude, dtype=jnp.float64)

        @jax.jit
        def _calculate_rotation_matrices_jit(theta_bar: jnp.ndarray, time_bar: jnp.ndarray) -> jnp.ndarray:
            theta = theta_bar * self.theta_f_jax + self.arg_of_latitude_jax
            time = time_bar * self.time_f_jax

            # SLF to ECL
            cos_om_t = jnp.cos(self.omega_body_jax * time)
            sin_om_t = jnp.sin(self.omega_body_jax * time)
            R_ECL_to_SLF = jnp.array([[cos_om_t,-sin_om_t,0.],[sin_om_t,cos_om_t,0.],[0.,0.,1.]], dtype=jnp.float64)
            R_SLF_to_ECL = R_ECL_to_SLF.T

            # RAAN rotation
            cos_Om = jnp.cos(self.raan_jax)
            sin_Om = jnp.sin(self.raan_jax)
            Rz_Omega = jnp.array([[cos_Om,-sin_Om,0.],[sin_Om,cos_Om,0.],[0.,0.,1.]], dtype=jnp.float64)

            # inclination rotation
            cos_inc = jnp.cos(self.inclination_jax); sin_inc = jnp.sin(self.inclination_jax)
            Rx_incl = jnp.array([[1.,0.,0.],[0.,cos_inc,-sin_inc],[0.,sin_inc,cos_inc]], dtype=jnp.float64)

            # ORB to ECL
            R_ORB_to_ECL = jnp.matmul(Rz_Omega, Rx_incl)

            # RTN to ORB
            cos_th = jnp.cos(theta)
            sin_th = jnp.sin(theta)
            R_RTN_to_ORB = jnp.array([[cos_th,-sin_th,0.],[sin_th,cos_th,0.],[0.,0.,1.]], dtype=jnp.float64)
            R_ORB_to_RTN = R_RTN_to_ORB.T

            # Combined
            R_RTN_to_ECL = jnp.matmul(R_ORB_to_ECL, R_RTN_to_ORB)
            R_RTN_to_SLF = jnp.matmul(R_ECL_to_SLF, R_RTN_to_ECL)
            
            return R_RTN_to_SLF
        self._rotation_matrices_func = _calculate_rotation_matrices_jit

        @jax.jit
        def _calculate_acceleration_from_primer_vector_jit(lambda_u: jnp.ndarray, lambda_v: jnp.ndarray, R_RTN_to_SLF: jnp.ndarray) -> jnp.ndarray:
            # calculate normal vector in SLF frame
            n_slf = self._normal_func(lambda_u, lambda_v, R_RTN_to_SLF)

            # calculate acceleration in the SLF frame
            cos_alpha = n_slf[0]
            com_factor = self.C1_jax * cos_alpha + self.C2_jax
            a_SLF = self.a0_jax / 2 * cos_alpha * jnp.array([com_factor * cos_alpha + self.C3_jax, 
                                                             com_factor * n_slf[1], 
                                                             com_factor * n_slf[2]], dtype=jnp.float64)

            # rotate acceleration to RTN frame to use in the dynamics
            a_RTN = jnp.matmul(R_RTN_to_SLF.T, a_SLF)

            return a_RTN
        self._acceleration_func = _calculate_acceleration_from_primer_vector_jit

        @jax.jit
        def _calculate_normal_from_primer_vector_jit(lambda_u: jnp.ndarray, lambda_v: jnp.ndarray, R_RTN_to_SLF: jnp.ndarray) -> jnp.ndarray:
            # define primer vector
            primer_vector_rtn = jnp.array([-lambda_u, -lambda_v, 0.0], dtype=jnp.float64)

            # convert primer vector to SLF
            primer_vector_slf = jnp.matmul(R_RTN_to_SLF, primer_vector_rtn)
            norm_pv_slf = jnp.linalg.norm(primer_vector_slf)
            safe_norm_pv_slf = jnp.where(norm_pv_slf < 1e-12, 1e-12, norm_pv_slf)
            primer_vector_slf_unit = primer_vector_slf / safe_norm_pv_slf

            # calculate angle between the sun line and the primer vector (phi)
            ref_vec=jnp.array([1.,0.,0.], dtype=jnp.float64)
            norm_diff = jnp.linalg.norm(primer_vector_slf_unit - ref_vec)
            norm_sum = jnp.linalg.norm(primer_vector_slf_unit + ref_vec)
            phi = 2 * jnp.arctan2(norm_diff, norm_sum + 1e-15)

            # calculate cone angle (alpha)
            sin_phi = jnp.sin(phi)
            safe_asin_arg = jnp.clip(sin_phi/3., -1.+1e-12, 1.-1e-12)
            alpha_ideal = 0.5 * (phi - jnp.arcsin(safe_asin_arg))

            def alpha_critical() -> jnp.ndarray:
                alpha_critical = (-self.C3_jax*self.C2_jax - 2*self.C1_jax*self.C2_jax + jnp.sqrt(self.C3_jax**2*self.C2_jax**2 - 4*self.C3_jax*self.C2_jax**2*self.C1_jax + 8*self.C3_jax**2*self.C1_jax**2 + 4*self.C1_jax**3*self.C3_jax))/(4*self.C3_jax*self.C1_jax + 2*self.C1_jax**2)
                return jnp.arccos(alpha_critical)

            def phi_critical() -> jnp.ndarray:
                alpha = alpha_critical()
                phi_critical = jnp.sin(alpha)*(3*self.C1_jax*jnp.cos(alpha)**2 + 2*self.C2_jax*jnp.cos(alpha) + self.C3_jax)/(jnp.cos(alpha)**2*(self.C1_jax*jnp.cos(alpha) + self.C2_jax) - jnp.sin(alpha)**2*(2*self.C1_jax*jnp.cos(alpha) + self.C2_jax))
                phi_critical = jnp.arctan(phi_critical)
                return jnp.where(phi_critical < 0, phi_critical + jnp.pi, phi_critical)

            def dJ_dalpha(alpha: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
                k = jnp.cos(phi) / jnp.sin(phi)
                return k * jnp.sin(alpha) * (3 * self.C1_jax * jnp.cos(alpha)**2 + 2 * self.C2_jax * jnp.cos(alpha) + self.C3_jax) - self.C1_jax * jnp.cos(alpha) * (1 - 3 * jnp.sin(alpha)**2) - self.C2_jax * jnp.cos(2 * alpha)

            def dJ_dalpha2(alpha: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
                k = jnp.cos(phi) / jnp.sin(phi)
                return k * (3 * self.C1_jax * (jnp.cos(alpha)**3 - 2 * jnp.cos(alpha) * jnp.sin(alpha)**2) + 2 * self.C2_jax * jnp.cos(2 * alpha) + self.C3_jax * jnp.cos(alpha)) - self.C1_jax * jnp.sin(alpha) * (2 - 9 * jnp.cos(alpha)**2) + 2 * self.C2_jax * jnp.sin(2 * alpha)

            # Implement control logic with jnp.where to avoid branching
            phi_critical = phi_critical()
            alpha_corrected = alpha_ideal - dJ_dalpha(alpha_ideal, phi) / dJ_dalpha2(alpha_ideal, phi)
            alpha = jnp.zeros_like(phi)
            alpha = jnp.where(phi == 0, 0.0, alpha) # Avoid singularity
            alpha = jnp.where(phi > 0, alpha_corrected, alpha) # Correct ideal cone angle
            alpha = jnp.where(phi > phi_critical, jnp.pi/2, alpha) # For larger angles, deactivate sail

            # calculate clock angle (delta)
            is_zero = jnp.abs(primer_vector_slf[2]) < 1e-15
            divisor = jnp.where(is_zero, 1e-15, primer_vector_slf[2])
            delta = jnp.arctan2(primer_vector_slf[1], divisor)

            # calculate normal vector in the SLF frame
            cos_alpha = jnp.cos(alpha); sin_alpha = jnp.sin(alpha); 
            n_SLF = jnp.array([cos_alpha,
                               sin_alpha * jnp.sin(delta), 
                               sin_alpha * jnp.cos(delta)], dtype=jnp.float64)
            return n_SLF
        self._normal_func = _calculate_normal_from_primer_vector_jit

        @jax.jit
        def _calculate_state_dynamics_jax_jit(theta_bar: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
            """Compute scaled dynamics for true anomaly integration.

            Parameters
            ----------
            theta_bar : jnp.ndarray
                Normalized true anomaly variable.
            x : jnp.ndarray
                Concatenated state and costate vector.

            Returns
            -------
            jnp.ndarray
                Derivatives of the state variables with respect to ``theta_bar``.
            """
            # extract the necessary states and costates
            r_bar, time_bar, u_bar, v_bar = x[0], x[1], x[2], x[3]
            lambda_u, lambda_v = x[6], x[7]

            # calculate the rotation matrices
            R_RTN_to_SLF = self._rotation_matrices_func(theta_bar, time_bar)

            # calculate the acceleration in RTN frame
            a_RTN = self._acceleration_func(lambda_u, lambda_v, R_RTN_to_SLF)
            a_R, a_T = a_RTN[0], a_RTN[1]

            # calculate the state dynamics
            accel_factor = (self.r0_jax * self.theta_f_jax) / (self.v0_jax**2)
            v_bar_near_zero = jnp.abs(v_bar) < 1e-12
            v_bar_sign = jnp.sign(jnp.where(v_bar_near_zero, 1.0, v_bar))
            v_bar_safe = jnp.where(v_bar_near_zero, v_bar_sign * 1e-12, v_bar)
            r_bar_near_zero = jnp.abs(r_bar) < 1e-12
            r_bar_sign = jnp.sign(jnp.where(r_bar_near_zero, 1.0, r_bar))
            r_bar_safe = jnp.where(r_bar_near_zero, r_bar_sign * 1e-12, r_bar)

            dr_bar_dtheta_bar = u_bar * r_bar_safe * self.theta_f_jax / v_bar_safe
            dtime_bar_dtheta_bar = r_bar_safe / v_bar_safe * (self.theta_f_jax * self.r0_jax) / (self.v0_jax * self.time_f_jax)
            du_bar_dtheta_bar = (v_bar_safe * self.theta_f_jax - self.theta_f_jax / (r_bar_safe * v_bar_safe) + a_R * r_bar_safe / v_bar_safe * accel_factor)
            dv_bar_dtheta_bar = (-u_bar * self.theta_f_jax + a_T * r_bar_safe / v_bar_safe * accel_factor)

            return jnp.array([dr_bar_dtheta_bar, dtime_bar_dtheta_bar, du_bar_dtheta_bar, dv_bar_dtheta_bar])
        self._state_dynamics_func = _calculate_state_dynamics_jax_jit

        @jax.jit
        def _calculate_hamiltonian_jax_jit(state_vars: jnp.ndarray, costate_vars: jnp.ndarray, theta_bar: jnp.ndarray) -> jnp.ndarray:
            x_for_dynamics = jnp.concatenate([state_vars, costate_vars])
            dxdtheta_states = self._state_dynamics_func(theta_bar, x_for_dynamics)
            H = jnp.dot(costate_vars, dxdtheta_states)
            return H
        self._hamiltonian_func = _calculate_hamiltonian_jax_jit
        self._grad_H_wrt_state_func = jax.jit(jax.grad(self._hamiltonian_func, argnums=0))

        def ode_system_for_diffrax_internal(theta_bar: float, x: jnp.ndarray, args: Any) -> jnp.ndarray:
            state_vars = x[0:4]; costate_vars = x[4:8]
            dxdtheta_states = self._state_dynamics_func(theta_bar, x)
            dH_dstate = self._grad_H_wrt_state_func(state_vars, costate_vars, theta_bar)
            return jnp.concatenate([dxdtheta_states, -dH_dstate])
        self.ode_system_for_diffrax = ode_system_for_diffrax_internal

        @jax.jit
        def _calculate_terminal_constraint_matrix_jit(X_final_state_vars: jnp.ndarray) -> jnp.ndarray:
            r_bar, u_bar, v_bar = X_final_state_vars[0], X_final_state_vars[2], X_final_state_vars[3]
            term1 = u_bar
            safe_r_bar = jnp.where(r_bar > 1e-12, r_bar, 1e-12)
            v_target = 1.0 / jnp.sqrt(safe_r_bar)
            term2 = v_bar - v_target
            return jnp.array([term1, term2])
        self._terminal_constraint_func = _calculate_terminal_constraint_matrix_jit

        @jax.jit
        def _terminal_function_phi_jit(state_final_vars: jnp.ndarray, p_free_params: jnp.ndarray) -> jnp.ndarray:
            r_bar = state_final_vars[0]
            Psi = self._terminal_constraint_func(state_final_vars)
            Phi = -1 * self.direction * r_bar + jnp.dot(p_free_params, Psi)
            return Phi
        self._terminal_phi_func = _terminal_function_phi_jit
        self._grad_Phi_wrt_state_func = jax.jit(jax.grad(self._terminal_phi_func, argnums=0))

        def shooting_residuals(s_jax: jnp.ndarray) -> jnp.ndarray:
            # s_jax = [lambda_r0, lambda_t0, lambda_u0, lambda_v0, nu1, nu2] (all JAX arrays)
            X0_state = jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.float64) # Initial states (r,t,u,v)
            X0_costate = s_jax[0:4] # Initial costates from s_jax
            X0_full_jax = jnp.concatenate([X0_state, X0_costate])
            
            p_free_params_jax = s_jax[4:6] # Lagrange multipliers nu from s_jax

            theta_bar_span = (0.0, 1.0)
            term = diffrax.ODETerm(self.ode_system_for_diffrax)
            solver = diffrax.Dopri8()
            stepsize_controller = diffrax.PIDController(rtol=1e-12, atol=1e-12) # Tolerances for ODE solve
            saveat = diffrax.SaveAt(t1=True, t0=False)
            
            sol_jax = diffrax.diffeqsolve(
                term, solver, theta_bar_span[0], theta_bar_span[1], dt0=None, y0=X0_full_jax,
                args=None, stepsize_controller=stepsize_controller, saveat=saveat,
                max_steps=16**5, adjoint=diffrax.ForwardMode()
            )
                        
            X_final_jax = sol_jax.ys[0]
            state_final_jax = X_final_jax[:4]
            costate_final_jax = X_final_jax[4:]

            res_state_constraints = self._terminal_constraint_func(state_final_jax)
            lambda_terminal_target_jax = self._grad_Phi_wrt_state_func(state_final_jax, p_free_params_jax)
            res_costate_constraits = costate_final_jax - lambda_terminal_target_jax
            
            return jnp.concatenate([res_state_constraints, res_costate_constraits])

        self.shooting_residuals = shooting_residuals

    def optimize_jaxopt(self, s_initial_guess_np: np.ndarray, tol_jaxopt: float = 1e-10) -> Tuple[jnp.ndarray, float, int, bool]:
        """
        Optimizes the problem using jaxopt.ScipyRootFinding.
        """
        s_initial_guess_jax = jnp.array(s_initial_guess_np)

        solver = jaxopt.ScipyRootFinding(
            optimality_fun=self.shooting_residuals,
            method="hybr", # hybr or lm
            tol=tol_jaxopt,
            jit=True, 
            use_jacrev=False 
        )

        sol = solver.run(s_initial_guess_jax)
        
        return sol.params, jnp.linalg.norm(sol.state.fun_val), sol.state.num_fun_eval, sol.state.success
    
    def optimize_fsolve(self, s_initial_guess_np: np.ndarray, xtol_fsolve: float = 1e-10) -> Tuple[np.ndarray, float, int, bool]:
        
        def residual_np(s_np: np.ndarray) -> np.ndarray:
            # 1. to JAX
            s_jax = jnp.array(s_np, dtype=jnp.float64)
            # 2. compute JAX residual
            res_jax = self.shooting_residuals(s_jax)
            # 3. back to NumPy
            return np.asarray(res_jax)


        # `full_output=True` provides more info from fsolve
        solution_np, infodict, ier, mesg = fsolve(
            residual_np,
            s_initial_guess_np,
            xtol=xtol_fsolve,
            full_output=True
        )
        
        final_residuals_np = residual_np(solution_np)
        final_norm_np = np.linalg.norm(final_residuals_np)
        
        success = (ier == 1) # ier=1 means solution converged
        num_evals = infodict['nfev'] # Number of function evaluations
        
        return solution_np, final_norm_np, num_evals, success
    
    def integrate(
        self,
        s_solution: jnp.ndarray,
        theta_bar_points: np.ndarray
    ) -> jnp.ndarray:
        """Integrate the optimal state and costate over specified points.

        Parameters
        ----------
        s_solution : jnp.ndarray
            Vector of optimal initial costates and multipliers.
        theta_bar_points : np.ndarray
            Array of normalized true anomaly values where the solution is saved.

        Returns
        -------
        jnp.ndarray
            Integrated state and costate vectors at ``theta_bar_points``.
        """
        # 1) Cast inputs to JAX
        theta_pts = jnp.array(theta_bar_points, dtype=jnp.float64)
        s_jax     = jnp.array(s_solution,    dtype=jnp.float64)

        # 2) Build the initial full vector [state0, costate0]
        x0 = jnp.concatenate([
            jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.float64),
            s_jax[:4]
        ])

        # 3) Set up and solve
        term       = diffrax.ODETerm(self.ode_system_for_diffrax)
        solver     = diffrax.Dopri8()
        controller = diffrax.PIDController(rtol=1e-12, atol=1e-12)
        saveat     = diffrax.SaveAt(ts=theta_pts)

        sol = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=1.0, dt0=None,
            y0=x0,
            args=None,
            stepsize_controller=controller,
            saveat=saveat,
            max_steps=10**6
        )

        return sol.ys
    
    def ecliptic_vectors(
        self,
        s_solution: jnp.ndarray,
        theta_bar_points: np.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Given an optimal solution, return ECL position, velocity, and sail acceleration
        at the specified normalized true anomaly samples.

        Parameters
        ----------
        s_solution : jnp.ndarray
            Optimal initial costates and multipliers (same vector you pass to integrate).
        theta_bar_points : np.ndarray
            Normalized true anomaly samples in [0, 1] at which to output vectors.

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            Tuple of (pos_ecl, vel_ecl, acc_ecl), each shaped (N, 3) with
            columns (x, y, z), (vx, vy, vz), (nx, ny, nz) in SI units.
        """
        # 1) Integrate to get state+costate history at these samples
        ys = self.integrate(s_solution, theta_bar_points)  # shape (N, 8)

        theta_bar_arr = jnp.array(theta_bar_points, dtype=jnp.float64)

        def one_sample(theta_bar: jnp.ndarray, x: jnp.ndarray):
            # Unpack states and costates
            r_bar, time_bar, u_bar, v_bar = x[0], x[1], x[2], x[3]
            lambda_u, lambda_v = x[6], x[7]

            # Physical RTN position/velocity
            r = r_bar * self.r0_jax
            u = u_bar * self.v0_jax
            v = v_bar * self.v0_jax
            pos_rtn = jnp.array([r, 0.0, 0.0], dtype=jnp.float64)
            vel_rtn = jnp.array([u, v, 0.0], dtype=jnp.float64)

            # Rotation matrices
            R_rtn_to_slf = self._rotation_matrices_func(theta_bar, time_bar)

            # Control acceleration in RTN (physical units), then rotate to ECL
            a_rtn = self._acceleration_func(lambda_u, lambda_v, R_rtn_to_slf)

            # Build ECL <-> SLF block for the current time, then recover RTN->ECL
            time_phys = time_bar * self.time_f_jax
            cos_om_t = jnp.cos(self.omega_body_jax * time_phys)
            sin_om_t = jnp.sin(self.omega_body_jax * time_phys)
            R_ecl_to_slf = jnp.array(
                [[cos_om_t, -sin_om_t, 0.0],
                [sin_om_t,  cos_om_t, 0.0],
                [0.0,       0.0,      1.0]],
                dtype=jnp.float64
            )
            R_slf_to_ecl = R_ecl_to_slf.T
            R_rtn_to_ecl = jnp.matmul(R_slf_to_ecl, R_rtn_to_slf)

            # Rotate RTN vectors into ECL
            pos_ecl = jnp.matmul(R_rtn_to_ecl, pos_rtn)
            vel_ecl = jnp.matmul(R_rtn_to_ecl, vel_rtn)
            acc_ecl = jnp.matmul(R_rtn_to_ecl, a_rtn)

            return pos_ecl, vel_ecl, acc_ecl

        # Vectorize over samples
        pos, vel, acc = jax.vmap(one_sample, in_axes=(0, 0))(theta_bar_arr, ys)
        return pos, vel, acc

    def orbital_plane_vectors(
        self,
        s_solution: jnp.ndarray,
        theta_bar_points: np.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Return 2D position and 2D velocity in the fixed orbital plane (ORB frame)
        at the specified normalized true-anomaly samples.

        Parameters
        ----------
        s_solution : jnp.ndarray
            Initial costates + multipliers (same vector you pass to integrate).
        theta_bar_points : np.ndarray
            Normalized true anomaly samples in [0, 1].

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            (pos_orb, vel_orb) each shaped (N, 2):
            pos_orb columns: (x_orb, y_orb) in meters
            vel_orb columns: (vx_orb, vy_orb) in m/s
        """
        ys = self.integrate(s_solution, theta_bar_points)  # (N, 8)
        theta_bar_arr = jnp.array(theta_bar_points, dtype=jnp.float64)

        def one_sample(theta_bar: jnp.ndarray, x: jnp.ndarray):
            r_bar, time_bar, u_bar, v_bar = x[0], x[1], x[2], x[3]

            # physical radial/tangential components
            r = r_bar * self.r0_jax
            u = u_bar * self.v0_jax
            v = v_bar * self.v0_jax

            # angle of the radius in the ORB frame
            theta = theta_bar * self.theta_f_jax + self.arg_of_latitude_jax
            cos_theta = jnp.cos(theta)
            sin_theta = jnp.sin(theta)

            # RTN -> ORB rotation applied to [r, 0] and [u, v]
            pos_orb_x = r * cos_theta
            pos_orb_y = r * sin_theta
            vel_orb_x = u * cos_theta - v * sin_theta
            vel_orb_y = u * sin_theta + v * cos_theta

            return jnp.array([pos_orb_x, pos_orb_y]), jnp.array([vel_orb_x, vel_orb_y])

        pos2, vel2 = jax.vmap(one_sample, in_axes=(0, 0))(theta_bar_arr, ys)

        # normalize position to unit radius
        pos2 = pos2 / self.r0_jax
        vel2 = vel2 / self.v0_jax

        return pos2, vel2


    def control_unit_slf(
        self,
        s_solution: jnp.ndarray,
        theta_bar_points: np.ndarray
    ) -> jnp.ndarray:
        """
        Return the 3D unit control direction in the SLF frame at the given samples.

        Parameters
        ----------
        s_solution : jnp.ndarray
            Initial costates + multipliers (same vector you pass to integrate).
        theta_bar_points : np.ndarray
            Normalized true anomaly samples in [0, 1].

        Returns
        -------
        jnp.ndarray
            Array of shape (N, 3) with unit vectors (nx, ny, nz) in the SLF frame.
        """
        ys = self.integrate(s_solution, theta_bar_points)  # (N, 8)
        theta_bar_arr = jnp.array(theta_bar_points, dtype=jnp.float64)

        def one_sample(theta_bar: jnp.ndarray, x: jnp.ndarray):
            # unpack states/costates
            r_bar, time_bar, u_bar, v_bar = x[0], x[1], x[2], x[3]
            lambda_u, lambda_v = x[6], x[7]

            # RTN->SLF for this (theta_bar, time_bar)
            R_rtn_to_slf = self._rotation_matrices_func(theta_bar, time_bar)

            # normal vector in SLF
            return self._normal_func(lambda_u, lambda_v, R_rtn_to_slf)

        n_slf = jax.vmap(one_sample, in_axes=(0, 0))(theta_bar_arr, ys)  # (N, 3)
        return n_slf
