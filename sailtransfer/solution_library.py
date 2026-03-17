# SPDX-FileCopyrightText: 2026 Oskar Miller <Oskar.Miller2001@gmail.com>
# SPDX-FileCopyrightText: 2026 Fernando Gamez Losada <F.GamezLosada@tudelft.nl>
# SPDX-License-Identifier: MIT
from __future__ import annotations
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import jax.numpy as jnp

Mode = Literal["true_anomaly", "time"]

_TWO_PI = 2.0 * math.pi


def _angle_delta(a: float, b: float) -> float:
    d = (a - b) % _TWO_PI
    if d > math.pi:
        d -= _TWO_PI
    return d


def _angle_distance(a: float, b: float) -> float:
    return abs(_angle_delta(a, b))


def _safe_float(x) -> float:
    return float(np.asarray(x))


@dataclass
class Entry:
    mode: Mode
    params: Dict[str, float]          # physical inputs
    solution: List[float]             # length-6 shooting vector
    features: List[float]             # cached feature vector (dimensionless)
    notes: str = ""                   # optional free text

    def to_jsonable(self):
        return asdict(self)

    @staticmethod
    def from_jsonable(d: Dict) -> "Entry":
        return Entry(
            mode=d["mode"],
            params=dict(d["params"]),
            solution=list(d["solution"]),
            features=list(d["features"]),
            notes=d.get("notes", "")
        )


class SolutionLibrary:
    def __init__(self):
        self._entries: List[Entry] = []

    # ---------------- Feature design ----------------

    def _compute_features(self, mode: Mode, params: Dict[str, float]) -> np.ndarray:
        r0 = _safe_float(params["r_0"])
        mu = _safe_float(params["mu_central_body"])
        omega = _safe_float(params["rotational_speed_central_body"])
        a0 = _safe_float(params["a_0"])
        raan = _safe_float(params["raan"])
        inc = _safe_float(params["inclination"])
        arglat = _safe_float(params["arg_of_latitude"])

        v0 = math.sqrt(mu / r0)
        T0 = 2.0 * math.pi * math.sqrt(r0**3 / mu)
        eps = a0 * r0 / (v0**2 + 1e-30)

        if mode == "true_anomaly":
            final_span = _safe_float(params["theta_f"])      # radians
            final_span_dimless = final_span % _TWO_PI
        else:
            time_f = _safe_float(params["time_f"])
            final_span_dimless = time_f / (T0 + 1e-30)       # normalized time span

        log_r0 = math.log10(r0 + 1e-30)
        log_mu = math.log10(mu + 1e-30)
        log_omega = math.log10(abs(omega) + 1e-30)
        log_a0 = math.log10(a0 + 1e-30)

        return np.array([
            eps,
            inc % _TWO_PI,
            raan % _TWO_PI,
            arglat % _TWO_PI,
            final_span_dimless,
            log_r0, log_mu, log_omega, log_a0
        ], dtype=np.float64)

    def _distance(self, mode: Mode, f_a: np.ndarray, f_b: np.ndarray) -> float:
        IDX_EPS = 0
        IDX_INC = 1
        IDX_RAAN = 2
        IDX_ARGLAT = 3
        IDX_FINAL = 4
        IDX_LOG_R0 = 5
        IDX_LOG_MU = 6
        IDX_LOG_OMEGA = 7
        IDX_LOG_A0 = 8

        d_eps = f_a[IDX_EPS] - f_b[IDX_EPS]
        d_inc = _angle_distance(f_a[IDX_INC], f_b[IDX_INC])
        d_raan = _angle_distance(f_a[IDX_RAAN], f_b[IDX_RAAN])
        d_arglat = _angle_distance(f_a[IDX_ARGLAT], f_b[IDX_ARGLAT])
        d_log_r0 = f_a[IDX_LOG_R0] - f_b[IDX_LOG_R0]
        d_log_mu = f_a[IDX_LOG_MU] - f_b[IDX_LOG_MU]
        d_log_omega = f_a[IDX_LOG_OMEGA] - f_b[IDX_LOG_OMEGA]
        d_log_a0 = f_a[IDX_LOG_A0] - f_b[IDX_LOG_A0]

        if mode == "true_anomaly":
            d_final = _angle_distance(f_a[IDX_FINAL], f_b[IDX_FINAL])
        else:
            d_final = f_a[IDX_FINAL] - f_b[IDX_FINAL]

        w_eps = 3.0
        w_inc = 1.5
        w_raan = 0.5
        w_arglat = 0.5
        w_final = 2.0
        w_log_r0 = 0.8
        w_log_mu = 0.2
        w_log_omega = 0.2
        w_log_a0 = 0.8

        return math.sqrt(
            (w_eps * d_eps) ** 2
            + (w_inc * d_inc) ** 2
            + (w_raan * d_raan) ** 2
            + (w_arglat * d_arglat) ** 2
            + (w_final * d_final) ** 2
            + (w_log_r0 * d_log_r0) ** 2
            + (w_log_mu * d_log_mu) ** 2
            + (w_log_omega * d_log_omega) ** 2
            + (w_log_a0 * d_log_a0) ** 2
        )

    # ---------------- Core library API ----------------

    def add_solution(
        self,
        mode: Mode,
        params: Dict[str, float],
        solution_vector: np.ndarray,
        notes: str = ""
    ) -> None:
        f = self._compute_features(mode, params)
        entry = Entry(
            mode=mode,
            params=dict(params),
            solution=[float(x) for x in np.asarray(solution_vector).ravel().tolist()],
            features=f.astype(np.float64).ravel().tolist(),
            notes=notes
        )
        self._entries.append(entry)

    def find_nearest(
        self,
        mode: Mode,
        params: Dict[str, float],
        k: int = 1
    ) -> List[Tuple[Entry, float]]:
        query_f = self._compute_features(mode, params)
        scored: List[Tuple[Entry, float]] = []
        for e in self._entries:
            if e.mode != mode:
                continue
            d = self._distance(mode, query_f, np.asarray(e.features))
            scored.append((e, d))
        scored.sort(key=lambda t: t[1])
        return scored[:k]

    def suggest_initial_guess(
        self,
        mode: Mode,
        params: Dict[str, float],
        k: int = 1,
        blend: bool = True
    ) -> Optional[np.ndarray]:
        neighbors = self.find_nearest(mode, params, k=k)
        if not neighbors:
            return np.full(6, -2, dtype=np.float64)  # flag value for "no guess"
        if not blend or k == 1:
            return np.asarray(neighbors[0][0].solution, dtype=np.float64)

        sols = []
        weights = []
        for e, d in neighbors:
            w = 1.0 / max(d, 1e-12)
            sols.append(np.asarray(e.solution, dtype=np.float64))
            weights.append(w)
        W = np.asarray(weights)
        W /= W.sum()
        S = np.vstack(sols)
        return (W[:, None] * S).sum(axis=0)

    # ---------------- Validation helpers ----------------

    @staticmethod
    def compute_residual_norm(problem, s_vector: np.ndarray) -> float:
        """Return 2-norm of shooting residuals at s_vector (no solve)."""
        s_jax = jnp.array(np.asarray(s_vector), dtype=jnp.float64)
        res = problem.shooting_residuals(s_jax)
        return float(jnp.linalg.norm(res))

    def add_solution_validated(
        self,
        mode: Mode,
        params: Dict[str, float],
        candidate_solution: np.ndarray,
        problem,                                # an instance of your problem class
        tol_residual: float = 1e-10,
        try_refine_with_jaxopt: bool = True,
        notes: str = ""
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Validate a candidate by checking residual norm and (optionally) running a
        jaxopt solve starting from the candidate. Store only if acceptable.

        Returns (accepted, info) where info has:
          - "residual_initial"
          - "residual_final"
          - "solver_success" (0 or 1)
          - "num_evals" (if available)
        """
        info: Dict[str, float] = {
            "residual_initial": float("nan"),
            "residual_final": float("nan"),
            "solver_success": 0.0,
            "num_evals": float("nan"),
        }

        # 1) Check residual at candidate
        r0 = self.compute_residual_norm(problem, candidate_solution)
        info["residual_initial"] = r0

        if r0 <= tol_residual:
            # Accept as-is
            self.add_solution(mode, params, candidate_solution, notes=notes)
            info["residual_final"] = r0
            info["solver_success"] = 1.0
            info["num_evals"] = 0.0
            return True, info

        # 2) Optionally refine using your jaxopt wrapper
        if try_refine_with_jaxopt:
            try:
                s_refined, rfin, nfev, success = problem.optimize_jaxopt(
                    np.asarray(candidate_solution), tol_jaxopt=tol_residual
                )
                info["residual_final"] = float(rfin)
                info["solver_success"] = 1.0 if bool(success) else 0.0
                info["num_evals"] = float(nfev)

                if bool(success) or float(rfin) <= tol_residual:
                    # Store the refined solution
                    self.add_solution(
                        mode, params, np.asarray(s_refined), notes=notes or "refined"
                    )
                    return True, info
            except Exception:
                # Fall through to rejection
                pass

        # 3) Reject
        if math.isnan(info["residual_final"]):
            info["residual_final"] = r0
        return False, info

    # ---------------- Persistence ----------------

    def save(self, path: str) -> None:
        payload = {"entries": [e.to_jsonable() for e in self._entries]}
        text = json.dumps(payload)
        np.savez_compressed(path, json_text=np.array(text))

    @staticmethod
    def load(path: str) -> "SolutionLibrary":
        lib = SolutionLibrary()
        data = np.load(path, allow_pickle=False)
        text = str(data["json_text"])
        payload = json.loads(text)
        for ed in payload.get("entries", []):
            lib._entries.append(Entry.from_jsonable(ed))
        return lib

    def __len__(self) -> int:
        return len(self._entries)
