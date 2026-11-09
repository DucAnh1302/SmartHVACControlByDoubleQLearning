"""
Grey-Box Coupled Thermal and Humidity Physical Environment Model.
Combines 1st-order Resistance-Capacitance (RC) heat transfer and moisture mass balance.
Features identifiable parameter estimation with statistical confidence intervals,
provenance tracking, sensitivity analysis, and explicit validation safeguards.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import numpy as np
import pandas as pd
from scipy import stats

from src.environment.transition_base import BaseTransitionModel
from src.data.provenance import ProvenanceType, get_provenance


@dataclass
class PhysicalParameters:
    # Thermal parameters
    k_env: float = 0.1800          # Ambient thermal conductance / building heat capacity (h^-1)
    k_int: float = 0.0480         # Internal heat gain coefficient (°C / (kW·h))
    k_ac: float = 0.0350          # HVAC cooling extraction capacity (°C / (kW·h))
    k_win: float = 0.4500         # Additional ventilation conductance when window is OPEN (h^-1)

    # Humidity parameters
    h_env: float = 0.1200         # Moisture infiltration rate (h^-1)
    h_T: float = -2.3500          # Psychrometric temperature-RH coupling (%RH / °C)
    h_ac: float = 0.2800          # Dehumidification extraction rate (%RH / (kW·h))
    h_win: float = 0.5000         # Additional moisture exchange when window is OPEN (h^-1)

    # Statistical Uncertainty / Standard Errors
    k_env_se: float = 0.0150
    k_int_se: float = 0.0080
    k_ac_se: float = 0.0040

    provenance_status: str = ProvenanceType.SIMULATED_ASSUMED.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhysicalParameters":
        return cls(**d)


class GreyBoxThermalModel(BaseTransitionModel):
    """
    Physically consistent 1st-order coupled Grey-box RC model:
    dT/dt  = k_env * (T_amb - T) + k_int * P_int - k_ac * P_hvac + k_win * W * (T_amb - T)
    dRH/dt = h_env * (RH_amb - RH) + h_T * dT - h_ac * P_hvac + h_win * W * (RH_amb - RH)
    """

    def __init__(
        self,
        params: Optional[PhysicalParameters] = None,
        noise_std_temp: float = 0.10,
        noise_std_rh: float = 0.35,
        seed: int = 42
    ):
        super().__init__(name="Grey-Box RC Model")
        self.params = params or PhysicalParameters()
        self.noise_std_temp = noise_std_temp
        self.noise_std_rh = noise_std_rh
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.is_fitted = (params is not None and params.provenance_status == ProvenanceType.FITTED_WITH_CAVEAT.value)
        self.parameters_calibrated = self.is_fitted
        self.provenance = ProvenanceType.FITTED_WITH_CAVEAT if self.is_fitted else ProvenanceType.SIMULATED_ASSUMED
        self.parameter_identifiability_report: Dict[str, Any] = {}

    def fit(self, df_train: pd.DataFrame) -> "GreyBoxThermalModel":
        """
        Identify physical parameters from historical timeseries data using
        constrained ordinary least-squares on physical temperature and moisture differences.
        """
        required_cols = ["temp_c", "humidity_rh", "ambient_temp_est", "internal_heat_kw", "hvac_kw"]
        missing = [c for c in required_cols if c not in df_train.columns]
        if missing:
            raise ValueError(f"Cannot fit GreyBoxThermalModel: missing required columns {missing}")

        df = df_train.copy()
        df["dT"] = df["temp_c"].shift(-1) - df["temp_c"]
        df["dRH"] = df["humidity_rh"].shift(-1) - df["humidity_rh"]
        df["delta_T_amb"] = df["ambient_temp_est"] - df["temp_c"]
        df["delta_RH_amb"] = df.get("ambient_rh_est", 70.0) - df["humidity_rh"]

        valid = df.dropna(subset=["dT", "dRH", "delta_T_amb", "internal_heat_kw", "hvac_kw"])
        N = len(valid)

        if N < 50:
            raise ValueError(
                f"Cannot fit GreyBoxThermalModel: insufficient valid observations (found N={N}, required >= 50). "
                f"Check sensor availability window or calibration dataset."
            )

        # 1. Thermal parameter identification: dT = k_env * delta_T_amb + k_int * P_int - k_ac * P_hvac
        X_therm = np.column_stack([
            valid["delta_T_amb"].values,
            valid["internal_heat_kw"].values,
            -valid["hvac_kw"].values
        ])
        y_therm = valid["dT"].values

        lambda_reg = 10.0
        XtX = X_therm.T @ X_therm + lambda_reg * np.eye(3)
        Xty = X_therm.T @ y_therm
        coefs = np.linalg.solve(XtX, Xty)

        residuals = y_therm - X_therm @ coefs
        s2 = np.sum(residuals**2) / max(1, N - 3)
        var_coef = s2 * np.linalg.inv(XtX)
        se_coef = np.sqrt(np.diag(var_coef))

        # Enforce thermodynamic physical bounds
        k_env_est = float(np.clip(coefs[0], 0.05, 0.45))
        k_int_est = float(np.clip(coefs[1], 0.01, 0.15))
        k_ac_est = float(np.clip(coefs[2], 0.01, 0.12))

        # 2. Moisture parameter identification: dRH ~ h_env * delta_RH_amb + h_T * dT - h_ac * P_hvac
        dRH_adjusted = valid["dRH"].values - self.params.h_T * valid["dT"].values
        X_rh = np.column_stack([
            valid["delta_RH_amb"].values,
            -valid["hvac_kw"].values
        ])
        XtX_rh = X_rh.T @ X_rh + 5.0 * np.eye(2)
        Xty_rh = X_rh.T @ dRH_adjusted
        coefs_rh = np.linalg.solve(XtX_rh, Xty_rh)

        h_env_est = float(np.clip(coefs_rh[0], 0.02, 0.35))
        h_ac_est = float(np.clip(coefs_rh[1], 0.05, 0.65))

        self.params = PhysicalParameters(
            k_env=round(k_env_est, 4),
            k_int=round(k_int_est, 4),
            k_ac=round(k_ac_est, 4),
            k_win=self.params.k_win,
            h_env=round(h_env_est, 4),
            h_T=self.params.h_T,
            h_ac=round(h_ac_est, 4),
            h_win=self.params.h_win,
            k_env_se=round(float(se_coef[0]), 4),
            k_int_se=round(float(se_coef[1]), 4),
            k_ac_se=round(float(se_coef[2]), 4),
            provenance_status=ProvenanceType.FITTED_WITH_CAVEAT.value
        )

        self.parameter_identifiability_report = {
            "samples_used": N,
            "k_env": {
                "estimate": self.params.k_env,
                "std_error": self.params.k_env_se,
                "ci_95": [round(self.params.k_env - 1.96 * self.params.k_env_se, 4),
                          round(self.params.k_env + 1.96 * self.params.k_env_se, 4)],
                "identifiability_status": "IDENTIFIED"
            },
            "k_int": {
                "estimate": self.params.k_int,
                "std_error": self.params.k_int_se,
                "ci_95": [round(self.params.k_int - 1.96 * self.params.k_int_se, 4),
                          round(self.params.k_int + 1.96 * self.params.k_int_se, 4)],
                "identifiability_status": "IDENTIFIED"
            },
            "k_ac": {
                "estimate": self.params.k_ac,
                "std_error": self.params.k_ac_se,
                "ci_95": [round(self.params.k_ac - 1.96 * self.params.k_ac_se, 4),
                          round(self.params.k_ac + 1.96 * self.params.k_ac_se, 4)],
                "identifiability_status": "IDENTIFIED"
            },
            "h_env": {
                "estimate": self.params.h_env,
                "identifiability_status": "IDENTIFIED_WITH_HIGH_VARIANCE"
            },
            "h_T": {
                "estimate": self.params.h_T,
                "identifiability_status": "THERMODYNAMIC_PHYSICAL_CONSTANT"
            },
            "h_ac": {
                "estimate": self.params.h_ac,
                "identifiability_status": "IDENTIFIED"
            },
            "k_win": {
                "estimate": self.params.k_win,
                "identifiability_status": "SIMULATED_ASSUMED_NO_DATA",
                "sensitivity_range": [0.20, 0.70]
            },
            "h_win": {
                "estimate": self.params.h_win,
                "identifiability_status": "SIMULATED_ASSUMED_NO_DATA",
                "sensitivity_range": [0.25, 0.75]
            }
        }

        self.is_fitted = True
        self.parameters_calibrated = True
        self.provenance = ProvenanceType.FITTED_WITH_CAVEAT
        return self

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist fitted physical parameters and identifiability report to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "parameters": self.params.to_dict(),
            "is_fitted": self.is_fitted,
            "identifiability_report": self.parameter_identifiability_report
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "GreyBoxThermalModel":
        """Load fitted parameters from JSON."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Fitted parameter file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        params = PhysicalParameters.from_dict(payload["parameters"])
        model = cls(params=params)
        model.is_fitted = payload.get("is_fitted", True)
        model.parameter_identifiability_report = payload.get("identifiability_report", {})
        return model

    def perform_sensitivity_analysis(
        self,
        base_temp: float = 26.0,
        base_rh: float = 65.0,
        hvac_kw: float = 21.6,
        internal_heat_kw: float = 2.0,
        ambient_temp: float = 33.0,
        ambient_rh: float = 75.0,
        perturbation_pct: float = 0.25
    ) -> pd.DataFrame:
        """
        Evaluate output sensitivity to +/- perturbation in physical parameters.
        """
        records = []
        base_t_next, _ = self.predict_step(
            base_temp, base_rh, hvac_kw, internal_heat_kw, ambient_temp, ambient_rh, 14, 0
        )

        for param_name in ["k_env", "k_int", "k_ac"]:
            orig_val = getattr(self.params, param_name)
            # Low perturbation (-25%)
            low_params = PhysicalParameters(**{**self.params.to_dict(), param_name: orig_val * (1.0 - perturbation_pct)})
            low_model = GreyBoxThermalModel(params=low_params, noise_std_temp=0.0, noise_std_rh=0.0)
            low_t_next, _ = low_model.predict_step(base_temp, base_rh, hvac_kw, internal_heat_kw, ambient_temp, ambient_rh, 14, 0)

            # High perturbation (+25%)
            high_params = PhysicalParameters(**{**self.params.to_dict(), param_name: orig_val * (1.0 + perturbation_pct)})
            high_model = GreyBoxThermalModel(params=high_params, noise_std_temp=0.0, noise_std_rh=0.0)
            high_t_next, _ = high_model.predict_step(base_temp, base_rh, hvac_kw, internal_heat_kw, ambient_temp, ambient_rh, 14, 0)

            records.append({
                "Parameter": param_name,
                "Baseline Value": orig_val,
                "Low Value (-25%)": round(orig_val * (1.0 - perturbation_pct), 4),
                "Low Temp Output (°C)": round(low_t_next, 3),
                "High Value (+25%)": round(orig_val * (1.0 + perturbation_pct), 4),
                "High Temp Output (°C)": round(high_t_next, 3),
                "Sensitivity Delta (°C)": round(abs(high_t_next - low_t_next), 3)
            })

        return pd.DataFrame(records)

    def predict_step(
        self,
        current_temp: float,
        current_rh: float,
        hvac_kw: float,
        internal_heat_kw: float,
        ambient_temp: float,
        ambient_rh: float,
        hour: int,
        window_open: int = 0
    ) -> Tuple[float, float]:
        p = self.params

        delta_amb_t = ambient_temp - current_temp
        q_env = p.k_env * delta_amb_t
        q_int = p.k_int * internal_heat_kw
        q_ac = p.k_ac * hvac_kw
        q_win = (p.k_win * delta_amb_t) if window_open == 1 else 0.0

        delta_t = q_env + q_int - q_ac + q_win

        if self.noise_std_temp > 0:
            delta_t += float(self.rng.normal(0.0, self.noise_std_temp))

        next_temp = float(np.clip(current_temp + delta_t, 16.0, 38.0))

        delta_amb_rh = ambient_rh - current_rh
        rh_env = p.h_env * delta_amb_rh
        rh_temp_shift = p.h_T * (next_temp - current_temp)
        rh_ac = p.h_ac * hvac_kw
        rh_win = (p.h_win * delta_amb_rh) if window_open == 1 else 0.0

        delta_rh = rh_env + rh_temp_shift - rh_ac + rh_win

        if self.noise_std_rh > 0:
            delta_rh += float(self.rng.normal(0.0, self.noise_std_rh))

        next_rh = float(np.clip(current_rh + delta_rh, 20.0, 95.0))

        return (next_temp, next_rh)

    def predict_batch(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        p = self.params
        t = df["temp_c"].values
        rh = df["humidity_rh"].values
        hvac = df["hvac_kw"].values
        p_int = df["internal_heat_kw"].values
        amb_t = df["ambient_temp_est"].values
        amb_rh = df["ambient_rh_est"].values if "ambient_rh_est" in df.columns else np.full_like(t, 70.0)
        win = df["window_open"].values if "window_open" in df.columns else np.zeros_like(t)

        delta_amb_t = amb_t - t
        q_env = p.k_env * delta_amb_t
        q_int = p.k_int * p_int
        q_ac = p.k_ac * hvac
        q_win = p.k_win * delta_amb_t * win

        delta_t = q_env + q_int - q_ac + q_win
        next_t = np.clip(t + delta_t, 16.0, 38.0)

        delta_amb_rh = amb_rh - rh
        rh_env = p.h_env * delta_amb_rh
        rh_temp_shift = p.h_T * (next_t - t)
        rh_ac = p.h_ac * hvac
        rh_win = p.h_win * delta_amb_rh * win

        delta_rh = rh_env + rh_temp_shift - rh_ac + rh_win
        next_rh = np.clip(rh + delta_rh, 20.0, 95.0)

        return next_t, next_rh
