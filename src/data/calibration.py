"""
HVAC Equipment Calibration and Drift Detection Module.
Performs K-Means clustering on active compressor power to identify discrete operating stages.
Detects parameter drift across equipment, zones, and operational years.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.data.provenance import ProvenanceType


@dataclass
class EquipmentStageCalibration:
    domain_id: str
    year: int
    floor: int
    zone: str
    off_power_kw: float
    low_power_kw: float
    med_power_kw: float
    high_power_kw: float
    off_frequency: float
    total_samples: int
    active_samples: int
    cluster_centers: List[float]
    cluster_boundaries: List[float]
    inertia: float
    provenance: str = ProvenanceType.FITTED_WITH_CAVEAT.value

    @property
    def stage_power_map(self) -> Dict[int, float]:
        """Mapping from discrete action stage (0..3) to physical kW."""
        return {
            0: self.off_power_kw,
            1: self.low_power_kw,
            2: self.med_power_kw,
            3: self.high_power_kw
        }

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, filepath: Union[str, Path]) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "EquipmentStageCalibration":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class EquipmentCalibrator:
    """
    Fits discrete HVAC operating stages using K-Means clustering
    on active (non-zero) historical power readings.
    """

    def __init__(self, n_clusters: int = 3, off_threshold_kw: float = 1.0, seed: int = 42):
        self.n_clusters = n_clusters
        self.off_threshold_kw = off_threshold_kw
        self.seed = seed

    def calibrate_from_series(
        self,
        hvac_power_kw: pd.Series,
        domain_id: str = "custom",
        year: int = 2018,
        floor: int = 2,
        zone: str = "z1"
    ) -> EquipmentStageCalibration:
        """
        Calibrate 4 HVAC stages:
        Stage 0: OFF (0.0 kW)
        Stage 1: LOW (cluster center 1)
        Stage 2: MED (cluster center 2)
        Stage 3: HIGH (cluster center 3)
        """
        valid_power = hvac_power_kw.dropna()
        total_count = len(valid_power)
        if total_count == 0:
            raise ValueError("Cannot calibrate equipment with empty power series!")

        # Separate OFF state
        off_mask = valid_power <= self.off_threshold_kw
        off_count = int(off_mask.sum())
        off_freq = float(off_count / total_count)
        off_power = 0.0

        active_power = valid_power[~off_mask].values.reshape(-1, 1)
        active_count = len(active_power)

        if active_count < self.n_clusters * 10:
            # Fallback heuristic if insufficient active data
            low_p = float(np.quantile(active_power, 0.25)) if active_count > 0 else 12.0
            med_p = float(np.quantile(active_power, 0.50)) if active_count > 0 else 22.0
            high_p = float(np.quantile(active_power, 0.75)) if active_count > 0 else 32.0
            centers = [low_p, med_p, high_p]
            boundaries = [(low_p + med_p) / 2.0, (med_p + high_p) / 2.0]
            inertia = 0.0
        else:
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.seed,
                n_init=10
            ).fit(active_power)

            centers = sorted([float(c[0]) for c in kmeans.cluster_centers_])
            low_p, med_p, high_p = centers[0], centers[1], centers[2]

            # Midpoint decision boundaries
            boundaries = [
                (low_p + med_p) / 2.0,
                (med_p + high_p) / 2.0
            ]
            inertia = float(kmeans.inertia_)

        return EquipmentStageCalibration(
            domain_id=domain_id,
            year=year,
            floor=floor,
            zone=zone,
            off_power_kw=round(off_power, 2),
            low_power_kw=round(low_p, 2),
            med_power_kw=round(med_p, 2),
            high_power_kw=round(high_p, 2),
            off_frequency=round(off_freq, 4),
            total_samples=total_count,
            active_samples=active_count,
            cluster_centers=[round(c, 2) for c in centers],
            cluster_boundaries=[round(b, 2) for b in boundaries],
            inertia=round(inertia, 2)
        )


class EquipmentDriftDetector:
    """
    Detects physical drift and discrepancies between two calibrated equipment profiles.
    """

    @staticmethod
    def compare_calibrations(
        baseline: EquipmentStageCalibration,
        target: EquipmentStageCalibration,
        drift_alert_pct: float = 15.0
    ) -> Dict[str, Union[float, bool, str]]:
        """
        Compare baseline and target calibrations to check for equipment drift.
        """
        low_shift_pct = abs(target.low_power_kw - baseline.low_power_kw) / baseline.low_power_kw * 100.0
        med_shift_pct = abs(target.med_power_kw - baseline.med_power_kw) / baseline.med_power_kw * 100.0
        high_shift_pct = abs(target.high_power_kw - baseline.high_power_kw) / baseline.high_power_kw * 100.0
        max_shift_pct = max(low_shift_pct, med_shift_pct, high_shift_pct)

        has_drift = max_shift_pct >= drift_alert_pct
        off_freq_diff = abs(target.off_frequency - baseline.off_frequency)

        return {
            "baseline_domain": baseline.domain_id,
            "target_domain": target.domain_id,
            "low_shift_pct": round(low_shift_pct, 2),
            "med_shift_pct": round(med_shift_pct, 2),
            "high_shift_pct": round(high_shift_pct, 2),
            "max_shift_pct": round(max_shift_pct, 2),
            "off_freq_diff": round(off_freq_diff, 4),
            "has_significant_drift": bool(has_drift),
            "recommendation": (
                "CALIBRATION_REQUIRED: Target domain exhibits physical stage shifts. Direct zero-shot transfer will distort energy accounting."
                if has_drift else
                "ZERO_SHOT_ACCEPTABLE: Equipment stages are sufficiently aligned."
            )
        }
