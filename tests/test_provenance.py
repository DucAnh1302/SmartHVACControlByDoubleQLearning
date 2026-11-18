"""
Unit tests for Data Provenance Registry and Traceability Framework.
Ensures every sensor, parameter, and assumption adheres to rigorous scientific governance.
"""

import pytest
from src.data.provenance import (
    ProvenanceType,
    ProvenanceRecord,
    SYSTEM_PROVENANCE_REGISTRY,
    get_provenance,
    export_provenance_table_md
)


def test_provenance_types_coverage():
    """Verify all standard provenance categories exist."""
    assert ProvenanceType.OBSERVED == "OBSERVED"
    assert ProvenanceType.DERIVED == "DERIVED"
    assert ProvenanceType.EXTERNAL == "EXTERNAL"
    assert ProvenanceType.SIMULATED_ASSUMED == "SIMULATED_ASSUMED"
    assert ProvenanceType.FITTED_WITH_CAVEAT == "FITTED_WITH_CAVEAT"


def test_registry_contains_mandatory_variables():
    """Verify critical variables are registered with explicit provenance."""
    mandatory_vars = [
        "temp_c", "humidity_rh", "light_lux", "plug_kw", "light_kw", "hvac_kw",
        "internal_heat_kw", "occupancy_proxy",
        "ambient_temp", "ambient_rh",
        "k_env", "k_int", "k_ac", "h_env", "h_T", "h_ac",
        "sim_ac_setpoint", "sim_window_open"
    ]
    for var in mandatory_vars:
        assert var in SYSTEM_PROVENANCE_REGISTRY, f"Variable {var} missing from registry"
        rec = SYSTEM_PROVENANCE_REGISTRY[var]
        assert isinstance(rec, ProvenanceRecord)
        assert rec.name
        assert rec.unit
        assert rec.confidence in ["HIGH", "MEDIUM", "LOW"]
        assert rec.provenance_type in ProvenanceType


def test_observed_sensors_have_high_confidence():
    """Physical observed sensors from CU-BEMS should have HIGH confidence."""
    observed_keys = ["temp_c", "humidity_rh", "plug_kw", "light_kw", "hvac_kw"]
    for k in observed_keys:
        rec = SYSTEM_PROVENANCE_REGISTRY[k]
        assert rec.provenance_type == ProvenanceType.OBSERVED
        assert rec.confidence == "HIGH"


def test_simulated_and_fitted_caveats_present():
    """Fitted parameters and simulated variables must carry explicit caveats."""
    assert SYSTEM_PROVENANCE_REGISTRY["sim_window_open"].provenance_type == ProvenanceType.SIMULATED_ASSUMED
    assert SYSTEM_PROVENANCE_REGISTRY["sim_window_open"].caveat is not None
    assert "actuator" in SYSTEM_PROVENANCE_REGISTRY["sim_window_open"].caveat.lower() or "simulated" in SYSTEM_PROVENANCE_REGISTRY["sim_window_open"].caveat.lower()

    assert SYSTEM_PROVENANCE_REGISTRY["k_env"].provenance_type == ProvenanceType.FITTED_WITH_CAVEAT
    assert SYSTEM_PROVENANCE_REGISTRY["k_env"].caveat is not None


def test_get_provenance_fallback():
    """Test get_provenance for registered and unregistered variables."""
    rec = get_provenance("temp_c")
    assert rec.name == "Indoor Temperature"

    unreg = get_provenance("non_existent_var")
    assert unreg.name == "non_existent_var"
    assert unreg.provenance_type == ProvenanceType.SIMULATED_ASSUMED
    assert unreg.confidence == "LOW"


def test_export_provenance_table_md():
    """Verify markdown table generation produces valid markdown with headers."""
    md = export_provenance_table_md()
    assert "| Variable / Parameter |" in md
    assert "| :--- |" in md
    assert "Indoor Temperature" in md
    assert "Simulated Window State" in md
