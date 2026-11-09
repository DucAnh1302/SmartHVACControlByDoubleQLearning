"""
Domain Configuration Module for Smart HVAC Control.
Defines domain specifications across years (2018, 2019), floors (Floor 2, 3), and zones (Zone 1, 2).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from pathlib import Path


@dataclass(frozen=True)
class DomainSpec:
    domain_id: str
    year: int
    floor: int
    zone: str
    is_primary: bool
    description: str
    ac_cols: Tuple[str, ...]   # Tuple of AC columns to sum for this zone
    temp_col: str
    rh_col: str
    lux_col: str
    plug_col: str
    light_col: str


# Predefined domain catalog
DOMAINS = {
    # 1. PRIMARY TRAINING DOMAIN (2018 Floor 2 Zone 1)
    "2018_F2_Z1": DomainSpec(
        domain_id="2018_F2_Z1",
        year=2018,
        floor=2,
        zone="z1",
        is_primary=True,
        description="Primary Training Domain: 2018 Floor 2 Zone 1 (High quality sensor span Jul-Sep 2018)",
        ac_cols=("z1_AC1(kW)",),
        temp_col="z1_S1(degC)",
        rh_col="z1_S1(RH%)",
        lux_col="z1_S1(lux)",
        plug_col="z1_Plug(kW)",
        light_col="z1_Light(kW)"
    ),

    # 2. SECONDARY CROSS-YEAR DOMAIN (2019 Floor 2 Zone 1)
    "2019_F2_Z1": DomainSpec(
        domain_id="2019_F2_Z1",
        year=2019,
        floor=2,
        zone="z1",
        is_primary=False,
        description="Secondary Cross-Year Domain: 2019 Floor 2 Zone 1 (Equipment shifted stages)",
        ac_cols=("z1_AC1(kW)",),
        temp_col="z1_S1(degC)",
        rh_col="z1_S1(RH%)",
        lux_col="z1_S1(lux)",
        plug_col="z1_Plug(kW)",
        light_col="z1_Light(kW)"
    ),

    # 3. CROSS-ZONE DOMAIN (2018 Floor 2 Zone 2)
    "2018_F2_Z2": DomainSpec(
        domain_id="2018_F2_Z2",
        year=2018,
        floor=2,
        zone="z2",
        is_primary=False,
        description="Cross-Zone Domain: 2018 Floor 2 Zone 2 (Large open-plan zone with multi-unit HVAC)",
        ac_cols=tuple(f"z2_AC{i}(kW)" for i in range(1, 15)),
        temp_col="z2_S1(degC)",
        rh_col="z2_S1(RH%)",
        lux_col="z2_S1(lux)",
        plug_col="z2_Plug(kW)",
        light_col="z2_Light(kW)"
    ),

    # 4. CROSS-FLOOR DOMAIN (2018 Floor 3 Zone 1)
    "2018_F3_Z1": DomainSpec(
        domain_id="2018_F3_Z1",
        year=2018,
        floor=3,
        zone="z1",
        is_primary=False,
        description="Cross-Floor Domain: 2018 Floor 3 Zone 1 (Different architectural floor slab)",
        ac_cols=("z1_AC1(kW)", "z1_AC2(kW)", "z1_AC3(kW)", "z1_AC4(kW)"),
        temp_col="z1_S1(degC)",
        rh_col="z1_S1(RH%)",
        lux_col="z1_S1(lux)",
        plug_col="z1_Plug(kW)",
        light_col="z1_Light(kW)"
    ),

    # 5. CROSS-FLOOR CROSS-YEAR DOMAIN (2019 Floor 3 Zone 1)
    "2019_F3_Z1": DomainSpec(
        domain_id="2019_F3_Z1",
        year=2019,
        floor=3,
        zone="z1",
        is_primary=False,
        description="Cross-Floor Cross-Year Domain: 2019 Floor 3 Zone 1",
        ac_cols=("z1_AC1(kW)", "z1_AC2(kW)", "z1_AC3(kW)", "z1_AC4(kW)"),
        temp_col="z1_S1(degC)",
        rh_col="z1_S1(RH%)",
        lux_col="z1_S1(lux)",
        plug_col="z1_Plug(kW)",
        light_col="z1_Light(kW)"
    )
}


def get_domain_spec(domain_id: str) -> DomainSpec:
    """Retrieve DomainSpec by identifier."""
    if domain_id not in DOMAINS:
        raise KeyError(f"Domain '{domain_id}' not found! Available domains: {list(DOMAINS.keys())}")
    return DOMAINS[domain_id]
