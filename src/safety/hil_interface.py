"""
Hardware-In-The-Loop (HIL) Interface Simulator for Building Management Systems (BMS).
Emulates standard BACnet MS/TP and Modbus TCP telemetry exchange and actuator setpoint commands.
"""

from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import time

from src.safety.safety_gateway import SafetyGateway, DeploymentReadinessLevel


@dataclass
class BMSTelemetryPacket:
    timestamp_epoch: float
    device_id: str
    indoor_temperature_c: float
    indoor_relative_humidity: float
    hvac_power_kw: float
    plug_power_kw: float
    light_power_kw: float
    outdoor_temperature_c: float
    compressor_status: str  # "RUNNING", "STANDBY", "FAULT"


@dataclass
class BMSCommandPacket:
    device_id: str
    target_setpoint_c: Optional[float]
    compressor_enable: bool
    window_actuator_enable: bool
    safety_interlock_verified: bool
    readiness_level: str = DeploymentReadinessLevel.HARDWARE_IN_LOOP_VALIDATED.value


class MockBMSControllerHIL:
    """
    Simulates a digital BMS substation controller (e.g. Siemens Desigo or Honeywell Alerton).
    """

    def __init__(self, device_id: str = "BMS_AHU_FL02_Z01"):
        self.device_id = device_id
        self.is_connected = False
        self.last_command: Optional[BMSCommandPacket] = None
        self.command_history = []

    def connect(self) -> bool:
        """Establish BACnet/IP mock socket connection."""
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        self.is_connected = False

    def send_control_packet(self, command: BMSCommandPacket) -> Dict[str, Any]:
        """Dispatch control command packet to virtual HVAC hardware."""
        if not self.is_connected:
            raise ConnectionError(f"Cannot send command: {self.device_id} is disconnected!")

        if not command.safety_interlock_verified:
            raise PermissionError("REJECTED: Command packet rejected by hardware controller because safety interlock was unverified!")

        self.last_command = command
        self.command_history.append(command)
        return {
            "status": "ACK",
            "device_id": self.device_id,
            "setpoint_applied": command.target_setpoint_c,
            "compressor_enabled": command.compressor_enable,
            "timestamp": time.time()
        }
