from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from pid_controller import PIDController


class AdaptiveCruiseControl:
    def __init__(self, config: Dict[str, Any]) -> None:
        vehicle = config.get("vehicle", {})
        acc_settings = config.get("acc_settings", {})

        self.max_acceleration = float(vehicle.get("max_acceleration", 3.0))
        self.max_deceleration = float(vehicle.get("max_deceleration", -8.0))

        self.set_speed = float(acc_settings.get("set_speed", 30.0))
        self.time_headway = float(acc_settings.get("time_headway", 1.5))
        self.min_distance = float(acc_settings.get("min_distance", 10.0))
        self.emergency_ttc_threshold = float(
            acc_settings.get("emergency_ttc_threshold", 3.0)
        )

        pid_speed_cfg = config.get("pid_speed", {})
        pid_distance_cfg = config.get("pid_distance", {})

        self.speed_pid = PIDController(
            pid_speed_cfg.get("kp", 0.1),
            pid_speed_cfg.get("ki", 0.01),
            pid_speed_cfg.get("kd", 0.0),
            output_limits=(self.max_deceleration, self.max_acceleration),
            integrator_limits=(-1000.0, 1000.0),
            derivative_filter_tau=0.2,
        )

        self.distance_pid = PIDController(
            pid_distance_cfg.get("kp", 0.1),
            pid_distance_cfg.get("ki", 0.01),
            pid_distance_cfg.get("kd", 0.0),
            # Interpreted as speed adjustment (m/s), not acceleration.
            output_limits=(-self.set_speed, self.set_speed),
            integrator_limits=(-200.0, 200.0),
            derivative_filter_tau=0.5,
        )

        self._mode: str = "cruise"

    def compute(
        self,
        ego_speed: float,
        lead_speed: Optional[float],
        distance: Optional[float],
        dt: float,
    ) -> Tuple[float, str, Optional[float]]:
        lead_present = (
            lead_speed is not None
            and distance is not None
            and not (math.isnan(lead_speed) or math.isnan(distance))
        )

        ttc: Optional[float] = None
        if lead_present and distance is not None and lead_speed is not None:
            if ego_speed > lead_speed and distance > 0:
                ttc = distance / (ego_speed - lead_speed)

        if lead_present and ttc is not None and ttc < self.emergency_ttc_threshold:
            mode = "emergency"
        elif lead_present:
            mode = "follow"
        else:
            mode = "cruise"

        if mode != self._mode:
            self.speed_pid.reset()
            self.distance_pid.reset()
            self._mode = mode

        if mode == "cruise":
            speed_error = self.set_speed - ego_speed
            acceleration_cmd = self.speed_pid.compute(speed_error, dt)
            return float(acceleration_cmd), mode, None

        # Lead present: compute spacing error.
        safe_distance = ego_speed * self.time_headway + self.min_distance
        distance_error = float(distance) - safe_distance

        if mode == "emergency":
            return float(self.max_deceleration), mode, distance_error

        # FOLLOW: use distance PID to generate a target speed and track it with the speed PID.
        speed_adjust = self.distance_pid.compute(distance_error, dt)
        spacing_speed = max(0.0, (float(distance) - self.min_distance) / self.time_headway)
        target_speed = min(self.set_speed, spacing_speed, float(lead_speed) + speed_adjust)
        target_speed = max(0.0, target_speed)

        speed_error = target_speed - ego_speed
        acceleration_cmd = self.speed_pid.compute(speed_error, dt)

        return float(acceleration_cmd), mode, distance_error

