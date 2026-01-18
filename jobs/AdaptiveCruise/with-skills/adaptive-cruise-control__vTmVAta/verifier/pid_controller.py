from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


Number = float


@dataclass
class PIDGains:
    kp: Number
    ki: Number
    kd: Number


class PIDController:
    def __init__(
        self,
        kp: Number,
        ki: Number,
        kd: Number,
        *,
        output_limits: Tuple[Optional[Number], Optional[Number]] = (None, None),
        integrator_limits: Tuple[Optional[Number], Optional[Number]] = (None, None),
        derivative_filter_tau: Optional[Number] = None,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        self.output_limits = output_limits
        self.integrator_limits = integrator_limits
        self.derivative_filter_tau = derivative_filter_tau

        self.reset()

    def reset(self) -> None:
        self._integral: Number = 0.0
        self._prev_error: Optional[Number] = None
        self._prev_derivative: Number = 0.0

    def compute(self, error: Number, dt: Number) -> Number:
        if dt <= 0:
            raise ValueError("dt must be > 0")

        error = float(error)
        dt = float(dt)

        p = self.kp * error

        self._integral += error * dt
        self._integral = self._clamp(self._integral, *self.integrator_limits)
        i = self.ki * self._integral

        if self._prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self._prev_error) / dt

        if self.derivative_filter_tau is not None and self.derivative_filter_tau > 0:
            alpha = self.derivative_filter_tau / (self.derivative_filter_tau + dt)
            derivative = alpha * self._prev_derivative + (1.0 - alpha) * derivative

        d = self.kd * derivative

        output = p + i + d
        output_clamped = self._clamp(output, *self.output_limits)

        if self.ki != 0.0 and output_clamped != output:
            self._integral = (output_clamped - p - d) / self.ki
            self._integral = self._clamp(self._integral, *self.integrator_limits)
            i = self.ki * self._integral
            output_clamped = self._clamp(p + i + d, *self.output_limits)

        self._prev_error = error
        self._prev_derivative = derivative

        return output_clamped

    @staticmethod
    def _clamp(value: Number, lo: Optional[Number], hi: Optional[Number]) -> Number:
        if lo is not None and value < lo:
            return float(lo)
        if hi is not None and value > hi:
            return float(hi)
        return float(value)

