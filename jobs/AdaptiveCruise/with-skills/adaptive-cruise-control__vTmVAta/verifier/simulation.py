from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from acc_system import AdaptiveCruiseControl


CONFIG_PATH = "/root/vehicle_params.yaml"
SENSOR_DATA_PATH = "/root/sensor_data.csv"
TUNING_PATH = "/root/tuning_results.yaml"
OUTPUT_PATH = "/root/simulation_results.csv"


@dataclass(frozen=True)
class PidConfig:
    kp: float
    ki: float
    kd: float


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {path}")
    return data


def load_tuning(path: str) -> Optional[Dict[str, PidConfig]]:
    if not os.path.exists(path):
        return None

    data = load_yaml(path)
    tuning: Dict[str, PidConfig] = {}

    for section in ("pid_speed", "pid_distance"):
        if section not in data:
            raise ValueError(f"{path} missing required section: {section}")
        sec = data[section]
        if not isinstance(sec, dict):
            raise ValueError(f"{path} section {section} must be a mapping")
        try:
            kp = float(sec["kp"])
            ki = float(sec["ki"])
            kd = float(sec["kd"])
        except Exception as e:
            raise ValueError(f"{path} section {section} must have numeric kp/ki/kd") from e
        tuning[section] = PidConfig(kp=kp, ki=ki, kd=kd)

    return tuning


def apply_tuning(base_config: Dict[str, Any], tuning: Optional[Dict[str, PidConfig]]) -> Dict[str, Any]:
    if tuning is None:
        return base_config
    config = dict(base_config)
    for section, gains in tuning.items():
        config[section] = {"kp": gains.kp, "ki": gains.ki, "kd": gains.kd}
    return config


def simulate(config: Dict[str, Any], sensor_df: pd.DataFrame) -> pd.DataFrame:
    dt = float(config.get("simulation", {}).get("dt", 0.1))

    acc = AdaptiveCruiseControl(config)

    ego_speed = float(sensor_df.loc[0, "ego_speed"]) if "ego_speed" in sensor_df.columns else 0.0

    sim_distance: Optional[float] = None
    prev_lead_present = False

    rows = []

    for i in range(len(sensor_df)):
        t = float(sensor_df.at[i, "time"])
        lead_speed_raw = sensor_df.at[i, "lead_speed"]
        distance_raw = sensor_df.at[i, "distance"]

        lead_present = not (pd.isna(lead_speed_raw) or pd.isna(distance_raw))
        lead_speed = float(lead_speed_raw) if lead_present else None
        sensor_distance = float(distance_raw) if lead_present else None

        if lead_present and not prev_lead_present:
            sim_distance = sensor_distance
        elif not lead_present:
            sim_distance = None

        accel_cmd, mode, distance_error = acc.compute(
            ego_speed=ego_speed,
            lead_speed=lead_speed,
            distance=sim_distance,
            dt=dt,
        )

        max_acc = float(config.get("vehicle", {}).get("max_acceleration", 3.0))
        max_dec = float(config.get("vehicle", {}).get("max_deceleration", -8.0))
        accel_cmd = max(max_dec, min(max_acc, float(accel_cmd)))

        ttc: Optional[float] = None
        if lead_present and sim_distance is not None and lead_speed is not None:
            if ego_speed > lead_speed and sim_distance > 0:
                ttc_val = sim_distance / (ego_speed - lead_speed)
                if math.isfinite(ttc_val):
                    ttc = float(ttc_val)

        rows.append(
            {
                "time": t,
                "ego_speed": float(ego_speed),
                "acceleration_cmd": float(accel_cmd),
                "mode": mode,
                "distance_error": float(distance_error) if lead_present else math.nan,
                "distance": float(sim_distance) if lead_present and sim_distance is not None else math.nan,
                "ttc": float(ttc) if ttc is not None else math.nan,
            }
        )

        # Physics update (Euler integration).
        current_speed = ego_speed
        ego_speed = max(0.0, current_speed + accel_cmd * dt)

        if lead_present and sim_distance is not None and lead_speed is not None:
            sim_distance = max(0.0, sim_distance - (current_speed - lead_speed) * dt)

        prev_lead_present = lead_present

    return pd.DataFrame(rows, columns=["time", "ego_speed", "acceleration_cmd", "mode", "distance_error", "distance", "ttc"])


def summarize(results_df: pd.DataFrame, config: Dict[str, Any]) -> None:
    set_speed = float(config.get("acc_settings", {}).get("set_speed", 30.0))
    dt = float(config.get("simulation", {}).get("dt", 0.1))

    # Speed performance on t=0..30
    seg = results_df[(results_df["time"] >= 0.0) & (results_df["time"] <= 30.0)]
    t3 = seg.loc[seg["ego_speed"] >= 3.0, "time"].min()
    t27 = seg.loc[seg["ego_speed"] >= 27.0, "time"].min()
    rise = float(t27 - t3) if pd.notna(t3) and pd.notna(t27) else math.nan
    overshoot = float(seg["ego_speed"].max() - set_speed)
    ss = results_df[(results_df["time"] >= 25.0) & (results_df["time"] <= 30.0)]
    ss_err = float((set_speed - ss["ego_speed"]).abs().max()) if len(ss) else math.nan

    # Mode distribution
    def frac(t0: float, t1: float, mode: str) -> float:
        m = results_df[(results_df["time"] >= t0) & (results_df["time"] <= t1)]
        if len(m) == 0:
            return math.nan
        return float((m["mode"] == mode).mean())

    min_distance = float(results_df["distance"].min(skipna=True))
    min_ttc = float(results_df["ttc"].min(skipna=True))
    min_acc = float(results_df["acceleration_cmd"].min())
    max_acc = float(results_df["acceleration_cmd"].max())

    print(f"dt={dt:.3f}s  set_speed={set_speed:.2f} m/s")
    print(f"speed: rise(3->27)={rise:.2f}s  overshoot={overshoot:.2f} m/s  ss_err(25-30)={ss_err:.2f} m/s")
    print(
        "modes: "
        f"cruise[0-30]={frac(0,30,'cruise'):.2%}  "
        f"follow[30-60]={frac(30,60,'follow'):.2%}  "
        f"emergency[120-122]={frac(120,122,'emergency'):.2%}  "
        f"cruise[130-150]={frac(130,150,'cruise'):.2%}"
    )
    print(f"safety: min_distance={min_distance:.2f} m  min_ttc={min_ttc:.2f} s  accel_range=[{min_acc:.2f},{max_acc:.2f}] m/s^2")


def main() -> None:
    config = load_yaml(CONFIG_PATH)
    tuning = load_tuning(TUNING_PATH)
    config = apply_tuning(config, tuning)

    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    results_df = simulate(config, sensor_df)

    if len(results_df) != 1501:
        raise RuntimeError(f"Expected 1501 rows, got {len(results_df)}")

    results_df.to_csv(OUTPUT_PATH, index=False)
    summarize(results_df, config)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
