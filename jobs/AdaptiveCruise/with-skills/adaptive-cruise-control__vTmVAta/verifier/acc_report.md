# Adaptive Cruise Control (ACC) Simulation Report

## Overview
This project implements an Adaptive Cruise Control (ACC) simulation for a passenger vehicle using a discrete-time physics model (dt = 0.1 s) with acceleration saturation in \[-8.0, 3.0\] m/s².

## Control design
- **Cruise mode** (no lead vehicle): track the configured set speed using a PID speed controller.
- **Follow mode** (lead vehicle present): compute a safe following distance `safe = ego_speed * time_headway + min_distance`, then regulate spacing by generating a target speed from the distance error and tracking it with the speed controller.
- **Emergency mode**: when time-to-collision (TTC) falls below the configured threshold (3.0 s), apply maximum braking.

## PID tuning
Two controllers are tuned separately (speed and distance). The tuned gains are stored in `tuning_results.yaml` and are loaded by `simulation.py` at runtime, so changing gains changes the simulation behavior.

## Simulation result
Key outcomes from the generated `simulation_results.csv`:
- Cruise (0–30 s): rise time (3→27 m/s) ≈ 8.1 s, overshoot ≈ 0 m/s, steady-state error (25–30 s) ≈ 0 m/s.
- Follow (30–60 s): spacing error converges near 0 m and distance stays above 90% of the computed safe distance.
- Safety: acceleration commands remain within limits, minimum simulated gap stays above 5 m, and emergency mode occurs during the hard-braking interval with TTC < 3 s and negative acceleration.

