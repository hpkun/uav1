# FireReady-Plain MAPPO development experiment

FireReady-Plain is a single-intervention development comparison against the
exact-3M Plain MAPPO baseline. The legacy 52-dimensional paper observation is
preserved. The opt-in `fire_ready_v1` schema appends one scalar to each Red
agent's local observation:

`o'_i = [o_i, 1{own FireState.armed}]`

Indices 0–51 remain byte-for-byte equivalent to the legacy observation and
index 52 reports only the observing Red UAV's current `FireState.armed`. A dead
Red UAV still receives an all-zero vector. The feature reads the state after
the environment transition, including entry-trigger consumption, out-of-window
re-arming, and intermediate-wave FireState reset.

This experiment removes avoidable own-fire-state aliasing. It does **not**
claim to restore a fully Markov local observation: wave index, remaining
horizon, teammate weapon state, and Blue weapon state may remain hidden. It
does not change weapon, reward, Blue policy, spawn, arena, or persistent-wave
dynamics.
