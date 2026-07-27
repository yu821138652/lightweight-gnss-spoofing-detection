# Mixed outer-CV v2 protocol

This directory fixes the recording-level input to the reviewed 24-Session static+dynamic baseline.

## Acquisition groups

Outer splitting uses `outer_group`, not the Session name alone. Most Sessions are independent groups. Two pairs must remain in the same test fold because their canonical UTC epochs overlap and they represent the same acquisition conditions:

| Group | Sessions | Shared canonical epochs |
|---|---|---:|
| `G08` | new-building `st_L5/20.16` and `st_L5/20.36` | 730 |
| `G09` | playground `dy_L_15/08.12-08.16` and `dy_L_15/08.26-08.32` | 354 |

No identical `DeviceName + utcTimeMillis + signal_id` rows were found across either pair. The grouping prevents the same field event and time conditions from appearing on both sides of an outer split; it is not a claim that rows are duplicated.

## Balance constraint

Every fold contains six complete Sessions and covers static/dynamic, both environments, and L1/L5/L1+L5. Every development pool retains all six motion-by-band Scenarios.

An exact offline set-partition audit over the 22 indivisible groups established:

- the old unconstrained weighted objective has a global optimum with 58.97% worst-fold positive-row deviation;
- the theoretical minimax positive-row deviation is 20.5833%;
- a 25% cap is the Pareto knee: the constrained optimum has objective 1.75774948 and 24.1788% actual worst-fold deviation;
- tightening the cap to 22.5% raises the objective to 2.26133, while relaxing it from 25% through 54% gives no improvement.

`38_generate_mixed_timeblock_protocol.py` therefore enforces a 25% maximum positive-row deviation by default, then minimises the existing balance objective within the feasible region. The production script does not depend on SciPy; SciPy/HiGHS was used only for the offline optimality audit.

## Inner validation

The outer assignment is followed by strict v2 inner validation:

- 64 canonical epochs per block;
- target validation fraction 20%;
- validation size optimised before label ratio, with a 2 percentage-point tolerance;
- four guard epochs on each side of a train/validation boundary for W5;
- every development Session has usable validation epochs;
- every fold has positive and negative validation support for all six Scenarios.

The generated manifests and tensors are local rebuildable outputs under `output/`; only this source manifest and rationale are tracked.
