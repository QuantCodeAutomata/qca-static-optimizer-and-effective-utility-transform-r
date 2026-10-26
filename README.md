# Duality of Shortfall Aversion — Numerical Replication

Numerical replication of the paper 'Duality of Shortfall Aversion': static and dynamic duality
results for optimal consumption with habit formation and shortfall aversion.

## Structure

- `src/primitives.py` — Model utility families (Exponential, Power, AppendixC)
- `src/static_block.py` — Proposition 2.2: c*(y,h), ū(y,h), ū_y, ū_h
- `src/discounted_dual.py` — Discounted dual J_ρ(y,h), free boundary z_ρ(a) (ρ>0)
- `src/undiscounted_dual.py` — Undiscounted dual V_0(y,h), free boundary z_0(h) (ρ=0)
- `src/inverse_maps.py` — Monotone function inversion
- `src/simulation.py` — Exact Y_t simulation, Brownian bridge, habit path, MC
- `src/diagnostics.py` — Validation diagnostics

## Experiments

1. `exp/exp1_static.py` — Static optimizer c*(y,h) and transform ū(y,h)
2. `exp/exp2_discounted.py` — Discounted dual replication (ρ>0)
3. `exp/exp3_undiscounted.py` — Undiscounted dual replication (ρ=0)
4. `exp/exp4_special_cases.py` — Parameter sweeps for exponential and power families
5. `exp/exp5_counterexample.py` — Appendix C exact replication
6. `exp/exp6_validation.py` — Cross-cutting robustness diagnostics

## Usage

```bash
pip install -r requirements.txt
python exp/exp1_static.py
python exp/exp5_counterexample.py
pytest tests/ -v
```

## Key Results

See `results/RESULTS.md` for a summary of all experiment outputs.
