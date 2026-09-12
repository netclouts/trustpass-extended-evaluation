# TRUSTPASS Extended Evaluation

This repository contains the corrected extended experiments for:

> Cybersecurity Assurance of AI-Generated Scientific Claims in Autonomous Deep-Space Multi-Agent Systems

The experiments evaluate TRUSTPASS under adversarial multi-agent conditions, component ablations, compromised-agent sweeps, and packet loss.

## Contents

- `main.tex` — revised IEEE-style manuscript.
- `references.bib` — bibliography for the revised manuscript.
- `run_extended_experiments2.py` — reproducible experiment script.
- `results_extended_v2/` — raw trial records, summary CSV files, and figures.

## Experiments

The revised evaluation includes:

1. Component ablation:
   - TrustPass.
   - TrustPass without dynamic reputation.
   - TrustPass without sensor-health weighting.
   - TrustPass without uncertainty weighting.
   - TrustPass without cross-relay verification.
   - TrustPass without provenance/integrity weighting.

2. Attack-intensity sweep:
   - Compromised-agent fractions from 10% to 70%.
   - 80 trials per configuration.

3. Packet-loss sweep:
   - Packet-loss rates from 0% to 50%.
   - 80 trials per configuration.

## Output files

contains:

- `ablation_raw.csv`
- `ablation_summary.csv`
- `ablation_accuracy.pdf`
- `attack_sweep_raw.csv`
- `attack_sweep_summary.csv`
- `attack_sweep.pdf`
- `packet_loss_raw.csv`
- `packet_loss_summary.csv`
- `packet_loss_sweep.pdf`

## Reproduction

Requirements:

- Python 3.10 or later.
- NumPy.
- Pandas.
- Matplotlib.

Run:

```bash
python3 run_extended_experiments2.py
```

The script creates a new `results_extended_v2/` directory and regenerates the experiment outputs.

## Important interpretation

The simulator uses synthetic observations, synthetic provenance-quality values, synthetic integrity outcomes, and simplified adversarial behavior. The results are a controlled proof of concept and do not represent flight validation or mission-ready cybersecurity performance.

The experiments show that sensor-health and provenance/integrity weighting provide the largest contributions under the tested configuration. Dynamic reputation is most useful for smaller constellations. The current cross-relay verification heuristic does not improve accuracy in every regime and can reinforce a misleading population consensus.

## Paper and code

Paper source and extended evaluation:

```text
[https://github.com/netclouts/trustpass-extended-evaluation](https://github.com/netclouts/trustpass-extended-evaluation)
```

Original experiment repository:

```text
[https://github.com/netclouts/deep-space-trustpass](https://github.com/netclouts/deep-space-trustpass)
```

## License

The simulation code and generated data are provided for research and reproducibility purposes. 

## Contact

Abba Abdullahi Wakili  
abbaabdullahiwakeel@gmail.com
