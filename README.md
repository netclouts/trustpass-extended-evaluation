# TRUSTPASS Extended Evaluation

This repository contains the reproducible code, data, figures, bibliography,
and IEEE-style manuscript source for:

**Cybersecurity Assurance of AI-Generated Scientific Claims in Autonomous
Deep-Space Multi-Agent Systems**

The experiments evaluate TRUSTPASS under adversarial multi-agent conditions,
component ablations, compromised-agent sweeps, and packet loss.

## Repository contents

- `main.tex` — IEEE-style manuscript source.
- `references.bib` — BibTeX bibliography used by the manuscript.
- `run_extended_experiments2.py` — reproducible simulation script.
- `ablation_raw.csv` — raw ablation trial results.
- `ablation_summary.csv` — aggregated ablation results.
- `ablation_accuracy.pdf` — ablation figure.
- `attack_sweep_raw.csv` — raw compromised-agent sweep results.
- `attack_sweep_summary.csv` — aggregated compromised-agent sweep results.
- `attack_sweep.pdf` — compromised-agent sweep figure.
- `packet_loss_raw.csv` — raw packet-loss trial results.
- `packet_loss_summary.csv` — aggregated packet-loss results.
- `packet_loss_sweep.pdf` — packet-loss sweep figure.

All experiment outputs are currently stored in the repository root directory.

## Experiments

### Component ablation

The ablation study compares:

- TRUSTPASS.
- TRUSTPASS without dynamic reputation.
- TRUSTPASS without sensor-health weighting.
- TRUSTPASS without uncertainty weighting.
- TRUSTPASS without cross-relay verification.
- TRUSTPASS without provenance and integrity weighting.

The ablation experiment evaluates constellation sizes of 6, 12, and 25 agents.
Each configuration uses 80 Monte Carlo trials.

### Attack-intensity sweep

The attack sweep varies the compromised-agent fraction from 10% to 70% at
a constellation size of 12 agents.

Each compromised-agent configuration uses 80 Monte Carlo trials.

### Packet-loss sweep

The packet-loss sweep varies packet loss from 0% to 50% at a constellation
size of 12 agents, with 55% compromised agents and 12% relay compromise.

Each packet-loss configuration uses 80 Monte Carlo trials.

## Main findings

Under the tested synthetic configuration:

- Sensor-health weighting provides the largest measured ablation contribution.
- Provenance and integrity weighting provide the second-largest measured contribution.
- Dynamic reputation provides its greatest benefit in the smallest constellation.
- Uncertainty weighting has negligible effect under the current calibration.
- The current population-based cross-relay verification heuristic does not improve accuracy in every regime.
- Packet loss causes gradual performance degradation while TRUSTPASS remains above the selected baselines in the tested model.

The results should be interpreted as simulator-specific findings rather than
general security guarantees.

## Reproduction

### Requirements

- Python 3.13.14.
- NumPy.
- Pandas.
- Matplotlib.

Install the dependencies with:

```bash
python3 -m pip install numpy pandas matplotlib
```

Run the experiments with:

```bash
python3 run_extended_experiments2.py
```

The script regenerates the experiment CSV files and PDF figures in the
repository root directory.



## Important interpretation

The simulator uses:

- Synthetic observations.
- Synthetic sensor-health values.
- Synthetic provenance-quality values.
- Synthetic integrity outcomes.
- Simplified adversarial behavior.
- Simplified packet-loss behavior.
- Offline ground truth for scoring and reputation feedback.

The results are a controlled proof of concept. They do not represent flight
validation, operational cryptographic deployment, or mission-ready cybersecurity
performance.

The evaluation does not implement a complete PROV-DM or PROV-AGENT system,
secure hardware roots of trust, cryptographic key-management protocols, or a
real deep-space delay-tolerant networking schedule.

## Limitations

The current evaluation does not fully model:

- Real spacecraft telemetry.
- Planetary science workflows.
- Correlated sensor failures.
- Perfectly coordinated collusion.
- Compromised signing keys.
- Real cryptographic attestation.
- Realistic deep-space contact schedules.
- Communication delays and queueing.
- Bandwidth scheduling.
- Bundle expiration.
- Delayed operational validation.
- Common-mode failures across supposedly independent agents.

These limitations should be considered when interpreting the reported accuracy
values.

## Data and code availability

Paper source, experiment code, generated figures, and summary data:

[TRUSTPASS Extended Evaluation](https://github.com/netclouts/trustpass-extended-evaluation)

Original experiment repository:

[Deep-Space TRUSTPASS](https://github.com/netclouts/deep-space-trustpass)

## Acknowledgment

The author is the founder of Netclouts, a self-funded community initiative
that provides cybersecurity training and technical mentorship. Netclouts
provided an independent environment in which this research was developed.

The author also used AI for LaTeX editing and troubleshooting. The author
reviewed and verified the final manuscript and takes full responsibility for
its content, mathematical formulations, experimental design, results,
interpretations, and references.

## License

The simulation code and generated data are provided for research and
reproducibility purposes.

No specific open-source license has been declared for this repository. Users
should contact the author before redistributing modified versions or using the
materials for commercial purposes.

## Contact

**Abba Abdullahi Wakili**

[abbaabdullahiwakeel@gmail.com][(mailto:abbaabdullahiwakeel@gmail.com)](https://www.linkedin.com/in/abba-abdullahi-wakili-88b006346)
