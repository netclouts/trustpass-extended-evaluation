#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Output
# ============================================================

OUT = Path("results_extended_v2")
OUT.mkdir(parents=True, exist_ok=True)

METHODS = [
    "Majority",
    "EBT",
    "ProvOnly",
    "DS-Trust",
    "TrustPass",
    "TP-NoDRM",
    "TP-NoHealth",
    "TP-NoUncertainty",
    "TP-NoCRV",
    "TP-NoProvenance",
]

ABLATIONS = [
    "TrustPass",
    "TP-NoDRM",
    "TP-NoHealth",
    "TP-NoUncertainty",
    "TP-NoCRV",
    "TP-NoProvenance",
]

# ============================================================
# Agent model
# ============================================================

class Agent:
    def __init__(self, agent_id, role="generic", base_trust=0.70):
        self.agent_id = agent_id
        self.role = role
        self.tau = base_trust

        self.sigma_base = {
            "orbiter": 0.15,
            "lander": 0.10,
            "rover": 0.12,
            "generic": 0.13,
        }.get(role, 0.13)

    def observe(self, truth, sigma, fault, rng):
        return truth + rng.normal(0.0, sigma * self.sigma_base) + fault


# ============================================================
# Utility functions
# ============================================================

def safe_weighted_prediction(claims, confidence, uncertainty, weights, received_mask):
    """
    Weighted binary prediction using ONLY actually received evidence.
    Missing packets are not interpreted as negative claims.
    """
    mask = received_mask

    if not np.any(mask):
        return 0

    c = claims[mask].astype(float)
    v = confidence[mask].astype(float)
    u = uncertainty[mask].astype(float)
    w = weights[mask].astype(float)

    effective = w * v * (1.0 - u)
    denominator = np.sum(effective)

    if denominator <= 1e-12:
        return 0

    score = np.sum(effective * c) / denominator
    return int(score >= 0.5)


def dempster_combine(m1, m2):
    """
    Dempster-Shafer combination for [positive, negative, unknown].
    """
    conflict = m1[0] * m2[1] + m1[1] * m2[0]

    if conflict >= 1.0 - 1e-12:
        return 0.5 * (m1 + m2)

    positive = m1[0] * m2[0] + m1[0] * m2[2] + m1[2] * m2[0]
    negative = m1[1] * m2[1] + m1[1] * m2[2] + m1[2] * m2[1]
    unknown = m1[2] * m2[2]

    result = np.array([positive, negative, unknown], dtype=float) / (1.0 - conflict)
    total = result.sum()

    if total <= 1e-12:
        return np.array([0.33, 0.33, 0.34])

    return result / total


def calculate_crv_adjustment(
    claims,
    confidence,
    uncertainty,
    health,
    provenance_quality,
    integrity_valid,
    received_mask,
):
    """
    Cross-Relay Verification (CRV). Uses only information available to the
    verifier: agreement with the received population, evidence quality,
    health, provenance/integrity. Does NOT use the hidden ground truth.

    Returns a per-agent CRV factor in [0, 1].
    """
    n = len(claims)
    factors = np.ones(n, dtype=float)

    received = received_mask

    if np.sum(received) <= 1:
        return factors

    quality = confidence * (1.0 - uncertainty) * health * provenance_quality * integrity_valid.astype(float)
    quality = np.where(received, quality, 0.0)

    if np.sum(quality) <= 1e-12:
        return factors

    population_score = np.sum(quality * claims) / np.sum(quality)
    population_prediction = int(population_score >= 0.5)

    for i in range(n):
        if not received[i]:
            factors[i] = 0.0
            continue

        agreement = 1.0 if int(claims[i]) == population_prediction else 0.35

        evidence_quality = (
            0.40 * confidence[i]
            + 0.25 * (1.0 - uncertainty[i])
            + 0.20 * health[i]
            + 0.15 * provenance_quality[i]
        )
        evidence_quality = np.clip(evidence_quality, 0.0, 1.0)

        factors[i] = np.clip(0.50 * agreement + 0.50 * evidence_quality, 0.05, 1.0)

    return factors


def make_prediction(
    claims,
    confidence,
    uncertainty,
    health,
    provenance_quality,
    integrity_valid,
    taus,
    received_mask,
    method,
):
    """
    Compute a prediction for one phenomenon. Each ablation removes exactly
    one component from the TrustPass weighting model. No hidden ground
    truth is used here.
    """
    if method == "Majority":
        if not np.any(received_mask):
            return 0
        c = claims[received_mask]
        return int(np.mean(c) >= 0.5)

    if method == "EBT":
        return safe_weighted_prediction(claims, confidence, uncertainty, taus, received_mask)

    if method == "ProvOnly":
        return safe_weighted_prediction(
            claims, confidence, uncertainty, np.ones_like(taus), received_mask
        )

    if method == "DS-Trust":
        masses = np.array([0.0, 0.0, 1.0], dtype=float)

        for i in range(len(claims)):
            if not received_mask[i]:
                continue

            belief = np.clip(
                taus[i] * confidence[i] * (1.0 - uncertainty[i]) * provenance_quality[i] * integrity_valid[i],
                0.0,
                0.95,
            )

            if claims[i] == 1:
                mass = np.array([belief, 0.0, 1.0 - belief])
            else:
                mass = np.array([0.0, belief, 1.0 - belief])

            masses = dempster_combine(masses, mass)

        return int(masses[0] > masses[1])

    # --------------------------------------------------------
    # TrustPass component switches
    # --------------------------------------------------------

    use_drm = method != "TP-NoDRM"
    use_health = method != "TP-NoHealth"
    use_uncertainty = method != "TP-NoUncertainty"
    use_crv = method != "TP-NoCRV"
    use_provenance = method != "TP-NoProvenance"

    reputation = taus.copy() if use_drm else np.ones_like(taus)
    health_component = health.copy() if use_health else np.ones_like(health)

    provenance_component = (
        provenance_quality * integrity_valid if use_provenance else np.ones_like(provenance_quality)
    )

    if use_crv:
        crv_component = calculate_crv_adjustment(
            claims=claims,
            confidence=confidence,
            uncertainty=uncertainty,
            health=health_component,
            provenance_quality=provenance_quality if use_provenance else np.ones_like(provenance_quality),
            integrity_valid=integrity_valid if use_provenance else np.ones_like(integrity_valid),
            received_mask=received_mask,
        )
    else:
        crv_component = np.ones_like(taus)

    weights = reputation * health_component * provenance_component * crv_component

    if use_uncertainty:
        effective_uncertainty = np.clip(uncertainty, 0.0, 1.0)
    else:
        effective_uncertainty = np.zeros_like(uncertainty)

    return safe_weighted_prediction(claims, confidence, effective_uncertainty, weights, received_mask)


# ============================================================
# Main simulation
# ============================================================

def simulate_trial(
    N,
    K=3,
    T=30,
    adversary_fraction=0.55,
    relay_compromise_fraction=0.12,
    packet_loss=0.0,
    alpha=0.12,
    seed=42,
):
    """
    One complete Monte Carlo trial.

    1. Missing packets remain missing.
    2. Missing evidence is NOT converted to a negative claim.
    3. Ground truth is used only for evaluation and reputation feedback.
    4. CRV does not use ground truth.
    5. Provenance/integrity are explicit evidence attributes.
    6. Each ablation removes one computational component.
    """
    rng = np.random.default_rng(seed)

    roles = ["orbiter", "lander", "rover", "generic"]

    n_adv = max(1, round(N * adversary_fraction))
    adversaries = set(range(n_adv))

    agents = [Agent(i, roles[i % len(roles)], base_trust=0.70) for i in range(N)]

    truths = rng.binomial(1, 0.5, size=K)
    taus = np.array([agent.tau for agent in agents], dtype=float)

    scores = {method: [] for method in METHODS}

    received_counts = []
    integrity_failures = []

    for _ in range(T):
        claims = np.zeros((N, K), dtype=int)
        confidence = np.zeros((N, K), dtype=float)
        uncertainty = np.zeros((N, K), dtype=float)
        health = np.zeros((N, K), dtype=float)
        provenance_quality = np.ones((N, K), dtype=float)
        integrity_valid = np.ones((N, K), dtype=bool)

        # ----------------------------------------------------
        # Agent observations
        # ----------------------------------------------------
        for i, agent in enumerate(agents):
            for k in range(K):
                truth = float(truths[k])
                fault = 0.0

                if i in adversaries:
                    fault = rng.normal(0.0, 0.95)

                obs = agent.observe(truth, 0.22, fault, rng)
                claim = int(obs > 0.5)

                dist = abs(obs - 0.5)
                conf = np.clip(dist / (dist + 0.45), 0.05, 0.97)

                eff_sigma = 0.22 * agent.sigma_base + abs(fault) * 0.55
                unc = np.clip(eff_sigma / (eff_sigma + 0.28), 0.03, 0.94)

                h = np.clip(1.0 - abs(fault) / 1.4, 0.05, 1.0)

                if i in adversaries and rng.random() < 0.60:
                    claim = 1 - int(truth)
                    conf = rng.uniform(0.78, 0.96)
                    unc = rng.uniform(0.04, 0.14)
                    h = rng.uniform(0.25, 0.55)

                if i in adversaries:
                    provenance_quality[i, k] = rng.uniform(0.35, 0.90)
                else:
                    provenance_quality[i, k] = rng.uniform(0.90, 1.00)

                claims[i, k] = claim
                confidence[i, k] = conf
                uncertainty[i, k] = unc
                health[i, k] = h

        # ----------------------------------------------------
        # Relay compromise
        # ----------------------------------------------------
        relay_mask = rng.random(claims.shape) < relay_compromise_fraction
        received = claims.copy()
        received[relay_mask] = 1 - received[relay_mask]

        integrity_valid[relay_mask] = False
        provenance_quality[relay_mask] *= rng.uniform(0.20, 0.60, size=np.sum(relay_mask))

        # ----------------------------------------------------
        # Packet loss (missing evidence, NOT a negative claim)
        # ----------------------------------------------------
        received_mask = rng.random(received.shape) >= packet_loss

        received_counts.append(float(np.mean(received_mask)))
        integrity_failures.append(float(np.mean(~integrity_valid & received_mask)))

        # ----------------------------------------------------
        # Evaluate each phenomenon
        # ----------------------------------------------------
        for k in range(K):
            c = received[:, k]
            v = confidence[:, k]
            u = uncertainty[:, k]
            h = health[:, k]
            p = provenance_quality[:, k]
            integrity = integrity_valid[:, k]
            mask = received_mask[:, k]

            for method in METHODS:
                prediction = make_prediction(
                    claims=c,
                    confidence=v,
                    uncertainty=u,
                    health=h,
                    provenance_quality=p,
                    integrity_valid=integrity,
                    taus=taus,
                    received_mask=mask,
                    method=method,
                )
                scores[method].append(float(prediction == truths[k]))

        # ----------------------------------------------------
        # Reputation update (only for evidence actually received)
        # ----------------------------------------------------
        for i in range(N):
            feedback_values = []

            for k in range(K):
                if received_mask[i, k]:
                    feedback_values.append(float(claims[i, k] == truths[k]))

            if feedback_values:
                feedback = float(np.mean(feedback_values))
                taus[i] = (1.0 - alpha) * taus[i] + alpha * feedback

        taus = np.clip(taus, 0.05, 0.98)

    result = {method: float(np.mean(values)) for method, values in scores.items()}
    result["mean_evidence_received"] = float(np.mean(received_counts))
    result["mean_integrity_failure_rate"] = float(np.mean(integrity_failures))

    return result


# ============================================================
# Statistical summary
# ============================================================

def summarize(rows, group_columns):
    """
    Summarize independent Monte Carlo trials. Derives the number of
    observations from the actual raw trial rows.
    """
    df = pd.DataFrame(rows)
    grouped = df.groupby(group_columns, dropna=False)

    result = grouped.mean(numeric_only=True).reset_index()

    trial_counts = grouped.size().reset_index(name="trials")
    result = result.merge(trial_counts, on=group_columns, how="left")

    for method in METHODS:
        if method not in df.columns:
            continue

        std = grouped[method].std(ddof=1).reset_index(name=f"{method}_std")
        result = result.merge(std, on=group_columns, how="left")

        result[f"{method}_ci95"] = 1.96 * result[f"{method}_std"] / np.sqrt(result["trials"])

    return result


# ============================================================
# Ablation experiment
# ============================================================

def run_ablation():
    rows = []

    for N in [6, 12, 25]:
        adv = 0.60 if N == 6 else 0.55

        for trial in range(80):
            result = simulate_trial(
                N=N,
                adversary_fraction=adv,
                relay_compromise_fraction=0.12,
                packet_loss=0.0,
                seed=42 + trial,
            )

            row = {"N": N, "trial": trial}
            for method in METHODS:
                row[method] = result[method]

            row["mean_evidence_received"] = result["mean_evidence_received"]
            row["mean_integrity_failure_rate"] = result["mean_integrity_failure_rate"]

            rows.append(row)

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "ablation_raw.csv", index=False)

    summary = summarize(raw, ["N"])
    summary.to_csv(OUT / "ablation_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)

    for N in [6, 12, 25]:
        row = summary[summary["N"] == N].iloc[0]
        print(f"\nN={N} | trials={int(row['trials'])}")

        for method in ABLATIONS:
            print(f"{method:22s} {row[method]:.4f} +/- {row[f'{method}_ci95']:.4f}")

    plt.figure(figsize=(9, 5.5))
    x = np.arange(len(ABLATIONS))
    width = 0.25

    for j, N in enumerate([6, 12, 25]):
        values = summary[summary["N"] == N].iloc[0]
        y = [values[m] for m in ABLATIONS]
        err = [values[f"{m}_ci95"] for m in ABLATIONS]
        plt.bar(x + j * width, y, width, yerr=err, capsize=3, label=f"N={N}")

    plt.xticks(x + width, ABLATIONS, rotation=35, ha="right")
    plt.ylabel("Mean accuracy")
    plt.ylim(0.0, 1.02)
    plt.title("TrustPass ablation study with 95% confidence intervals")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "ablation_accuracy.pdf")
    plt.close()


# ============================================================
# Attack-fraction sweep
# ============================================================

def run_attack_sweep():
    rows = []
    fractions = np.arange(0.10, 0.71, 0.10)

    for fraction in fractions:
        for trial in range(80):
            result = simulate_trial(
                N=12,
                adversary_fraction=float(fraction),
                relay_compromise_fraction=0.12,
                packet_loss=0.0,
                seed=5000 + trial,
            )

            rows.append({
                "adversary_fraction": round(float(fraction), 2),
                "trial": trial,
                "TrustPass": result["TrustPass"],
                "Majority": result["Majority"],
                "EBT": result["EBT"],
                "DS-Trust": result["DS-Trust"],
            })

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "attack_sweep_raw.csv", index=False)

    summary = (
        raw.groupby("adversary_fraction")
        .agg(
            TrustPass_mean=("TrustPass", "mean"),
            TrustPass_std=("TrustPass", "std"),
            Majority_mean=("Majority", "mean"),
            EBT_mean=("EBT", "mean"),
            DSTrust_mean=("DS-Trust", "mean"),
            trials=("trial", "count"),
        )
        .reset_index()
    )

    summary["TrustPass_ci95"] = 1.96 * summary["TrustPass_std"] / np.sqrt(summary["trials"])
    summary.to_csv(OUT / "attack_sweep_summary.csv", index=False)

    plt.figure(figsize=(6.5, 4.2))
    x = summary["adversary_fraction"]

    plt.errorbar(x, summary["TrustPass_mean"], yerr=summary["TrustPass_ci95"], marker="o", capsize=3, label="TrustPass")
    plt.plot(x, summary["Majority_mean"], "s--", label="Majority")
    plt.plot(x, summary["EBT_mean"], "^--", label="EBT")
    plt.plot(x, summary["DSTrust_mean"], "d--", label="DS-Trust")

    plt.xlabel("Compromised-agent fraction")
    plt.ylabel("Mean accuracy")
    plt.ylim(0.0, 1.02)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "attack_sweep.pdf")
    plt.close()


# ============================================================
# Packet-loss sweep
# ============================================================

def run_packet_loss_sweep():
    rows = []
    losses = np.arange(0.0, 0.51, 0.10)

    for loss in losses:
        for trial in range(80):
            result = simulate_trial(
                N=12,
                adversary_fraction=0.55,
                relay_compromise_fraction=0.12,
                packet_loss=float(loss),
                seed=9000 + trial,
            )

            rows.append({
                "packet_loss": round(float(loss), 2),
                "trial": trial,
                "TrustPass": result["TrustPass"],
                "Majority": result["Majority"],
                "EBT": result["EBT"],
                "mean_evidence_received": result["mean_evidence_received"],
            })

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "packet_loss_raw.csv", index=False)

    summary = (
        raw.groupby("packet_loss")
        .agg(
            TrustPass_mean=("TrustPass", "mean"),
            TrustPass_std=("TrustPass", "std"),
            Majority_mean=("Majority", "mean"),
            EBT_mean=("EBT", "mean"),
            evidence_received_mean=("mean_evidence_received", "mean"),
            trials=("trial", "count"),
        )
        .reset_index()
    )

    summary["TrustPass_ci95"] = 1.96 * summary["TrustPass_std"] / np.sqrt(summary["trials"])
    summary.to_csv(OUT / "packet_loss_summary.csv", index=False)

    plt.figure(figsize=(6.5, 4.2))
    x = summary["packet_loss"]

    plt.errorbar(x, summary["TrustPass_mean"], yerr=summary["TrustPass_ci95"], marker="o", capsize=3, label="TrustPass")
    plt.plot(x, summary["Majority_mean"], "s--", label="Majority")
    plt.plot(x, summary["EBT_mean"], "^--", label="EBT")

    plt.xlabel("Packet-loss fraction")
    plt.ylabel("Mean accuracy")
    plt.ylim(0.0, 1.02)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "packet_loss_sweep.pdf")
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Running corrected TrustPass extended experiments...")
    print(f"\nOutput directory: {OUT.resolve()}")

    run_ablation()
    run_attack_sweep()
    run_packet_loss_sweep()

    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"Outputs saved in: {OUT.resolve()}")
    print("\nGenerated files:")

    for path in sorted(OUT.iterdir()):
        print(f"  {path.name}")
