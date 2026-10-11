"""
generate_dataset.py
-------------------
Synthetic Financial Transaction Dataset Generator for Academic Prototype.
Dataset Title: "Synthetic Financial Transaction Dataset for Academic Prototype"
NOTE: This dataset is synthetically generated for academic demonstration
and research evaluation. It is NOT real banking data.
"""

import os
import numpy as np
import pandas as pd

def generate_synthetic_transactions(
    n_samples: int = 5000,
    target_fraud_ratio: float = 0.07,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates realistic synthetic financial transactions with realistic correlations
    and non-deterministic fraud outcomes.

    Parameters:
    -----------
    n_samples : int
        Total number of transactions to generate (default 5,000).
    target_fraud_ratio : float
        Target proportion of fraudulent transactions (default 0.07 / ~7%).
    random_state : int
        Seed for reproducibility.

    Returns:
    --------
    pd.DataFrame containing 12 features including 'fraud_label'.
    """
    rng = np.random.default_rng(random_state)

    # 1. Base Feature Generation
    # Transaction amount: log-normal distribution (mostly small, occasional large)
    # Median around ₹1,500, with long tail reaching ₹100,000+
    amounts = rng.lognormal(mean=7.5, sigma=1.2, size=n_samples)
    amounts = np.clip(np.round(amounts, 2), 50.0, 150000.0)

    # Transaction frequency: Poisson distributed transactions per day (1 - 25)
    frequency = rng.poisson(lam=3.5, size=n_samples) + 1
    frequency = np.clip(frequency, 1, 30)

    # Location Risk: 0=Low, 1=Medium, 2=High
    location_risk = rng.choice([0, 1, 2], size=n_samples, p=[0.72, 0.20, 0.08])

    # Device Changed: 0=No, 1=Yes (approx 12% change rate)
    device_changed = rng.choice([0, 1], size=n_samples, p=[0.88, 0.12])

    # Previous Fraud on Account: 0=No, 1=Yes (approx 4% history)
    previous_fraud = rng.choice([0, 1], size=n_samples, p=[0.96, 0.04])

    # Distance from usual location in km (mostly local < 25km, occasional travel)
    distance = rng.exponential(scale=35.0, size=n_samples)
    distance = np.clip(np.round(distance, 1), 0.1, 2500.0)

    # Transaction Hour: 0 to 23
    # Legitimate transactions peak during 9 AM - 10 PM
    hour_weights = np.array([
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02, # 00-05
        0.03, 0.05, 0.06, 0.07, 0.08, 0.08, # 06-11
        0.08, 0.08, 0.07, 0.06, 0.06, 0.07, # 12-17
        0.06, 0.05, 0.03, 0.02, 0.01, 0.01  # 18-23
    ], dtype=float)
    hour_probs = hour_weights / hour_weights.sum()
    hours = rng.choice(range(24), size=n_samples, p=hour_probs)

    # Merchant Risk: 0=Low, 1=Medium, 2=High
    merchant_risk = rng.choice([0, 1, 2], size=n_samples, p=[0.75, 0.18, 0.07])

    # Failed Attempts prior to transaction (0 to 5)
    failed_attempts = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.85, 0.09, 0.04, 0.015, 0.005])

    # International Transaction: 0=No, 1=Yes (~5%)
    international = rng.choice([0, 1], size=n_samples, p=[0.95, 0.05])

    # Account Age in days (10 to 3650 days ~ 10 years)
    account_age_days = rng.integers(10, 3650, size=n_samples)

    # 2. Risk Scoring & Latent Fraud Probability Computation
    # Normalize components to calculate composite risk logit
    norm_amount = (amounts - 50.0) / 10000.0
    norm_freq = (frequency - 1.0) / 5.0
    norm_dist = distance / 150.0
    odd_hour = np.where((hours >= 1) & (hours <= 5), 1.8, 0.0)

    latent_risk = (
        0.28 * norm_amount +
        0.35 * norm_freq +
        0.90 * location_risk +
        1.10 * device_changed +
        1.40 * previous_fraud +
        0.40 * norm_dist +
        0.50 * odd_hour +
        0.80 * merchant_risk +
        0.75 * failed_attempts +
        0.90 * international -
        0.15 * (account_age_days / 365.0) -
        4.80  # Base intercept to achieve desired target prevalence
    )

    # Sigmoid function for probability
    fraud_prob = 1.0 / (1.0 + np.exp(-latent_risk))

    # Add stochastic noise for realistic non-deterministic overlap
    noise = rng.normal(0.0, 0.04, size=n_samples)
    adjusted_prob = np.clip(fraud_prob + noise, 0.001, 0.999)

    # Determine threshold to match target fraud ratio closely
    threshold = np.quantile(adjusted_prob, 1.0 - target_fraud_ratio)
    fraud_label = (adjusted_prob >= threshold).astype(int)

    # Assemble into DataFrame
    df = pd.DataFrame({
        "transaction_amount": amounts,
        "transaction_frequency": frequency,
        "location_risk": location_risk,
        "device_changed": device_changed,
        "previous_fraud": previous_fraud,
        "distance_from_usual_location": distance,
        "transaction_hour": hours,
        "merchant_risk": merchant_risk,
        "failed_attempts": failed_attempts,
        "international_transaction": international,
        "account_age_days": account_age_days,
        "fraud_label": fraud_label
    })

    return df

def save_dataset(output_path: str = "data/transactions.csv", n_samples: int = 5000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_synthetic_transactions(n_samples=n_samples)
    df.to_csv(output_path, index=False)
    fraud_count = df['fraud_label'].sum()
    fraud_pct = (fraud_count / len(df)) * 100
    print(f"Dataset successfully created at: {output_path}")
    print(f"Total Transactions: {len(df)}")
    print(f"Legitimate Transactions (0): {len(df) - fraud_count} ({100 - fraud_pct:.2f}%)")
    print(f"Fraudulent Transactions (1): {fraud_count} ({fraud_pct:.2f}%)")
    return output_path

if __name__ == "__main__":
    save_dataset()
