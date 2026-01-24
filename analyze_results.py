# analyze_results.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import OUTPUT_DIR, PLOT_OUTPUT
import os


def calculate_deference_rates(df):
    """
    Calculates deference rates for M1 and M2.
    
    Args:
        df: DataFrame containing experimental results
        
    Returns:
        Dictionary containing deference statistics
    """
    # M1 deference rate (high status deferring to low status)
    m1_deference_rate = df["m1_toward_m2"].mean()
    
    # M2 deference rate (low status deferring to high status)
    m2_deference_rate = df["m2_toward_m1"].mean()
    
    # Asymmetry metric
    asymmetry = m2_deference_rate - m1_deference_rate
    
    # Change magnitudes (only when change occurred)
    m1_avg_magnitude = df[df["m1_changed"]]["m1_change_magnitude"].mean()
    m2_avg_magnitude = df[df["m2_changed"]]["m2_change_magnitude"].mean()
    
    stats = {
        "m1_deference_rate": m1_deference_rate,
        "m2_deference_rate": m2_deference_rate,
        "asymmetry": asymmetry,
        "m1_avg_change_magnitude": m1_avg_magnitude,
        "m2_avg_change_magnitude": m2_avg_magnitude,
        "total_trials": len(df),
        "m1_changed_count": df["m1_changed"].sum(),
        "m2_changed_count": df["m2_changed"].sum()
    }
    
    return stats


def print_summary_statistics(stats):
    """
    Prints summary statistics to console.
    
    Args:
        stats: Dictionary containing deference statistics
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Trials: {stats['total_trials']}")
    print(f"\nM1 (High Status) Deference Rate: {stats['m1_deference_rate']:.2%}")
    print(f"M1 Changed Ratings: {stats['m1_changed_count']} trials")
    print(f"M1 Average Change Magnitude: {stats['m1_avg_change_magnitude']:.3f}")
    print(f"\nM2 (Low Status) Deference Rate: {stats['m2_deference_rate']:.2%}")
    print(f"M2 Changed Ratings: {stats['m2_changed_count']} trials")
    print(f"M2 Average Change Magnitude: {stats['m2_avg_change_magnitude']:.3f}")
    print(f"\nAsymmetry (M2 - M1): {stats['asymmetry']:.2%}")
    print("="*60 + "\n")


def create_visualizations(df):
    """
    Creates visualizations of deference patterns.
    
    Args:
        df: DataFrame containing experimental results
        
    Returns:
        Path to saved plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Deference Rates
    ax1 = axes[0, 0]
    deference_data = pd.DataFrame({
        "Model": ["M1 (High Status)", "M2 (Low Status)"],
        "Deference Rate": [df["m1_toward_m2"].mean(), df["m2_toward_m1"].mean()]
    })
    sns.barplot(data=deference_data, x="Model", y="Deference Rate", ax=ax1, palette=["#3498db", "#e74c3c"])
    ax1.set_ylabel("Proportion of Trials")
    ax1.set_title("Deference Rates by Model Status")
    ax1.set_ylim(0, 1)
    
    # Plot 2: Change Magnitudes
    ax2 = axes[0, 1]
    magnitude_data = []
    for _, row in df.iterrows():
        if row["m1_changed"]:
            magnitude_data.append({"Model": "M1 (High Status)", "Magnitude": row["m1_change_magnitude"]})
        if row["m2_changed"]:
            magnitude_data.append({"Model": "M2 (Low Status)", "Magnitude": row["m2_change_magnitude"]})
    
    if magnitude_data:
        magnitude_df = pd.DataFrame(magnitude_data)
        sns.boxplot(data=magnitude_df, x="Model", y="Magnitude", ax=ax2, palette=["#3498db", "#e74c3c"])
        ax2.set_ylabel("Rating Change Magnitude")
        ax2.set_title("Distribution of Change Magnitudes")
    
    # Plot 3: Initial Disagreement vs Deference
    ax3 = axes[1, 0]
    m2_deference_by_disagreement = df.groupby(pd.cut(df["initial_disagreement"], bins=5))["m2_toward_m1"].mean()
    m2_deference_by_disagreement.plot(kind="bar", ax=ax3, color="#e74c3c")
    ax3.set_xlabel("Initial Disagreement (binned)")
    ax3.set_ylabel("M2 Deference Rate")
    ax3.set_title("M2 Deference by Initial Disagreement")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    
    # Plot 4: Interaction Style Effects
    ax4 = axes[1, 1]
    style_effects = df.groupby("interaction_style").agg({
        "m2_toward_m1": "mean",
        "m1_toward_m2": "mean"
    })
    style_effects.plot(kind="bar", ax=ax4, color=["#e74c3c", "#3498db"])
    ax4.set_xlabel("Interaction Style")
    ax4.set_ylabel("Deference Rate")
    ax4.set_title("Deference Rates by Interaction Style")
    ax4.legend(["M2 to M1", "M1 to M2"])
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, PLOT_OUTPUT)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {output_path}")
    
    return output_path


def analyze_by_condition(df):
    """
    Analyzes results broken down by experimental conditions.
    
    Args:
        df: DataFrame containing experimental results
    """
    print("\nANALYSIS BY CONDITION")
    print("="*60)
    
    # By interaction style
    print("\nBy Interaction Style:")
    for style in df["interaction_style"].unique():
        style_df = df[df["interaction_style"] == style]
        m2_rate = style_df["m2_toward_m1"].mean()
        print(f"  {style}: M2 deference = {m2_rate:.2%}")
    
    # By gender pairing
    print("\nBy Gender Pairing:")
    df["gender_pair"] = df["m1_gender"] + "-" + df["m2_gender"]
    for pair in df["gender_pair"].unique():
        pair_df = df[df["gender_pair"] == pair]
        m2_rate = pair_df["m2_toward_m1"].mean()
        print(f"  M1={pair.split('-')[0]}, M2={pair.split('-')[1]}: M2 deference = {m2_rate:.2%}")
    
    # By status profile type
    print("\nBy Status Profile:")
    for m1_prof in df["m1_profile"].unique():
        for m2_prof in df["m2_profile"].unique():
            profile_df = df[(df["m1_profile"] == m1_prof) & (df["m2_profile"] == m2_prof)]
            if len(profile_df) > 0:
                m2_rate = profile_df["m2_toward_m1"].mean()
                print(f"  M1={m1_prof}, M2={m2_prof}: M2 deference = {m2_rate:.2%}")
    
    print("="*60 + "\n")