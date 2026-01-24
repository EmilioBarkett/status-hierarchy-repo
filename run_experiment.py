# run_experiment.py

import asyncio
from data_handler import load_reviews, save_results
from experiment import run_experiment
from analyze_results import (
    calculate_deference_rates,
    print_summary_statistics,
    create_visualizations,
    analyze_by_condition
)
from config import STATUS_PROFILES, INTERACTION_STYLES, NUM_REVIEWS


async def main():
    """
    Main function to run the complete experiment.
    """
    print("="*60)
    print("STATUS HIERARCHY EXPERIMENT")
    print("="*60)
    
    # Load reviews
    reviews = load_reviews(num_reviews=NUM_REVIEWS)
    
    # Import condition configuration
    from config import (
        USE_SINGLE_STATUS_PAIR, 
        USE_SINGLE_INTERACTION_STYLE,
        DEFAULT_STATUS_PAIR,
        DEFAULT_INTERACTION_STYLE
    )
    
    # Define status pairs to test
    if USE_SINGLE_STATUS_PAIR:
        status_pairs = [
            (STATUS_PROFILES[DEFAULT_STATUS_PAIR[0]], STATUS_PROFILES[DEFAULT_STATUS_PAIR[1]])
        ]
    else:
        # Format: (high_status_profile, low_status_profile)
        status_pairs = [
            (STATUS_PROFILES["high_education"], STATUS_PROFILES["low_education"]),
            (STATUS_PROFILES["high_occupation"], STATUS_PROFILES["low_occupation"]),
            (STATUS_PROFILES["high_prestige_male"], STATUS_PROFILES["low_prestige_male"]),
            (STATUS_PROFILES["high_prestige_female"], STATUS_PROFILES["low_prestige_female"])
        ]
    
    # Define interaction styles to test
    if USE_SINGLE_INTERACTION_STYLE:
        interaction_styles = {DEFAULT_INTERACTION_STYLE: INTERACTION_STYLES[DEFAULT_INTERACTION_STYLE]}
    else:
        interaction_styles = INTERACTION_STYLES
    
    print(f"\nRunning experiment with:")
    print(f"  - {len(reviews)} reviews")
    print(f"  - {len(status_pairs)} status pair(s)")
    print(f"  - {len(interaction_styles)} interaction style(s)")
    print(f"  - Total trials: {len(reviews) * len(status_pairs) * len(interaction_styles)}\n")
    
    # Run experiment
    results = await run_experiment(reviews, status_pairs, interaction_styles)
    
    # Save results
    print(f"\nExperiment complete. Collected {len(results)} trials.")
    save_results(results)
    
    # Analyze results
    import pandas as pd
    df = pd.DataFrame(results)
    
    stats = calculate_deference_rates(df)
    print_summary_statistics(stats)
    
    analyze_by_condition(df)
    
    create_visualizations(df)
    
    print("Analysis complete!")


if __name__ == "__main__":
    asyncio.run(main())