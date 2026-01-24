# statistical_analysis.py

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
from statsmodels.stats.proportion import proportion_confint
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


def load_and_prepare_data(csv_files):
    """
    Load multiple CSV files and combine them into a single dataframe.
    
    Args:
        csv_files: List of CSV file paths or dict mapping condition names to file paths
        
    Returns:
        Combined dataframe with condition labels
    """
    if isinstance(csv_files, dict):
        dfs = []
        for condition_name, filepath in csv_files.items():
            df = pd.read_csv(filepath)
            df['condition'] = condition_name
            dfs.append(df)
        combined_df = pd.concat(dfs, ignore_index=True)
    else:
        dfs = [pd.read_csv(f) for f in csv_files]
        combined_df = pd.concat(dfs, ignore_index=True)
    
    return combined_df


def calculate_deference_rates_with_ci(df, groupby_cols=['condition']):
    """
    Calculate deference rates with 95% confidence intervals using bootstrap.
    
    Args:
        df: DataFrame with experimental data
        groupby_cols: Columns to group by (e.g., ['condition'] or ['m1_profile', 'm2_profile'])
        
    Returns:
        DataFrame with deference rates and confidence intervals
    """
    results = []
    
    for group_vals, group_df in df.groupby(groupby_cols):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        
        # M1 deference rate
        m1_deference_rate = group_df['m1_toward_m2'].mean()
        m1_n = len(group_df)
        m1_ci_low, m1_ci_high = proportion_confint(
            count=group_df['m1_toward_m2'].sum(),
            nobs=m1_n,
            alpha=0.05,
            method='wilson'
        )
        
        # M2 deference rate
        m2_deference_rate = group_df['m2_toward_m1'].mean()
        m2_n = len(group_df)
        m2_ci_low, m2_ci_high = proportion_confint(
            count=group_df['m2_toward_m1'].sum(),
            nobs=m2_n,
            alpha=0.05,
            method='wilson'
        )
        
        # Asymmetry
        asymmetry = m2_deference_rate - m1_deference_rate
        
        # Bootstrap CI for asymmetry
        asymmetry_ci = bootstrap_asymmetry_ci(
            group_df['m2_toward_m1'].values,
            group_df['m1_toward_m2'].values
        )
        
        result = {
            **dict(zip(groupby_cols, group_vals)),
            'n_trials': len(group_df),
            'm1_deference_rate': m1_deference_rate,
            'm1_ci_low': m1_ci_low,
            'm1_ci_high': m1_ci_high,
            'm2_deference_rate': m2_deference_rate,
            'm2_ci_low': m2_ci_low,
            'm2_ci_high': m2_ci_high,
            'asymmetry': asymmetry,
            'asymmetry_ci_low': asymmetry_ci[0],
            'asymmetry_ci_high': asymmetry_ci[1]
        }
        results.append(result)
    
    return pd.DataFrame(results)


def bootstrap_asymmetry_ci(m2_defers, m1_defers, n_bootstrap=10000, alpha=0.05):
    """
    Calculate bootstrap confidence interval for asymmetry metric.
    
    Args:
        m2_defers: Array of M2 deference (binary)
        m1_defers: Array of M1 deference (binary)
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level (0.05 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    asymmetries = []
    n = len(m2_defers)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        m2_sample = m2_defers[indices]
        m1_sample = m1_defers[indices]
        
        asymmetry = m2_sample.mean() - m1_sample.mean()
        asymmetries.append(asymmetry)
    
    asymmetries = np.array(asymmetries)
    ci_low = np.percentile(asymmetries, 100 * alpha / 2)
    ci_high = np.percentile(asymmetries, 100 * (1 - alpha / 2))
    
    return (ci_low, ci_high)


def cohens_h(p1, p2):
    """
    Calculate Cohen's h effect size for difference in proportions.
    
    Args:
        p1: Proportion 1
        p2: Proportion 2
        
    Returns:
        Cohen's h value
    """
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2


def run_logistic_regression(df):
    """
    Run mixed-effects logistic regression predicting deference.
    
    Args:
        df: DataFrame with experimental data in long format
        
    Returns:
        Model results
    """
    # First, reshape to long format if needed
    df_long = reshape_to_long_format(df)
    
    # Run logistic regression
    # Note: statsmodels doesn't have easy mixed effects logistic regression
    # For true mixed effects, you'd want to use R's lme4 or pymer4
    # Here we'll use a simpler GLM with robust standard errors
    
    formula = 'deferred ~ C(model_id) * C(status_condition) * C(model_type) + initial_disagreement'
    
    model = smf.logit(formula, data=df_long).fit(cov_type='HC3')
    
    return model


def reshape_to_long_format(df):
    """
    Reshape wide-format data to long format for regression analysis.
    
    Args:
        df: DataFrame in wide format (one row per trial)
        
    Returns:
        DataFrame in long format (two rows per trial, one for each model)
    """
    # Create M1 rows
    m1_data = df.copy()
    m1_data['model_id'] = 'M1'
    m1_data['deferred'] = m1_data['m1_toward_m2']
    m1_data['initial_rating'] = m1_data['m1_initial']
    m1_data['final_rating'] = m1_data['m1_final']
    m1_data['partner_rating'] = m1_data['m2_initial']
    
    # Create M2 rows
    m2_data = df.copy()
    m2_data['model_id'] = 'M2'
    m2_data['deferred'] = m2_data['m2_toward_m1']
    m2_data['initial_rating'] = m2_data['m2_initial']
    m2_data['final_rating'] = m2_data['m2_final']
    m2_data['partner_rating'] = m1_data['m1_initial']
    
    # Combine
    df_long = pd.concat([m1_data, m2_data], ignore_index=True)
    
    # Add derived variables for regression
    df_long['status_condition'] = df_long.apply(
        lambda row: 'standard' if row['m1_profile'] != row['m2_profile'] else 'equal',
        axis=1
    )
    df_long['model_type'] = 'same'  # Update based on your actual data
    
    return df_long


def compare_conditions(df, condition1, condition2, metric='m2_toward_m1'):
    """
    Statistical comparison between two conditions.
    
    Args:
        df: DataFrame with experimental data
        condition1: Name/identifier of first condition
        condition2: Name/identifier of second condition
        metric: Column name to compare
        
    Returns:
        Dictionary with test results
    """
    data1 = df[df['condition'] == condition1][metric]
    data2 = df[df['condition'] == condition2][metric]
    
    # Proportion test (for binary deference data)
    count1 = data1.sum()
    count2 = data2.sum()
    n1 = len(data1)
    n2 = len(data2)
    
    # Chi-square test
    contingency_table = np.array([
        [count1, n1 - count1],
        [count2, n2 - count2]
    ])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Effect size (Cohen's h)
    p1 = count1 / n1
    p2 = count2 / n2
    effect_size = cohens_h(p1, p2)
    
    return {
        'condition1': condition1,
        'condition2': condition2,
        'condition1_rate': p1,
        'condition2_rate': p2,
        'difference': p1 - p2,
        'chi2': chi2,
        'p_value': p_value,
        'cohens_h': effect_size,
        'effect_interpretation': interpret_cohens_h(effect_size)
    }


def interpret_cohens_h(h):
    """Interpret Cohen's h effect size."""
    abs_h = abs(h)
    if abs_h < 0.2:
        return 'negligible'
    elif abs_h < 0.5:
        return 'small'
    elif abs_h < 0.8:
        return 'medium'
    else:
        return 'large'


def planned_contrasts(df):
    """
    Run all planned contrasts from the statistical analysis plan.
    
    Args:
        df: DataFrame with all conditions
        
    Returns:
        DataFrame with contrast results
    """
    contrasts = [
        # Pure status effect tests
        ('same_standard', 'same_equal', 'Pure status effect (same models)'),
        ('same_standard', 'same_none', 'Status info effect (same models)'),
        
        # Pure capability effect tests
        ('different_equal', 'same_equal', 'Pure capability effect'),
        ('different_none', 'same_none', 'Capability baseline effect'),
        
        # Combined effect tests
        ('different_standard', 'different_equal', 'Status enhancement of capability'),
        
        # Status-capability conflict
        ('different_reversed', 'different_standard', 'Status reversal impact'),
    ]
    
    results = []
    for cond1, cond2, description in contrasts:
        if cond1 in df['condition'].values and cond2 in df['condition'].values:
            result = compare_conditions(df, cond1, cond2, metric='m2_toward_m1')
            result['contrast_description'] = description
            results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Apply multiple testing correction
    if len(results_df) > 0:
        _, p_corrected, _, _ = multipletests(
            results_df['p_value'],
            alpha=0.05,
            method='holm'
        )
        results_df['p_value_corrected'] = p_corrected
        results_df['significant_corrected'] = p_corrected < 0.05
    
    return results_df


def analyze_initial_disagreement_moderation(df):
    """
    Test whether initial disagreement moderates status effects.
    
    Args:
        df: DataFrame with experimental data
        
    Returns:
        Analysis results
    """
    # Bin initial disagreement
    df['disagreement_bin'] = pd.cut(
        df['initial_disagreement'],
        bins=[0, 0.1, 0.2, 0.3, 1.0],
        labels=['very_small', 'small', 'medium', 'large']
    )
    
    # Calculate deference rates by disagreement level
    moderation_results = df.groupby(['condition', 'disagreement_bin']).agg({
        'm2_toward_m1': ['mean', 'count'],
        'm1_toward_m2': ['mean', 'count']
    }).reset_index()
    
    return moderation_results


def create_comprehensive_report(df):
    """
    Generate a comprehensive statistical report.
    
    Args:
        df: DataFrame with experimental data
        
    Returns:
        Dictionary containing all statistical analyses
    """
    report = {}
    
    # 1. Descriptive statistics by condition
    report['descriptives'] = calculate_deference_rates_with_ci(df, ['condition'])
    
    # 2. Planned contrasts
    report['contrasts'] = planned_contrasts(df)
    
    # 3. Effect sizes for key comparisons
    report['effect_sizes'] = calculate_all_effect_sizes(df)
    
    # 4. Moderation analysis
    report['moderation'] = analyze_initial_disagreement_moderation(df)
    
    # 5. Regression results (if applicable)
    # report['regression'] = run_logistic_regression(df)
    
    return report


def calculate_all_effect_sizes(df):
    """
    Calculate Cohen's h for all pairwise condition comparisons.
    
    Args:
        df: DataFrame with experimental data
        
    Returns:
        DataFrame with effect sizes
    """
    conditions = df['condition'].unique()
    results = []
    
    for cond1, cond2 in combinations(conditions, 2):
        result = compare_conditions(df, cond1, cond2)
        results.append(result)
    
    return pd.DataFrame(results)


def print_statistical_report(report):
    """
    Print a formatted statistical report.
    
    Args:
        report: Dictionary from create_comprehensive_report
    """
    print("="*80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
    print("="*80)
    
    print("\n1. DESCRIPTIVE STATISTICS BY CONDITION")
    print("-"*80)
    desc = report['descriptives']
    for _, row in desc.iterrows():
        print(f"\nCondition: {row['condition']}")
        print(f"  N trials: {row['n_trials']}")
        print(f"  M1 deference: {row['m1_deference_rate']:.1%} "
              f"[{row['m1_ci_low']:.1%}, {row['m1_ci_high']:.1%}]")
        print(f"  M2 deference: {row['m2_deference_rate']:.1%} "
              f"[{row['m2_ci_low']:.1%}, {row['m2_ci_high']:.1%}]")
        print(f"  Asymmetry: {row['asymmetry']:.1%} "
              f"[{row['asymmetry_ci_low']:.1%}, {row['asymmetry_ci_high']:.1%}]")
    
    print("\n2. PLANNED CONTRASTS")
    print("-"*80)
    contrasts = report['contrasts']
    for _, row in contrasts.iterrows():
        print(f"\n{row['contrast_description']}")
        print(f"  {row['condition1']}: {row['condition1_rate']:.1%}")
        print(f"  {row['condition2']}: {row['condition2_rate']:.1%}")
        print(f"  Difference: {row['difference']:.1%}")
        print(f"  χ²({1}) = {row['chi2']:.2f}, p = {row['p_value']:.4f} "
              f"(corrected p = {row['p_value_corrected']:.4f})")
        print(f"  Cohen's h = {row['cohens_h']:.2f} ({row['effect_interpretation']})")
        print(f"  Significant (corrected): {'Yes' if row['significant_corrected'] else 'No'}")


# Example usage
if __name__ == "__main__":
    # Example: Load data from multiple conditions
    csv_files = {
        # 'same_standard': 'results/condition1_same_standard_imdb.csv',
        # 'same_equal': 'results/condition2_same_equal_imdb.csv',
        # 'different_standard': 'results/condition3_different_standard_imdb.csv',
        # 'different_reversed': 'results/condition4_different_reversed_imdb.csv',
        # 'different_equal': 'results/condition5_different_equal_imdb.csv',
        # 'different_none': 'results/condition6_different_none_imdb.csv',
        # 'same_standard': 'results/condition1_same_standard_imdb-v1.csv',
        # 'same_equal': 'results/condition2_same_equal_imdb-v1.csv',
        # 'different_standard': 'results/condition3_different_standard_imdb-v1.csv',
        # 'different_reversed': 'results/condition4_different_reversed_imdb-v1.csv',
        # 'different_equal': 'results/condition5_different_equal_imdb-v1.csv',
        # 'different_none': 'results/condition6_different_none_imdb-v1.csv',
        'same_standard': 'results/condition1_same_standard_imdb-v2.csv',
        'same_equal': 'results/condition2_same_equal_imdb-v2.csv',
        'different_standard': 'results/condition3_different_standard_imdb-v2.csv',
        'different_reversed': 'results/condition4_different_reversed_imdb-v2.csv',
        'different_equal': 'results/condition5_different_equal_imdb-v2.csv',
        'different_none': 'results/condition6_different_none_imdb-v2.csv',
        # Add more as needed
    }
    
    # Load and combine data
    df = load_and_prepare_data(csv_files)
    
    # Generate comprehensive report
    report = create_comprehensive_report(df)
    
    # Print results
    print_statistical_report(report)
    
    # Save results to file
    report['descriptives'].to_csv('results/descriptive_statistics-v2.csv', index=False)
    report['contrasts'].to_csv('results/planned_contrasts-v2.csv', index=False)