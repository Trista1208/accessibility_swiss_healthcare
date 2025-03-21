"""
Accessibility Analysis and Visualization Tool
------------------------------------------
This script analyzes accessibility data from Lighthouse scans and generates
comprehensive visualizations and reports. It processes data from CSV files
containing Lighthouse audit results and creates detailed statistical analysis
and visualizations.

Main components:
1. Data Loading and Preparation
2. Statistical Analysis
3. Visualization Generation
4. Report Generation

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
from datetime import datetime
import base64
from io import BytesIO
from collections import defaultdict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_and_prepare_data():
    """
    Load and prepare the Lighthouse results data for analysis.
    
    Returns:
        pandas.DataFrame: Prepared dataframe with accessibility metrics
        None: If there's an error loading the data
    """
    try:
        # Find the most recent detailed results file
        result_files = list(Path('.').glob('lighthouse_results_detailed_*.csv'))
        if result_files:
            latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
            print(f"Loading data from {latest_file}")
            df = pd.read_csv(latest_file)
        else:
            print("No detailed results found, loading from lighthouse_results.csv")
            df = pd.read_csv('lighthouse_results.csv')
        
        # Convert score columns to numeric values
        score_columns = ['Performance', 'Accessibility', 'Best Practices', 'SEO']
        for col in score_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def setup_plot_style():
    """
    Configure the plotting style for consistent visualization appearance.
    Sets up seaborn style with custom parameters for better readability.
    """
    try:
        # Reset to default style first
        plt.style.use('default')
        
        # Use seaborn settings for better visual appeal
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
        
        # Configure plot parameters for readability
        plt.rcParams.update({
            'figure.figsize': [10, 6],
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9
        })
    except Exception as e:
        print(f"Warning: Could not set plot style: {str(e)}")

# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_accessibility_issues(df):
    """
    Analyze accessibility issues and their distribution across categories.
    
    Args:
        df (pandas.DataFrame): Input dataframe with accessibility data
        
    Returns:
        dict: Counts of issues by category
    """
    # Count total issues per URL
    df['Total_Issues'] = df['Accessibility Issues'].fillna('').apply(
        lambda x: len(x.split(';')) if x else 0
    )
    
    # Get issue categories and count issues
    issue_columns = [col for col in df.columns 
                    if col.startswith('Accessibility_') and col.endswith('_details')]
    
    category_counts = {}
    for col in issue_columns:
        category = col.replace('Accessibility_', '').replace('_details', '')
        count = df[col].fillna('').apply(
            lambda x: len(x.split(';')) if x else 0
        ).sum()
        category_counts[category] = count
    
    return category_counts

def analyze_severity_distribution(df):
    """
    Analyze the distribution of issue severities (P0-P4).
    
    Args:
        df (pandas.DataFrame): Input dataframe with accessibility data
        
    Returns:
        dict: Counts of issues by priority level
    """
    severity_columns = [col for col in df.columns if col.endswith('_priority')]
    
    severity_counts = {
        'P0': 0,  # Critical issues
        'P1': 0,  # High priority
        'P2': 0,  # Medium priority
        'P3': 0,  # Low priority
        'P4': 0   # To be evaluated
    }
    
    for col in severity_columns:
        for priorities in df[col].fillna('').astype(str):
            for priority in priorities.split(';'):
                for p in severity_counts.keys():
                    if p in priority:
                        severity_counts[p] += 1
    
    return severity_counts

def analyze_wcag_guidelines(df):
    """
    Analyze WCAG guidelines coverage and violations.
    
    Args:
        df (pandas.DataFrame): Input dataframe with accessibility data
        
    Returns:
        dict: Sorted dictionary of WCAG guideline violations
    """
    wcag_columns = [col for col in df.columns if col.endswith('_wcag')]
    
    guideline_counts = {}
    for col in wcag_columns:
        for guidelines in df[col].fillna('').astype(str):
            for guideline in guidelines.split(';'):
                guideline = guideline.strip()
                if guideline:
                    guideline_counts[guideline] = guideline_counts.get(guideline, 0) + 1
    
    return dict(sorted(guideline_counts.items(), key=lambda x: x[1], reverse=True))

def generate_statistical_summary(df):
    """
    Generate comprehensive statistical summary of accessibility scores.
    
    Args:
        df (pandas.DataFrame): Input dataframe with accessibility data
        
    Returns:
        pandas.DataFrame: Statistical summary including mean, std, etc.
    """
    score_columns = ['Performance', 'Accessibility', 'Best Practices', 'SEO']
    
    # Calculate basic statistics
    stats_summary = df[score_columns].describe()
    
    # Add additional statistical measures
    for col in score_columns:
        stats_summary.loc['skew', col] = stats.skew(df[col].dropna())
        stats_summary.loc['kurtosis', col] = stats.kurtosis(df[col].dropna())
        stats_summary.loc['% below 50', col] = (df[col] < 50).mean() * 100
        stats_summary.loc['% above 90', col] = (df[col] > 90).mean() * 100
    
    return stats_summary

def generate_detailed_accessibility_stats(df):
    """
    Generate detailed statistical analysis of accessibility scores.
    
    Args:
        df (pandas.DataFrame): Input dataframe with accessibility data
        
    Returns:
        dict: Detailed statistics including score ranges and issue counts
    """
    # Define score ranges for analysis
    ranges = [(0, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
    
    # Calculate basic statistics
    stats = {
        'Mean Score': df['Accessibility'].mean(),
        'Median Score': df['Accessibility'].median(),
        'Standard Deviation': df['Accessibility'].std(),
        'Minimum Score': df['Accessibility'].min(),
        'Maximum Score': df['Accessibility'].max(),
    }
    
    # Calculate statistics for each score range
    total_sites = len(df)
    for start, end in ranges:
        range_count = sum((df['Accessibility'] >= start) & 
                         (df['Accessibility'] < (end if end < 100 else end + 1)))
        percentage = (range_count / total_sites) * 100
        stats[f'Sites {start}-{end}'] = range_count
        stats[f'Percentage {start}-{end}'] = percentage
    
    # Add issue-related statistics
    stats.update({
        'Average Issues per Site': df['Total_Issues'].mean(),
        'Average High Severity Issues': df['High_Severity_Issues'].mean(),
        'Sites with No Issues': sum(df['Total_Issues'] == 0),
        'Sites with Critical Issues (P0)': sum(df['P0_Issues'] > 0)
    })
    
    return stats

def analyze_accessibility_correlation(df):
    """
    Analyze correlations between accessibility scores and other metrics.
    
    Args:
        df (pandas.DataFrame): Input dataframe with accessibility data
        
    Returns:
        pandas.Series: Correlation coefficients with accessibility score
    """
    try:
        return df[['Accessibility', 'Performance', 'Best Practices', 'SEO', 
                  'Total_Issues', 'High_Severity_Issues']].corr()['Accessibility']
    except Exception as e:
        print(f"Error analyzing correlations: {str(e)}")
        return None

def analyze_wcag_compliance(df):
    """
    Analyze WCAG 2.1 AA compliance across websites.
    
    Args:
        df (pandas.DataFrame): Input dataframe with accessibility data
        
    Returns:
        dict: Compliance statistics and details
    """
    # Define WCAG 2.1 AA success criteria
    wcag_aa_criteria = {
        '1.4.3': 'Contrast (Minimum)',
        '1.4.11': 'Non-text Contrast',
        '2.4.6': 'Headings and Labels',
        '2.4.7': 'Focus Visible',
        '3.1.2': 'Language of Parts',
        '1.3.4': 'Orientation',
        '1.3.5': 'Identify Input Purpose',
        '1.4.10': 'Reflow',
        '1.4.12': 'Text Spacing',
        '1.4.13': 'Content on Hover or Focus',
        '4.1.3': 'Status Messages'
    }
    
    # Initialize compliance tracking
    total_sites = len(df)
    compliance_stats = {
        'fully_compliant': 0,
        'partially_compliant': 0,
        'non_compliant': 0,
        'criteria_compliance': defaultdict(int)
    }
    
    # Analyze each website's compliance
    wcag_columns = [col for col in df.columns if '_wcag' in col]
    
    for idx, row in df.iterrows():
        violations = set()
        for col in wcag_columns:
            if pd.notna(row[col]) and row[col]:
                guidelines = [g.strip() for g in str(row[col]).split(';')]
                for guideline in guidelines:
                    for criterion in wcag_aa_criteria:
                        if criterion in guideline:
                            violations.add(criterion)
        
        # Determine compliance level
        if not violations:
            compliance_stats['fully_compliant'] += 1
        elif len(violations) <= 2:  # Allow for minor violations
            compliance_stats['partially_compliant'] += 1
        else:
            compliance_stats['non_compliant'] += 1
            
        # Track individual criteria compliance
        for criterion in wcag_aa_criteria:
            if criterion not in violations:
                compliance_stats['criteria_compliance'][criterion] += 1
    
    # Calculate percentages
    compliance_stats['fully_compliant_pct'] = (compliance_stats['fully_compliant'] / total_sites) * 100
    compliance_stats['partially_compliant_pct'] = (compliance_stats['partially_compliant'] / total_sites) * 100
    compliance_stats['non_compliant_pct'] = (compliance_stats['non_compliant'] / total_sites) * 100
    
    # Calculate criteria compliance percentages
    for criterion in wcag_aa_criteria:
        compliance_stats[f'{criterion}_pct'] = (compliance_stats['criteria_compliance'][criterion] / total_sites) * 100
    
    return compliance_stats, wcag_aa_criteria

# ============================================================================
# Visualization Functions
# ============================================================================

def generate_png_report(df, stats_summary, category_counts, severity_counts, 
                       wcag_guidelines, accessibility_stats, correlations, timestamp):
    """
    Generate a comprehensive PNG report combining all visualizations and statistics.
    
    Args:
        df (pandas.DataFrame): Input dataframe with accessibility data
        stats_summary (pandas.DataFrame): Statistical summary
        category_counts (dict): Issue counts by category
        severity_counts (dict): Issue counts by severity
        wcag_guidelines (dict): WCAG guideline violation counts
        accessibility_stats (dict): Detailed accessibility statistics
        correlations (pandas.Series): Correlation coefficients
        timestamp (str): Report generation timestamp
    
    Returns:
        bool: True if report generation was successful, False otherwise
    """
    try:
        # Create main figure
        fig = plt.figure(figsize=(24, 42))  # Increased height for new plot
        fig.suptitle(f'Accessibility Analysis Report - {timestamp}', 
                    fontsize=20, y=0.95)
        
        # Set up grid layout with additional row
        gs = fig.add_gridspec(7, 2, hspace=0.4, wspace=0.3)  # Added one more row
        
        # Generate all subplots
        _create_score_distribution_plot(df, fig.add_subplot(gs[0, 0]))
        _create_score_ranges_plot(df, fig.add_subplot(gs[0, 1]))
        _create_issues_scatter_plot(df, fig.add_subplot(gs[1, 0]))
        _create_priority_distribution_plot(severity_counts, fig.add_subplot(gs[1, 1]))
        _create_wcag_violations_plot(wcag_guidelines, fig.add_subplot(gs[2, :]))
        _create_statistical_summary_table(stats_summary, fig.add_subplot(gs[3, :]))
        _create_detailed_stats_table(accessibility_stats, fig.add_subplot(gs[4, :]))
        _create_correlations_plot(correlations, fig.add_subplot(gs[5, :]))
        
        # Add WCAG compliance plot
        compliance_stats, wcag_criteria = analyze_wcag_compliance(df)
        _create_wcag_compliance_plot(compliance_stats, wcag_criteria, fig.add_subplot(gs[6, :]))
        
        # Save the report
        plt.savefig(
            f'accessibility_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        
        # Print compliance summary
        print("\nWCAG 2.1 AA Compliance Summary:")
        print(f"Fully Compliant: {compliance_stats['fully_compliant']} sites ({compliance_stats['fully_compliant_pct']:.1f}%)")
        print(f"Partially Compliant: {compliance_stats['partially_compliant']} sites ({compliance_stats['partially_compliant_pct']:.1f}%)")
        print(f"Non-Compliant: {compliance_stats['non_compliant']} sites ({compliance_stats['non_compliant_pct']:.1f}%)")
        
        print("\nSuccess Criteria Compliance:")
        for criterion, name in wcag_criteria.items():
            pct = compliance_stats[f'{criterion}_pct']
            print(f"{criterion} {name}: {pct:.1f}% compliant")
        
        print("\nSuccessfully generated comprehensive PNG report")
        return True
        
    except Exception as e:
        print(f"Error generating PNG report: {str(e)}")
        return False

# Helper functions for report generation
def _create_score_distribution_plot(df, ax):
    """Create score distributions boxplot."""
    score_columns = ['Performance', 'Accessibility', 'Best Practices', 'SEO']
    sns.boxplot(data=df[score_columns], ax=ax)
    ax.set_title('Score Distributions')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3)

def _create_score_ranges_plot(df, ax):
    """Create accessibility score ranges plot."""
    score_ranges = pd.cut(df['Accessibility'], 
                         bins=[0, 50, 60, 70, 80, 90, 100],
                         labels=['<50', '50-60', '60-70', '70-80', '80-90', '90-100'])
    score_dist = score_ranges.value_counts().sort_index()
    colors = ['#ff4d4d', '#ff9933', '#ffcc00', '#99cc33', '#66cc66', '#339966']
    
    bars = ax.bar(score_dist.index, score_dist.values, color=colors)
    ax.set_title('Accessibility Score Distribution')
    ax.set_xlabel('Score Range')
    ax.set_ylabel('Number of Websites')
    
    # Add percentage labels
    total_sites = len(df)
    for i, v in enumerate(score_dist.values):
        percentage = (v / total_sites) * 100
        ax.text(i, v, f'{int(v)}\n({percentage:.1f}%)', 
                horizontalalignment='center', verticalalignment='bottom', fontsize=8)

def _create_issues_scatter_plot(df, ax):
    """Create accessibility vs issues scatter plot."""
    scatter = ax.scatter(df['Accessibility'], df['Total_Issues'], 
                        alpha=0.6, c=df['High_Severity_Issues'],
                        cmap='YlOrRd', s=50)
    plt.colorbar(scatter, ax=ax, label='High Severity Issues')
    ax.set_title('Accessibility Score vs Issues')
    ax.set_xlabel('Accessibility Score')
    ax.set_ylabel('Total Issues')
    ax.grid(True, alpha=0.3)

def _create_priority_distribution_plot(severity_counts, ax):
    """Create priority distribution plot."""
    priority_items = sorted(severity_counts.items())
    bars = ax.bar([x[0] for x in priority_items], [x[1] for x in priority_items])
    ax.set_title('Issue Priority Distribution')
    ax.set_xlabel('Priority Level')
    ax.set_ylabel('Number of Issues')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', horizontalalignment='center', 
                verticalalignment='bottom')

def _create_wcag_violations_plot(wcag_guidelines, ax):
    """Create WCAG violations plot."""
    top_10_wcag = dict(list(wcag_guidelines.items())[:10])
    bars = ax.bar(top_10_wcag.keys(), top_10_wcag.values())
    ax.set_title('Top 10 WCAG Violations')
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_ylabel('Number of Violations')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', horizontalalignment='center', 
                verticalalignment='bottom')

def _create_statistical_summary_table(stats_summary, ax):
    """Create statistical summary table."""
    ax.axis('tight')
    ax.axis('off')
    table_data = []
    table_columns = ['Metric', 'Performance', 'Accessibility', 'Best Practices', 'SEO']
    
    for idx in ['mean', 'std', '% below 50', '% above 90']:
        row = [idx]
        row.extend([f"{stats_summary.loc[idx, col]:.2f}" 
                   for col in table_columns[1:]])
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=table_columns, 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title('Statistical Summary', pad=20)

def _create_detailed_stats_table(accessibility_stats, ax):
    """Create detailed accessibility statistics table."""
    ax.axis('tight')
    ax.axis('off')
    detailed_stats = [
        ['Mean Score', f"{accessibility_stats['Mean Score']:.2f}"],
        ['Median Score', f"{accessibility_stats['Median Score']:.2f}"],
        ['Sites with No Issues', f"{accessibility_stats['Sites with No Issues']:.0f}"],
        ['Sites with Critical Issues', 
         f"{accessibility_stats['Sites with Critical Issues (P0)']:.0f}"],
        ['Average Issues per Site', 
         f"{accessibility_stats['Average Issues per Site']:.2f}"]
    ]
    
    table = ax.table(cellText=detailed_stats, 
                     colLabels=['Metric', 'Value'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title('Detailed Accessibility Statistics', pad=20)

def _create_correlations_plot(correlations, ax):
    """Create correlations plot."""
    if correlations is not None:
        corr_data = correlations[correlations.index != 'Accessibility']
        bars = ax.bar(range(len(corr_data)), corr_data.values)
        ax.set_title('Correlations with Accessibility Score')
        ax.set_xticks(range(len(corr_data)))
        ax.set_xticklabels(corr_data.index, rotation=45, 
                          horizontalalignment='right')
        ax.set_ylabel('Correlation Coefficient')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', horizontalalignment='center', 
                    verticalalignment='bottom')

def _create_wcag_compliance_plot(compliance_stats, wcag_criteria, ax):
    """Create WCAG 2.1 AA compliance plot."""
    # Prepare data for plotting
    compliance_levels = ['Fully Compliant', 'Partially Compliant', 'Non-Compliant']
    percentages = [
        compliance_stats['fully_compliant_pct'],
        compliance_stats['partially_compliant_pct'],
        compliance_stats['non_compliant_pct']
    ]
    
    # Create color-coded bars
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    bars = ax.bar(compliance_levels, percentages, color=colors)
    ax.set_title('WCAG 2.1 AA Compliance Distribution')
    ax.set_ylabel('Percentage of Websites')
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', 
                horizontalalignment='center',
                verticalalignment='bottom')
    
    # Adjust layout
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=0)

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function for the accessibility analysis tool."""
    # Set up plotting style
    setup_plot_style()
    
    # Load and prepare data
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    if df is None or len(df) == 0:
        print("Error: Could not load data or empty dataset. Exiting...")
        return
    
    # Analyze data
    print("\nAnalyzing accessibility issues...")
    category_counts = analyze_accessibility_issues(df)
    severity_counts = analyze_severity_distribution(df)
    wcag_guidelines = analyze_wcag_guidelines(df)
    
    # Generate statistics
    print("\nGenerating statistical summary...")
    stats_summary = generate_statistical_summary(df)
    accessibility_stats = generate_detailed_accessibility_stats(df)
    correlations = analyze_accessibility_correlation(df)
    
    # Generate report
    print("\nGenerating comprehensive PNG report...")
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    success = generate_png_report(
        df, stats_summary, category_counts, severity_counts, wcag_guidelines,
        accessibility_stats, correlations, timestamp
    )
    
    if success:
        print("\nAnalysis completed! Report has been saved as a PNG file.")
    else:
        print("\nError: Failed to generate the report.")

if __name__ == "__main__":
    main() 