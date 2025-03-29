"""
Healthcare Website Accessibility Trend Analysis
--------------------------------------------
This script analyzes accessibility trends in healthcare websites,
identifies critical issues, and provides recommendations for improvement.
It focuses on healthcare-specific accessibility needs and WCAG compliance.

Author: Jiaqi Yu
Date: 11.03.2025
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_latest_data():
    """Load the most recent detailed accessibility data."""
    try:
        # Find most recent detailed results file
        import glob
        files = glob.glob('lighthouse_results_detailed_*.csv')
        if not files:
            print("No detailed results found, loading from lighthouse_results.csv")
            return pd.read_csv('lighthouse_results.csv')
        
        latest_file = max(files)
        print(f"Loading data from {latest_file}")
        return pd.read_csv(latest_file)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def identify_critical_sites(df):
    """
    Identify websites with critical accessibility issues.
    Returns sites sorted by severity of issues.
    """
    # Calculate composite score based on multiple factors
    df['composite_score'] = (
        df['Accessibility'] * 0.4 +  # Base accessibility score
        (100 - df['High_Severity_Issues'] * 10) * 0.3 +  # Penalize high severity issues
        (100 - df['P0_Issues'] * 20) * 0.3  # Heavily penalize P0 issues
    )
    
    # Create detailed analysis for each site
    site_analysis = []
    for idx, row in df.iterrows():
        issues_by_category = defaultdict(int)
        for col in df.columns:
            if col.endswith('_details') and 'Accessibility_' in col:
                if pd.notna(row[col]) and row[col]:
                    category = col.replace('Accessibility_', '').replace('_details', '')
                    issues_by_category[category] = len(row[col].split(';'))
        
        site_analysis.append({
            'URL': row['URL'],
            'Accessibility_Score': row['Accessibility'],
            'Composite_Score': row['composite_score'],
            'High_Severity_Issues': row['High_Severity_Issues'],
            'P0_Issues': row['P0_Issues'],
            'Total_Issues': row['Total_Issues'],
            'Main_Issue_Categories': sorted(
                issues_by_category.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]  # Top 3 issue categories
        })
    
    # Sort by composite score (ascending = worst first)
    return sorted(site_analysis, key=lambda x: x['Composite_Score'])

def analyze_healthcare_specific_issues(df):
    """
    Analyze accessibility issues particularly relevant to healthcare websites.
    """
    healthcare_critical_features = {
        'forms': 'Critical for appointment scheduling and patient forms',
        'color_contrast': 'Essential for reading medical information',
        'keyboard': 'Important for motor-impaired users',
        'aria': 'Crucial for screen reader users accessing medical info',
        'navigation': 'Key for finding emergency information',
        'language': 'Important for multi-language medical information'
    }
    
    feature_stats = defaultdict(lambda: {
        'total_issues': 0,
        'affected_sites': 0,
        'high_severity': 0,
        'example_urls': []
    })
    
    for idx, row in df.iterrows():
        for col in df.columns:
            if col.endswith('_details'):
                for feature in healthcare_critical_features:
                    if feature in col.lower() and pd.notna(row[col]) and row[col]:
                        feature_stats[feature]['total_issues'] += len(row[col].split(';'))
                        feature_stats[feature]['affected_sites'] += 1
                        if 'Severity: High' in row[col]:
                            feature_stats[feature]['high_severity'] += 1
                        if len(feature_stats[feature]['example_urls']) < 3:
                            feature_stats[feature]['example_urls'].append(row['URL'])
    
    return feature_stats, healthcare_critical_features

def generate_recommendations(feature_stats, healthcare_critical_features):
    """
    Generate specific recommendations based on the analysis.
    """
    recommendations = []
    
    for feature, stats in feature_stats.items():
        if stats['affected_sites'] > 0:
            severity = 'HIGH' if stats['high_severity'] / stats['affected_sites'] > 0.3 else 'MEDIUM'
            recommendations.append({
                'feature': feature,
                'importance': healthcare_critical_features[feature],
                'severity': severity,
                'affected_sites': stats['affected_sites'],
                'total_issues': stats['total_issues'],
                'recommendation': get_healthcare_recommendation(feature),
                'example_urls': stats['example_urls']
            })
    
    return sorted(recommendations, key=lambda x: (
        0 if x['severity'] == 'HIGH' else 1,
        -x['affected_sites']
    ))

def get_healthcare_recommendation(feature):
    """
    Provide healthcare-specific accessibility recommendations.
    """
    recommendations = {
        'forms': """
        - Implement step-by-step form wizards for complex medical forms
        - Provide clear error messages with medical terminology explanations
        - Allow form saving for lengthy medical history forms
        - Ensure all form fields have clear labels and instructions
        - Add support for medical document uploads
        """,
        'color_contrast': """
        - Ensure medical charts and graphs are readable with high contrast
        - Provide alternative color schemes for color-blind users
        - Use patterns in addition to colors for medical data visualization
        - Maintain WCAG 2.1 AAA contrast ratios for critical medical information
        """,
        'keyboard': """
        - Ensure all interactive elements are keyboard accessible
        - Provide keyboard shortcuts for emergency information access
        - Implement skip links for quick navigation to important sections
        - Ensure proper focus management for medical forms
        """,
        'aria': """
        - Add clear ARIA labels for medical terms and abbreviations
        - Implement proper ARIA landmarks for critical sections
        - Ensure dynamic content updates are announced appropriately
        - Provide clear status messages for form submissions
        """,
        'navigation': """
        - Implement clear navigation paths to emergency information
        - Provide consistent navigation patterns across all pages
        - Add breadcrumbs for complex medical information sections
        - Ensure clear wayfinding in multi-step processes
        """,
        'language': """
        - Implement proper language declarations for multi-language content
        - Provide language switching options on all pages
        - Ensure medical terminology is properly marked up
        - Support right-to-left languages where needed
        """
    }
    
    return recommendations.get(feature, "Implement healthcare-specific accessibility best practices.")

def calculate_compliance_metrics(df):
    """Calculate compliance metrics and percentages."""
    metrics = {}
    
    # Overall compliance levels
    metrics['fully_compliant'] = (df['Accessibility'] >= 90).mean() * 100
    metrics['partially_compliant'] = ((df['Accessibility'] >= 70) & (df['Accessibility'] < 90)).mean() * 100
    metrics['non_compliant'] = (df['Accessibility'] < 70).mean() * 100
    
    # Critical issues metrics
    metrics['sites_with_p0_issues'] = (df['P0_Issues'] > 0).mean() * 100
    metrics['sites_with_high_severity'] = (df['High_Severity_Issues'] > 0).mean() * 100
    
    # Feature-specific compliance
    for col in df.columns:
        if col.startswith('Accessibility_') and col.endswith('_details'):
            feature = col.replace('Accessibility_', '').replace('_details', '')
            metrics[f'{feature}_issues_pct'] = (df[col].notna() & (df[col] != '')).mean() * 100
    
    return metrics

def calculate_score_distribution(df):
    """Calculate score distribution metrics."""
    ranges = {
        '90-100': ((df['Accessibility'] >= 90) & (df['Accessibility'] <= 100)).mean() * 100,
        '80-89': ((df['Accessibility'] >= 80) & (df['Accessibility'] < 90)).mean() * 100,
        '70-79': ((df['Accessibility'] >= 70) & (df['Accessibility'] < 80)).mean() * 100,
        '60-69': ((df['Accessibility'] >= 60) & (df['Accessibility'] < 70)).mean() * 100,
        'Below 60': (df['Accessibility'] < 60).mean() * 100
    }
    return ranges

def analyze_issue_correlations(df):
    """Analyze correlations between different types of issues."""
    issue_cols = [col for col in df.columns if col.startswith('Accessibility_') and col.endswith('_details')]
    issue_presence = pd.DataFrame()
    
    for col in issue_cols:
        feature = col.replace('Accessibility_', '').replace('_details', '')
        issue_presence[feature] = df[col].notna() & (df[col] != '')
    
    return issue_presence.corr()

def plot_healthcare_trends(df, feature_stats, compliance_metrics, score_distribution):
    """
    Create enhanced visualizations for healthcare accessibility trends.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (24, 16)
    
    fig = plt.figure()
    fig.suptitle('Healthcare Websites Accessibility Analysis', fontsize=16, y=0.95)
    
    # 1. Compliance Distribution (Pie Chart)
    ax1 = plt.subplot(231)
    compliance_data = [
        compliance_metrics['fully_compliant'],
        compliance_metrics['partially_compliant'],
        compliance_metrics['non_compliant']
    ]
    labels = ['Fully Compliant\n(≥90%)', 'Partially Compliant\n(70-89%)', 'Non-Compliant\n(<70%)']
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    ax1.pie(compliance_data, labels=labels, colors=colors, autopct='%1.1f%%')
    ax1.set_title('Compliance Distribution')
    
    # 2. Score Distribution (Bar Chart)
    ax2 = plt.subplot(232)
    ranges = list(score_distribution.keys())
    percentages = list(score_distribution.values())
    ax2.bar(ranges, percentages, color='#3498db')
    ax2.set_title('Score Distribution')
    ax2.set_ylabel('Percentage of Sites')
    plt.xticks(rotation=45)
    
    # 3. Critical Issues Impact (Horizontal Bar Chart)
    ax3 = plt.subplot(233)
    critical_metrics = {
        'P0 Issues': compliance_metrics['sites_with_p0_issues'],
        'High Severity': compliance_metrics['sites_with_high_severity']
    }
    y_pos = np.arange(len(critical_metrics))
    ax3.barh(y_pos, list(critical_metrics.values()), color='#e67e22')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(list(critical_metrics.keys()))
    ax3.set_title('Sites with Critical Issues (%)')
    ax3.set_xlabel('Percentage of Sites')
    
    # 4. Feature Compliance (Sorted Bar Chart)
    ax4 = plt.subplot(234)
    feature_metrics = {k: v for k, v in compliance_metrics.items() if k.endswith('_issues_pct')}
    sorted_features = sorted(feature_metrics.items(), key=lambda x: x[1], reverse=True)
    features = [f.replace('_issues_pct', '').replace('_', ' ').title() for f, _ in sorted_features]
    values = [v for _, v in sorted_features]
    ax4.bar(features, values, color='#9b59b6')
    ax4.set_title('Feature-Specific Issues')
    ax4.set_ylabel('Percentage of Sites Affected')
    plt.xticks(rotation=45, ha='right')
    
    # 5. Accessibility vs Issues Scatter
    ax5 = plt.subplot(235)
    ax5.scatter(df['Accessibility'], df['Total_Issues'], alpha=0.5, color='#1abc9c')
    ax5.set_title('Accessibility Score vs Total Issues')
    ax5.set_xlabel('Accessibility Score')
    ax5.set_ylabel('Number of Issues')
    
    # 6. Priority Distribution
    ax6 = plt.subplot(236)
    priority_cols = [col for col in df.columns if '_priority' in col]
    priority_counts = defaultdict(int)
    for col in priority_cols:
        for priorities in df[col].dropna():
            for priority in priorities.split(';'):
                if priority.strip().startswith('P'):
                    priority_counts[priority.strip()[:2]] += 1
    
    priorities = sorted(priority_counts.keys())
    counts = [priority_counts[p] for p in priorities]
    ax6.bar(priorities, counts, color='#34495e')
    ax6.set_title('Issue Priority Distribution')
    ax6.set_ylabel('Number of Issues')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'healthcare_accessibility_trends_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_detailed_analysis(df, compliance_metrics, score_distribution):
    """Print comprehensive analysis with percentages and statistics."""
    print("\n=== Detailed Accessibility Analysis ===\n")
    
    print("1. Overall Compliance Metrics:")
    print(f"- Fully Compliant Sites (≥90%): {compliance_metrics['fully_compliant']:.1f}%")
    print(f"- Partially Compliant Sites (70-89%): {compliance_metrics['partially_compliant']:.1f}%")
    print(f"- Non-Compliant Sites (<70%): {compliance_metrics['non_compliant']:.1f}%")
    
    print("\n2. Score Distribution:")
    for range_name, percentage in score_distribution.items():
        print(f"- {range_name}: {percentage:.1f}%")
    
    print("\n3. Critical Issues:")
    print(f"- Sites with P0 Issues: {compliance_metrics['sites_with_p0_issues']:.1f}%")
    print(f"- Sites with High Severity Issues: {compliance_metrics['sites_with_high_severity']:.1f}%")
    
    print("\n4. Feature-Specific Analysis:")
    feature_metrics = {k: v for k, v in compliance_metrics.items() if k.endswith('_issues_pct')}
    for feature, percentage in sorted(feature_metrics.items(), key=lambda x: x[1], reverse=True):
        feature_name = feature.replace('_issues_pct', '').replace('_', ' ').title()
        print(f"- {feature_name}: {percentage:.1f}% of sites affected")
    
    print("\n5. Statistical Summary:")
    print(f"- Mean Accessibility Score: {df['Accessibility'].mean():.1f}")
    print(f"- Median Accessibility Score: {df['Accessibility'].median():.1f}")
    print(f"- Standard Deviation: {df['Accessibility'].std():.1f}")
    print(f"- Sites with Perfect Score (100): {(df['Accessibility'] == 100).mean() * 100:.1f}%")
    print(f"- Sites Below Average: {(df['Accessibility'] < df['Accessibility'].mean()).mean() * 100:.1f}%")

def main():
    # Load data
    print("Loading accessibility data...")
    df = load_latest_data()
    if df is None or len(df) == 0:
        print("Error: Could not load data or empty dataset. Exiting...")
        return
    
    # Identify critical sites
    print("\nAnalyzing critical sites...")
    critical_sites = identify_critical_sites(df)
    
    # Analyze healthcare-specific issues
    print("\nAnalyzing healthcare-specific accessibility issues...")
    feature_stats, healthcare_critical_features = analyze_healthcare_specific_issues(df)
    
    # Calculate metrics
    print("Calculating compliance metrics...")
    compliance_metrics = calculate_compliance_metrics(df)
    score_distribution = calculate_score_distribution(df)
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    recommendations = generate_recommendations(feature_stats, healthcare_critical_features)
    
    # Create visualizations
    print("\nGenerating enhanced visualizations...")
    plot_healthcare_trends(df, feature_stats, compliance_metrics, score_distribution)
    
    # Print detailed analysis
    print_detailed_analysis(df, compliance_metrics, score_distribution)
    
    # Print detailed report
    print("\n=== Healthcare Accessibility Analysis Report ===")
    print("\nTop 5 Most Critical Sites:")
    for site in critical_sites[:5]:
        print(f"\nURL: {site['URL']}")
        print(f"Accessibility Score: {site['Accessibility_Score']:.2f}")
        print(f"Composite Score: {site['Composite_Score']:.2f}")
        print(f"High Severity Issues: {site['High_Severity_Issues']}")
        print(f"P0 (Critical) Issues: {site['P0_Issues']}")
        print("Main Issue Categories:")
        for category, count in site['Main_Issue_Categories']:
            print(f"  - {category}: {count} issues")
    
    print("\nKey Recommendations by Feature:")
    for rec in recommendations:
        print(f"\n{rec['feature'].upper()} (Severity: {rec['severity']})")
        print(f"Importance: {rec['importance']}")
        print(f"Affected Sites: {rec['affected_sites']}")
        print("Recommendations:")
        for line in rec['recommendation'].strip().split('\n'):
            print(f"  {line.strip()}")
    
    print("\nAnalysis completed! Visualizations and detailed report have been saved.")

if __name__ == "__main__":
    main() 