import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from scipy import stats
import re
import shutil

# Set plot style and configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Create output directory for visualization
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f"accessibility_analysis_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

def load_data(file_path='lighthouse_scores_optimized.xlsx'):
    """Load and prepare the data for analysis"""
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    print(f"Data loaded: {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Display basic information
    print("\nData Overview:")
    print(f"- Score columns: {[col for col in df.columns if '_Score' in col]}")
    print(f"- Issue columns: {len([col for col in df.columns if 'A11y_Issue' in col])} columns")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing values:")
        print(missing[missing > 0])
    
    return df

def analyze_scores(df):
    """Analyze the main accessibility and performance scores"""
    print("\n===== SCORE ANALYSIS =====")
    
    # Basic statistics for scores
    score_cols = ['Performance_Score', 'Accessibility_Score', 'Best_Practices_Score', 'SEO_Score']
    score_stats = df[score_cols].describe()
    print("\nScore Statistics:")
    print(score_stats)
    
    # Calculate pass rates (≥90 is considered passing)
    pass_rates = {col: (df[col] >= 90).mean() * 100 for col in score_cols}
    print("\nPass Rates (≥90):")
    for score, rate in pass_rates.items():
        print(f"- {score}: {rate:.1f}%")
    
    # Visualize score distributions
    plt.figure(figsize=(14, 10))
    
    for i, col in enumerate(score_cols):
        plt.subplot(2, 2, i+1)
        sns.histplot(df[col].dropna(), kde=True, bins=20)
        plt.axvline(x=90, color='red', linestyle='--', label='Pass threshold (90)')
        plt.title(f"{col} Distribution")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.legend()
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, "score_distributions.png")
    plt.savefig(file_path)
    print(f"Saved score distributions to {file_path}")
    
    # Score correlation analysis - MODIFIED FOR BETTER VISIBILITY
    plt.figure(figsize=(14, 12))  # Increased figure size for better readability
    score_corr = df[score_cols].corr()
    mask = np.triu(np.ones_like(score_corr, dtype=bool))
    
    # Create heatmap with improved settings
    ax = sns.heatmap(score_corr, annot=True, mask=mask, vmin=-1, vmax=1, 
                cmap='coolwarm', annot_kws={"size": 14}, fmt='.2f',
                linewidths=1, cbar_kws={"shrink": 0.8})
    
    # Improve title and labels
    plt.title("Score Correlations", fontsize=20, pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, rotation=0)
    
    # Add more padding around the plot
    plt.tight_layout(pad=3.0)
    
    file_path = os.path.join(output_dir, "score_correlations.png")
    plt.savefig(file_path, bbox_inches='tight', dpi=300)  # Higher DPI for better quality
    print(f"Saved score correlations to {file_path}")
    
    # Score vs. Issue Count relationship
    plt.figure(figsize=(10, 6))
    
    # Create a clean dataframe with only the needed columns and no missing values
    plot_df = df[['Accessibility_Score', 'Accessibility_Issues_Count']].dropna()
    
    if len(plot_df) > 5:  # Ensure there's enough data to plot
        sns.scatterplot(x='Accessibility_Score', y='Accessibility_Issues_Count', data=plot_df)
        plt.title("Accessibility Score vs. Number of Issues")
        plt.xlabel("Accessibility Score")
        plt.ylabel("Number of Issues")
        
        # Trend line removed as it doesn't provide meaningful insights
    else:
        print("Warning: Not enough non-missing data to plot Accessibility Score vs Issues")
    
    file_path = os.path.join(output_dir, "accessibility_vs_issues.png")
    plt.savefig(file_path)
    print(f"Saved accessibility vs issues analysis to {file_path}")
    
    return score_stats, pass_rates

def analyze_domains(df):
    """Analyze performance by domain"""
    print("\n===== DOMAIN ANALYSIS =====")
    
    # Count sites per domain (some domains may have multiple pages)
    domain_counts = df['Domain'].value_counts()
    print(f"\nTotal unique domains: {len(domain_counts)}")
    print(f"Domains with multiple pages: {sum(domain_counts > 1)}")
    
    # Top and bottom domains by accessibility
    domain_scores = df.groupby('Domain')['Accessibility_Score'].mean().sort_values()
    
    print("\nTop 5 domains by accessibility:")
    for domain, score in domain_scores.tail(5)[::-1].items():
        print(f"- {domain}: {score:.1f}")
    
    print("\nBottom 5 domains by accessibility:")
    for domain, score in domain_scores.head(5).items():
        print(f"- {domain}: {score:.1f}")
    
    # Domain distribution plot removed as it doesn't provide meaningful insights
    
    return domain_scores

def analyze_accessibility_issues(df):
    """Analyze the types and distribution of accessibility issues"""
    print("\n===== ACCESSIBILITY ISSUES ANALYSIS =====")
    
    # Extract accessibility issues if the column exists
    if 'Accessibility_Issues_Details' in df.columns:
        # Count the most common issues
        all_issues = []
        
        for issues in df['Accessibility_Issues_Details'].dropna():
            if pd.isna(issues) or issues == '':
                continue
                
            # Split the issues by semicolon
            for issue in str(issues).split(';'):
                # Extract the issue title (before the Score part)
                title_match = re.search(r'^(.*?)\s*\(Score:', issue.strip())
                if title_match:
                    title = title_match.group(1).strip()
                    all_issues.append(title)
        
        # Count issue frequencies
        issue_counts = pd.Series(all_issues).value_counts()
        
        print(f"\nTotal unique issue types: {len(issue_counts)}")
        print("\nTop 10 most common issues:")
        for issue, count in issue_counts.head(10).items():
            print(f"- {issue}: {count} instances")
        
        # Visualize top issues
        plt.figure(figsize=(14, 8))
        top_issues = issue_counts.head(10)
        sns.barplot(x=top_issues.values, y=top_issues.index)
        plt.title("Top 10 Most Common Accessibility Issues")
        plt.xlabel("Count")
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, "common_issues.png")
        plt.savefig(file_path)
        print(f"Saved common issues analysis to {file_path}")
    
    # Analyze specific A11y issue columns
    a11y_cols = [col for col in df.columns if 'A11y_Issue' in col]
    if a11y_cols:
        print(f"\nAnalyzing {len(a11y_cols)} specific accessibility issue columns")
        
        # Mean values of each issue type
        issue_means = df[a11y_cols].mean().sort_values(ascending=False)
        
        print("\nAverage values for each issue type:")
        for issue, mean in issue_means.head(5).items():
            # Clean up the issue name for display
            issue_name = issue.replace('A11y_Issue_', '').replace('_', ' ')
            print(f"- {issue_name}: {mean:.2f}")
        
        # Visualize issue values
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df[a11y_cols])
        plt.title("Distribution of Accessibility Issue Values")
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, "issue_distributions.png")
        plt.savefig(file_path)
        print(f"Saved issue distributions to {file_path}")
    
    # Analyze pass/fail breakdown
    plt.figure(figsize=(8, 8))
    pass_counts = df['AccessibilityResult'].value_counts()
    plt.pie(pass_counts, labels=pass_counts.index, autopct='%1.1f%%', startangle=90,
           colors=['#ff9999','#66b3ff'])
    plt.title("Accessibility Pass/Fail Distribution")
    plt.axis('equal')
    
    file_path = os.path.join(output_dir, "pass_fail_distribution.png")
    plt.savefig(file_path)
    print(f"Saved pass/fail distribution to {file_path}")
    
    return issue_counts if 'issue_counts' in locals() else None

def analyze_keyboard_focus_accessibility(df):
    """Analyze keyboard focus accessibility issues specifically"""
    print("\n===== KEYBOARD FOCUS ACCESSIBILITY ANALYSIS =====")
    
    # Find keyboard and focus related columns
    keyboard_focus_cols = [col for col in df.columns if 'keyboard' in str(col).lower() or 'focus' in str(col).lower()]
    
    if not keyboard_focus_cols:
        print("No keyboard focus related columns found in the dataset.")
        return None
    
    print(f"\nFound {len(keyboard_focus_cols)} keyboard/focus related columns:")
    for idx, col in enumerate(keyboard_focus_cols):
        print(f"{idx+1}. {col}")
    
    # Preprocess the columns to handle missing and non-standard values
    df_processed = df.copy()
    
    for col in keyboard_focus_cols:
        # Convert NaN to 'Not Tested'
        df_processed[col] = df_processed[col].fillna('Not Tested')
        
        # Convert 0.0 values to 'Fail'
        if df_processed[col].astype(str).isin(['0.0', '0']).any():
            df_processed[col] = df_processed[col].astype(str).replace({'0.0': 'Fail', '0': 'Fail'})
        
        # Convert 1.0 values to 'Pass'
        if df_processed[col].astype(str).isin(['1.0', '1']).any():
            df_processed[col] = df_processed[col].astype(str).replace({'1.0': 'Pass', '1': 'Pass'})
    
    # Analyze each keyboard focus column
    summary_data = []
    
    for col in keyboard_focus_cols:
        print(f"\n{col}:")
        # Get value counts
        value_counts = df_processed[col].value_counts(dropna=False)
        print(value_counts)
        
        # Calculate statistics for tested sites (excluding 'Not Tested')
        tested_df = df_processed[df_processed[col] != 'Not Tested']
        tested_count = len(tested_df)
        
        if tested_count > 0:
            pass_count = (tested_df[col] == 'Pass').sum()
            fail_count = (tested_df[col] == 'Fail').sum()
            
            # Calculate pass rate among tested sites
            pass_rate = pass_count / tested_count * 100 if tested_count > 0 else 0
            print(f"Tested sites: {tested_count} out of {len(df)} ({tested_count/len(df)*100:.1f}%)")
            print(f"Pass rate: {pass_rate:.1f}% ({pass_count} out of {tested_count} tested sites)")
            
            # Add to summary data
            summary_data.append({
                'Criterion': col,
                'Pass Count': pass_count,
                'Fail Count': fail_count,
                'Not Tested Count': len(df) - tested_count,
                'Total Tested': tested_count,
                'Pass Rate (%)': pass_rate,
                'Coverage (%)': tested_count / len(df) * 100
            })
    
    # Create summary dataframe if we have data
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Check if we have any tested sites
        has_tested_data = (summary_df['Total Tested'].sum() > 0)
        
        if has_tested_data:
            # Calculate overall keyboard focus score
            if len(summary_df) > 0:
                avg_pass_rate = summary_df['Pass Rate (%)'].mean()
                print(f"\nOverall keyboard focus accessibility score: {avg_pass_rate:.1f}%")
                
                # Calculate percentage of sites with at least one keyboard focus issue
                sites_with_any_issue = df_processed[df_processed[keyboard_focus_cols].eq('Fail').any(axis=1)]
                issue_count = len(sites_with_any_issue)
                
                if issue_count > 0:
                    print(f"Sites with at least one keyboard focus issue: {issue_count} ({issue_count/len(df)*100:.1f}%)")
                
                # Calculate testing coverage
                avg_coverage = summary_df['Coverage (%)'].mean()
                print(f"Average testing coverage for keyboard focus criteria: {avg_coverage:.1f}%")
        else:
            print("\nNo keyboard focus accessibility data available for analysis. Most sites were not tested for these criteria.")
    
    return summary_data

def create_accessibility_dashboard(df):
    """Create a comprehensive dashboard of accessibility metrics"""
    print("\n===== CREATING ACCESSIBILITY DASHBOARD =====")
    
    # Create a 3x2 subplot layout
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 1. Accessibility score distribution
    sns.histplot(df['Accessibility_Score'].dropna(), kde=True, bins=20, ax=axes[0,0])
    axes[0,0].axvline(x=90, color='red', linestyle='--', label='Pass threshold (90)')
    axes[0,0].set_title("Accessibility Score Distribution", fontsize=16)
    axes[0,0].set_xlabel("Score")
    axes[0,0].set_ylabel("Frequency")
    axes[0,0].legend()
    
    # 2. Pass/Fail pie chart
    pass_counts = df['AccessibilityResult'].value_counts()
    axes[0,1].pie(pass_counts, labels=pass_counts.index, autopct='%1.1f%%', startangle=90,
           colors=['#ff9999','#66b3ff'])
    axes[0,1].set_title("Accessibility Pass/Fail Distribution", fontsize=16)
    axes[0,1].axis('equal')
    
    # 3. Accessibility vs Performance scatter
    valid_data = df[['Performance_Score', 'Accessibility_Score']].dropna()
    if len(valid_data) > 5:
        sns.scatterplot(x='Performance_Score', y='Accessibility_Score', data=valid_data, ax=axes[1,0])
        axes[1,0].set_title("Accessibility vs Performance Scores", fontsize=16)
        axes[1,0].set_xlabel("Performance Score")
        axes[1,0].set_ylabel("Accessibility Score")
        
        # Add reference lines for 90% thresholds
        axes[1,0].axhline(y=90, color='red', linestyle='--')
        axes[1,0].axvline(x=90, color='red', linestyle='--')
    else:
        axes[1,0].text(0.5, 0.5, "Insufficient data for scatter plot", 
                       horizontalalignment='center', fontsize=14)
    
    # 4. Issues count distribution
    sns.histplot(df['Accessibility_Issues_Count'].dropna(), kde=True, bins=15, ax=axes[1,1])
    axes[1,1].set_title("Number of Accessibility Issues Distribution", fontsize=16)
    axes[1,1].set_xlabel("Number of Issues")
    axes[1,1].set_ylabel("Frequency")
    
    # 5. Correlation heatmap for all score columns
    score_cols = ['Performance_Score', 'Accessibility_Score', 'Best_Practices_Score', 'SEO_Score', 'Accessibility_Issues_Count']
    score_cols = [col for col in score_cols if col in df.columns]
    
    if len(score_cols) > 1:  # Need at least 2 columns for correlation
        # Calculate correlation with pairwise deletion of missing values
        corr_data = df[score_cols].dropna()
        if len(corr_data) > 5:  # Need sufficient data points
            corr = corr_data.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[2,0])
            axes[2,0].set_title("Score Correlations", fontsize=16)
        else:
            axes[2,0].text(0.5, 0.5, "Insufficient data for correlation heatmap", 
                           horizontalalignment='center', fontsize=14)
    
    # 6. Top accessibility issues
    a11y_issue_cols = [col for col in df.columns if 'A11y_Issue' in col]
    
    if a11y_issue_cols:
        # Calculate average value for each issue
        issue_means = df[a11y_issue_cols].mean().sort_values(ascending=False)
        
        # Take top 10 or all if less than 10
        top_n = min(10, len(issue_means))
        top_issues = issue_means.head(top_n)
        
        # Clean names for display
        clean_names = [col.replace('A11y_Issue_', '').replace('_', ' ') for col in top_issues.index]
        
        sns.barplot(x=top_issues.values, y=clean_names, ax=axes[2,1])
        axes[2,1].set_title("Top Accessibility Issues (Average Values)", fontsize=16)
        axes[2,1].set_xlabel("Average Value")
    
    # Save the dashboard
    file_path = os.path.join(output_dir, "accessibility_dashboard.png")
    plt.savefig(file_path, bbox_inches='tight')
    print(f"Saved comprehensive dashboard to {file_path}")

def generate_insights(df, score_stats, domain_scores, issue_counts=None):
    """Generate meaningful insights based on the analysis"""
    print("\n===== KEY INSIGHTS =====")
    
    # Calculate key metrics
    pass_rate = (df['Accessibility_Score'] >= 90).mean() * 100
    avg_score = score_stats.loc['mean', 'Accessibility_Score']
    avg_issues = df['Accessibility_Issues_Count'].mean()
    
    insights = [
        f"1. Only {pass_rate:.1f}% of analyzed healthcare websites pass accessibility standards (score ≥90).",
        f"2. The average accessibility score is {avg_score:.1f}/100, indicating significant room for improvement.",
        f"3. Websites have an average of {avg_issues:.1f} accessibility issues that need to be addressed."
    ]
    
    # Domain insights
    if domain_scores is not None and len(domain_scores) > 1:
        score_range = domain_scores.max() - domain_scores.min()
        insights.append(f"4. There is a {score_range:.1f}-point difference between the best and worst performing domains.")
    
    # Performance correlation insights
    valid_data = df[['Performance_Score', 'Accessibility_Score']].dropna()
    if len(valid_data) > 5:
        perf_acc_corr = valid_data.corr().iloc[0,1]
        if abs(perf_acc_corr) > 0.3:  # Meaningful correlation
            direction = "positive" if perf_acc_corr > 0 else "negative"
            insights.append(f"5. There is a {direction} correlation ({perf_acc_corr:.2f}) between performance and accessibility scores.")
    
    # Issue-specific insights
    if 'Accessibility_Issues_Details' in df.columns:
        # Sites with no issues
        no_issues = (df['Accessibility_Issues_Count'] == 0).sum()
        insights.append(f"6. {no_issues} sites ({no_issues/len(df)*100:.1f}%) have no detected accessibility issues.")
    
    # Most common issues insight
    if issue_counts is not None and len(issue_counts) > 0:
        top_issue = issue_counts.index[0]
        top_issue_count = issue_counts.iloc[0]
        insights.append(f"7. The most common accessibility issue is '{top_issue}', found on {top_issue_count} sites.")
    
    # Print insights
    print("\nKey Insights:")
    for insight in insights:
        print(insight)
    
    # Save insights to text file
    insights_text = "HEALTHCARE WEBSITE ACCESSIBILITY ANALYSIS\n"
    insights_text += "=" * 50 + "\n\n"
    insights_text += "KEY INSIGHTS:\n"
    for insight in insights:
        insights_text += insight + "\n"
    
    # Add recommendations
    insights_text += "\n\nRECOMMENDATIONS:\n"
    recommendations = [
        "1. Prioritize fixing the most common accessibility issues identified in this analysis.",
        "2. Implement regular accessibility audits as part of the development process.",
        "3. Focus on meeting WCAG (Web Content Accessibility Guidelines) 2.1 AA standards.",
        "4. Provide accessibility training for web development and content teams.",
        "5. Test with real users who have disabilities to identify practical accessibility barriers."
    ]
    
    for rec in recommendations:
        insights_text += rec + "\n"
    
    # Add analysis summary
    insights_text += "\n\nANALYSIS SUMMARY:\n"
    insights_text += f"- Total websites analyzed: {len(df)}\n"
    insights_text += f"- Websites passing accessibility standards: {(df['Accessibility_Score'] >= 90).sum()} ({pass_rate:.1f}%)\n"
    insights_text += f"- Average accessibility score: {avg_score:.1f}/100\n"
    insights_text += f"- Average number of issues per site: {avg_issues:.1f}\n"
    
    # Write to file
    insights_file = os.path.join(output_dir, "accessibility_insights_summary.txt")
    with open(insights_file, "w") as f:
        f.write(insights_text)
    
    print(f"\nSaved detailed insights and recommendations to {insights_file}")
    
    return insights, insights_text

def create_comprehensive_insights_summary(df, score_stats, domain_scores, issue_counts, pass_rates):
    """Create a comprehensive insights summary document with references to visualizations"""
    print("\n===== CREATING COMPREHENSIVE INSIGHTS SUMMARY =====")
    
    # Create the summary document
    summary = f"""# HEALTHCARE WEBSITE ACCESSIBILITY ANALYSIS INSIGHTS
=======================================================

## EXECUTIVE SUMMARY

This analysis evaluated the accessibility of {len(df)} healthcare websites in Switzerland using Lighthouse metrics. 
Overall, Swiss healthcare websites show moderate accessibility compliance, with significant room for improvement.

## KEY FINDINGS

1. Only {(df['Accessibility_Score'] >= 90).mean() * 100:.1f}% of analyzed healthcare websites pass accessibility standards (score ≥90)
2. The average accessibility score is {score_stats.loc['mean', 'Accessibility_Score']:.1f}/100
3. Websites have an average of {df['Accessibility_Issues_Count'].mean():.1f} accessibility issues per site
4. There is a {domain_scores.max() - domain_scores.min():.1f}-point difference between the best and worst performing domains
5. Only {(df['Accessibility_Issues_Count'] == 0).sum()/len(df)*100:.1f}% of sites have no detected accessibility issues
6. Performance metrics are also problematic, with only {pass_rates['Performance_Score']:.1f}% of sites passing performance standards
7. Best Practices ({pass_rates['Best_Practices_Score']:.1f}% pass) and SEO ({pass_rates['SEO_Score']:.1f}% pass) have better compliance rates

## MOST COMMON ACCESSIBILITY ISSUES
"""

    # Add most common issues if available
    if issue_counts is not None and len(issue_counts) > 0:
        for i, (issue, count) in enumerate(issue_counts.head(10).items(), 1):
            summary += f"\n{i}. {issue} ({count} instances)"
    
    # Add domain analysis
    summary += "\n\n## DOMAIN ANALYSIS\n\nThe analysis revealed significant variations in accessibility scores across domains, with the lowest-scoring domains being:"
    
    for domain, score in domain_scores.head(5).items():
        if not pd.isna(score):
            summary += f"\n- {domain}: {score:.1f}"
    
    # Add recommendations section
    summary += """

## RECOMMENDATIONS

1. **Prioritize Critical Issues**: Focus on fixing the most common accessibility issues identified in this analysis, particularly those that affect core functionality (Time to Interactive, First Contentful Paint).

2. **Implement Regular Audits**: Establish regular accessibility testing as part of the development and maintenance process. Automated tools combined with manual testing provide the most comprehensive approach.

3. **Adopt WCAG Standards**: Focus on meeting Web Content Accessibility Guidelines (WCAG) 2.1 AA standards, which are recognized internationally as the benchmark for web accessibility.

4. **Training**: Provide accessibility training for web development, design, and content teams to ensure awareness of accessibility requirements during the creation process rather than retrofitting later.

5. **User Testing**: Conduct testing with real users who have disabilities to identify practical accessibility barriers that automated tools might miss.

6. **Performance Optimization**: Improve Time to Interactive and First Contentful Paint metrics by optimizing JavaScript loading, reducing server response times, and implementing efficient caching policies.

7. **Image Optimization**: Ensure all images have explicit width and height attributes to reduce layout shifts and improve user experience.

8. **Standardize Implementation**: Develop accessibility standards and patterns for common website elements to ensure consistency across the website.

## IMPACT OF ACCESSIBILITY IMPROVEMENTS

Improving website accessibility in the healthcare sector has significant benefits:

1. **Broader Reach**: Approximately 1 in 5 people have some form of disability. Accessible websites ensure healthcare information reaches all potential patients.

2. **Legal Compliance**: Many jurisdictions require websites to meet certain accessibility standards, reducing legal risk.

3. **Better User Experience**: Accessibility improvements benefit all users, not just those with disabilities.

4. **Improved SEO**: Many accessibility practices also improve search engine rankings.

5. **Ethical Responsibility**: Healthcare providers have a particular obligation to ensure equal access to health information.

## NEXT STEPS

1. Share these findings with stakeholders and website owners
2. Develop a prioritized remediation plan focusing on the most critical issues
3. Establish accessibility guidelines for future development
4. Implement regular monitoring to track improvements
5. Conduct more in-depth testing of the lowest-performing domains

## METHODOLOGY

This analysis was conducted using Google Lighthouse accessibility audits on healthcare websites in Switzerland. 
Data was collected and processed through automated scripts, with missing or invalid data appropriately handled. 
The analysis focused on four key metrics: Performance, Accessibility, Best Practices, and SEO.

## VISUALIZATIONS

The following visualizations were generated as part of this analysis:

1. **Accessibility Dashboard**: Comprehensive overview of key metrics
2. **Score Distributions**: Distribution of Performance, Accessibility, Best Practices, and SEO scores
3. **Common Issues**: The most frequently encountered accessibility issues
4. **Pass/Fail Distribution**: Proportion of websites passing accessibility standards
5. **Accessibility vs. Issues**: Relationship between accessibility scores and number of issues

=======================================================
Report generated on: {datetime.now().strftime('%B %d, %Y')}
Analysis based on Lighthouse audits of {len(df)} Swiss healthcare websites
"""

    # Save the comprehensive summary
    summary_file = "insights_summary.txt"
    with open(summary_file, "w") as f:
        f.write(summary)
    
    # Also save it to the output directory
    output_summary_file = os.path.join(output_dir, "insights_summary.txt")
    with open(output_summary_file, "w") as f:
        f.write(summary)
    
    print(f"Saved comprehensive insights summary to {summary_file} and {output_summary_file}")
    
    return summary

def main():
    """Main function to execute accessibility analysis"""
    # Load data
    df = load_data()
    print(f"Data loaded: {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Filter out records with missing scores - these are likely PDFs or non-HTML content
    original_count = len(df)
    df_filtered = df.dropna(subset=['Accessibility_Score', 'Performance_Score', 'Best_Practices_Score', 'SEO_Score'], how='any')
    excluded_count = original_count - len(df_filtered)
    
    if excluded_count > 0:
        print(f"\nExcluded {excluded_count} websites with missing scores (likely PDFs or non-HTML content).")
        print(f"Continuing analysis with {len(df_filtered)} websites.")
        
        # Show domains of excluded websites
        excluded_domains = df[df['Accessibility_Score'].isna()]['Domain'].tolist()
        print(f"Excluded domains: {', '.join(excluded_domains)}")
    
    # Run analyses on filtered data
    score_stats, pass_rates = analyze_scores(df_filtered)
    domain_scores = analyze_domains(df_filtered)
    issue_counts = analyze_accessibility_issues(df_filtered)
    keyboard_focus_results = analyze_keyboard_focus_accessibility(df_filtered)
    
    # Create dashboard visualization
    create_accessibility_dashboard(df_filtered)
    
    # Generate insights
    insights, insights_text = generate_insights(df_filtered, score_stats, domain_scores, issue_counts)
    print("\n".join(insights))
    
    # Create comprehensive insights file
    create_comprehensive_insights_summary(df_filtered, score_stats, domain_scores, issue_counts, pass_rates)

if __name__ == "__main__":
    main() 