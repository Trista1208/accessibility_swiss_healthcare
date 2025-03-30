import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import matplotlib.gridspec as gridspec

# Set plot style and configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Create output directory for visualization
output_dir = "final_accessibility_analysis"
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
    
    # Score correlation analysis
    plt.figure(figsize=(10, 8))
    score_corr = df[score_cols].corr()
    mask = np.triu(np.ones_like(score_corr, dtype=bool))
    
    sns.heatmap(score_corr, annot=True, mask=mask, vmin=-1, vmax=1, 
                cmap='coolwarm', fmt='.2f', linewidths=1)
    
    plt.title("Score Correlations")
    plt.tight_layout()
    
    file_path = os.path.join(output_dir, "score_correlations.png")
    plt.savefig(file_path)
    print(f"Saved score correlations to {file_path}")
    
    return score_stats, pass_rates

def analyze_domains(df):
    """Analyze performance by domain"""
    print("\n===== DOMAIN ANALYSIS =====")
    
    # Count sites per domain (some domains may have multiple pages)
    domain_counts = df['Domain'].value_counts()
    print(f"\nTotal unique domains: {len(domain_counts)}")
    
    # Top and bottom domains by accessibility
    domain_scores = df.groupby('Domain')['Accessibility_Score'].mean().sort_values()
    
    print("\nTop 5 domains by accessibility:")
    for domain, score in domain_scores.tail(5)[::-1].items():
        print(f"- {domain}: {score:.1f}")
    
    print("\nBottom 5 domains by accessibility:")
    for domain, score in domain_scores.head(5).items():
        print(f"- {domain}: {score:.1f}")
    
    return domain_scores

def analyze_accessibility_issues(df):
    """Analyze accessibility issues to identify patterns and common problems"""
    print("\n===== ACCESSIBILITY ISSUES ANALYSIS =====")
    
    # Find issue columns
    a11y_cols = []
    
    for col in df.columns:
        if (col.startswith('A11y_Issue_') or 
            col.startswith('Accessibility_Issue_') or 
            'Interactive' in col):
            a11y_cols.append(col)
    
    # If no standard issue columns, check for accessibility details columns
    if not a11y_cols:
        for col in df.columns:
            if 'details' in col.lower() and 'accessibility' in col.lower():
                a11y_cols.append(col)
    
    if not a11y_cols:
        print("No accessibility issue columns found in the dataset.")
        return None
    
    # Count the total number of unique issues
    issue_count = len(a11y_cols)
    print(f"\nTotal unique issue types: {issue_count}")
    
    if issue_count == 0:
        print("No valid issue metrics found for analysis.")
        return None
    
    # For categorical issue columns, count occurrence
    issue_counts = {}
    
    for col in a11y_cols:
        # Count non-NaN values as issue instances
        count = df[col].notna().sum()
        if count > 0:
            # Clean up column name for display
            display_name = col.replace('A11y_Issue_', '').replace('Accessibility_', '').replace('_', ' ')
            issue_counts[display_name] = count
    
    # Sort issues by frequency
    sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Display top 10 issues
    if sorted_issues:
        top_n = min(10, len(sorted_issues))
        print(f"\nTop {top_n} most common issues:")
        for issue, count in sorted_issues[:top_n]:
            print(f"- {issue}: {count} instances")
    
    # Create visualization of common issues
    if sorted_issues:
        plt.figure(figsize=(14, 10))
        
        # Get data for plotting
        issues = [issue for issue, _ in sorted_issues[:15]]  # Top 15 for visualization
        counts = [count for _, count in sorted_issues[:15]]
        
        # Create horizontal bar chart
        bars = plt.barh([issue[:50] + '...' if len(issue) > 50 else issue for issue in issues][::-1], 
                        counts[::-1], color='coral')
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 1
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, str(int(width)),
                    va='center')
        
        plt.xlabel('Number of Occurrences')
        plt.title('Most Common Accessibility Issues')
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, "common_issues.png")
        plt.savefig(file_path)
        print(f"Saved common issues analysis to {file_path}")
    
    # Create pass/fail distribution visualization
    if 'AccessibilityResult' in df.columns:
        plt.figure(figsize=(8, 8))
        pass_counts = df['AccessibilityResult'].value_counts()
        
        plt.pie(pass_counts, labels=pass_counts.index, autopct='%1.1f%%', startangle=90,
                colors=['#ff9999','#66b3ff'])
        plt.title("Accessibility Pass/Fail Distribution")
        plt.axis('equal')
        
        file_path = os.path.join(output_dir, "pass_fail_distribution.png")
        plt.savefig(file_path)
        print(f"Saved pass/fail distribution to {file_path}")
    
    return issue_counts

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
    
    # Preprocess values in these columns
    df_focus = df.copy()
    for col in keyboard_focus_cols:
        # Convert NaN to 'Not Tested'
        df_focus[col] = df_focus[col].fillna('Not Tested')
        
        # Convert 0.0 and 1.0 to 'Fail' and 'Pass'
        df_focus[col] = df_focus[col].apply(
            lambda x: 'Fail' if x == 0.0 else ('Pass' if x == 1.0 else x)
        )
    
    # Count sites with keyboard focus data
    has_keyboard_data = df_focus[keyboard_focus_cols].apply(
        lambda x: x != 'Not Tested').any(axis=1)
    sites_with_data = has_keyboard_data.sum()
    percent_with_data = sites_with_data / len(df) * 100
    
    print(f"\nSites with keyboard focus accessibility data: {sites_with_data} out of {len(df)} ({percent_with_data:.1f}%)")
    
    if sites_with_data == 0:
        print("No keyboard focus accessibility data available for analysis.")
        return None
    
    # Analyze each keyboard focus column
    pass_rates = {}
    for col in keyboard_focus_cols:
        # Count values excluding 'Not Tested'
        value_counts = df_focus[df_focus[col] != 'Not Tested'][col].value_counts()
        
        # Calculate statistics for sites that were tested
        tested_count = len(df_focus[df_focus[col] != 'Not Tested'])
        
        if tested_count > 0:
            print(f"\nAnalysis for {col}:")
            print(f"- Tested sites: {tested_count} ({tested_count/len(df)*100:.1f}%)")
            print("- Value counts:")
            for val, count in value_counts.items():
                print(f"  * {val}: {count} ({count/tested_count*100:.1f}% of tested sites)")
            
            # Calculate pass rate (looking for 'Pass' values)
            pass_count = value_counts.get('Pass', 0)
            pass_rate = pass_count / tested_count * 100 if tested_count > 0 else 0
            pass_rates[col] = pass_rate
            print(f"- Pass rate: {pass_rate:.1f}%")
    
    # Create visualization for keyboard focus accessibility
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Pass rates for each test
    cols_to_plot = [col for col in keyboard_focus_cols if pass_rates.get(col, -1) >= 0]
    
    if cols_to_plot:
        # Create readable names for the plot
        col_names = [col.replace('Accessibility_Keyboard_Focus_', '').replace('_', ' ') for col in cols_to_plot]
        pass_rate_values = [pass_rates[col] for col in cols_to_plot]
        
        # Create horizontal bar chart
        bars = plt.barh(col_names, pass_rate_values, color='skyblue')
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va='center')
        
        plt.xlabel('Pass Rate (%)')
        plt.title('Keyboard Focus Accessibility Pass Rates')
        plt.xlim(0, 105)  # Allow space for labels
    else:
        plt.text(0.5, 0.5, 'No pass rate data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
    
    # Save visualization
    file_path = os.path.join(output_dir, "keyboard_focus_accessibility.png")
    plt.savefig(file_path)
    print(f"Saved keyboard focus accessibility analysis to {file_path}")
    
    # Overall statistics
    total_tests = sum(len(df_focus[df_focus[col] != 'Not Tested']) for col in keyboard_focus_cols)
    total_passes = sum(value_counts.get('Pass', 0) for col in keyboard_focus_cols for value_counts in [df_focus[df_focus[col] != 'Not Tested'][col].value_counts()])
    
    overall_pass_rate = total_passes / total_tests * 100 if total_tests > 0 else 0
    
    print(f"\nOverall Keyboard Focus Accessibility:")
    print(f"- Total tests performed: {total_tests}")
    print(f"- Total passes: {total_passes}")
    print(f"- Overall pass rate: {overall_pass_rate:.1f}%")
    
    # Sites with at least one keyboard focus issue
    sites_with_issues = df_focus[has_keyboard_data & df_focus[keyboard_focus_cols].apply(
        lambda x: (x == 'Fail').any(), axis=1)].shape[0]
    percent_with_issues = sites_with_issues / sites_with_data * 100 if sites_with_data > 0 else 0
    
    print(f"- Sites with at least one keyboard focus issue: {sites_with_issues} ({percent_with_issues:.1f}% of tested sites)")
    
    return {
        'tested_sites': sites_with_data,
        'percent_tested': percent_with_data,
        'has_keyboard_issues': sites_with_issues,
        'overall_pass_rate': overall_pass_rate
    }

def create_accessibility_dashboard(df):
    """Create a simplified dashboard with key accessibility insights"""
    print("\n===== CREATING ACCESSIBILITY DASHBOARD =====")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # 1. Score Distribution Plot
    ax_scores = fig.add_subplot(gs[0, 0])
    
    # Get score columns
    score_cols = [col for col in df.columns if 'Score' in col and col != 'AccessibilityResult']
    for col in score_cols:
        sns.kdeplot(df[col].dropna(), ax=ax_scores, label=col.replace('_Score', ''))
    
    ax_scores.axvline(x=90, color='red', linestyle='--', label='Pass threshold (90)')
    ax_scores.set_title('Score Distributions')
    ax_scores.set_xlabel('Score')
    ax_scores.set_ylabel('Density')
    ax_scores.legend()
    
    # 2. Pass/Fail Pie Chart
    ax_pass_fail = fig.add_subplot(gs[0, 1])
    
    if 'AccessibilityResult' in df.columns:
        pass_counts = df['AccessibilityResult'].value_counts()
        ax_pass_fail.pie(pass_counts, labels=pass_counts.index, autopct='%1.1f%%', startangle=90,
                colors=['#ff9999','#66b3ff'])
        ax_pass_fail.set_title('Accessibility Pass/Fail Distribution')
        ax_pass_fail.axis('equal')
    
    # 3. Top Issues Bar Chart
    ax_issues = fig.add_subplot(gs[1, 0])
    
    # Find accessibility issue columns
    a11y_issue_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        # Look for issue-related columns with various naming patterns
        if (col.startswith('A11y_Issue_') or 
            'issue' in col_lower or 
            'accessibility' in col_lower and ('details' in col_lower or 'error' in col_lower) or
            'interactive' in col_lower and 'focusable' in col_lower):
            a11y_issue_cols.append(col)
    
    # Generate issue counts based on non-null values
    if a11y_issue_cols:
        issue_counts = {}
        
        for col in a11y_issue_cols:
            count = df[col].notna().sum()
            if count > 0:
                # Clean up column name for display
                display_name = col.replace('A11y_Issue_', '').replace('Accessibility_', '').replace('_', ' ')
                issue_counts[display_name] = count
                
        # Sort and plot top issues
        if issue_counts:
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            top_issues = sorted_issues[:8]  # Top 8 issues
            
            issues = [issue[:30] + '...' if len(issue) > 30 else issue for issue, _ in top_issues]
            counts = [count for _, count in top_issues]
            
            bars = ax_issues.barh(issues[::-1], counts[::-1], color='coral')
            
            # Add count labels
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 1
                ax_issues.text(label_x_pos, bar.get_y() + bar.get_height()/2, str(int(width)),
                       va='center')
            
            ax_issues.set_title('Most Common Accessibility Issues')
            ax_issues.set_xlabel('Number of Sites')
    
    # 4. Keyboard Focus Testing
    ax_keyboard = fig.add_subplot(gs[1, 1])
    
    # Find keyboard focus columns
    keyboard_cols = [col for col in df.columns if 'keyboard' in col.lower() or 'focus' in col.lower()]
    
    if keyboard_cols:
        # Count sites with keyboard data
        df_focus = df.copy()
        for col in keyboard_cols:
            df_focus[col] = df_focus[col].fillna('Not Tested')
        
        has_keyboard_data = df_focus[keyboard_cols].apply(lambda x: x != 'Not Tested').any(axis=1)
        sites_with_data = has_keyboard_data.sum()
        sites_without_data = len(df) - sites_with_data
        
        # Create donut chart showing tested vs untested
        data = [sites_with_data, sites_without_data]
        labels = [f'Tested ({sites_with_data})', f'Not Tested ({sites_without_data})']
        
        # Create a donut chart
        ax_keyboard.pie(data, labels=labels, autopct='%1.1f%%', startangle=90,
               colors=['#66b3ff', '#d9d9d9'])
        # Add a circle in the middle to create a donut
        centre_circle = plt.Circle((0,0), 0.7, fc='white')
        ax_keyboard.add_patch(centre_circle)
        
        ax_keyboard.set_title('Keyboard Focus Accessibility Testing Coverage')
        ax_keyboard.axis('equal')
    
    # 5. Severity Distribution
    ax_severity = fig.add_subplot(gs[2, :])
    
    # Extract severity information from keyboard focus column if available
    keyboard_focus_col = next((col for col in df.columns if 'keyboard' in col.lower() and 'focusable' in col.lower()), None)
    
    if keyboard_focus_col and df[keyboard_focus_col].notna().any():
        # Extract severity from each value
        severity_data = {'High': 0, 'Medium': 0, 'Low': 0, 'Unknown': 0}
        
        for val in df[keyboard_focus_col].dropna():
            val_str = str(val).lower()
            
            if 'severity: high' in val_str:
                severity_data['High'] += 1
            elif 'severity: medium' in val_str:
                severity_data['Medium'] += 1
            elif 'severity: low' in val_str:
                severity_data['Low'] += 1
            else:
                severity_data['Unknown'] += 1
        
        # Only show non-zero counts
        non_zero = {k: v for k, v in severity_data.items() if v > 0}
        
        if non_zero:
            # Define colors for different severity levels
            colors = {'High': '#FF5252', 'Medium': '#FFA726', 'Low': '#66BB6A', 'Unknown': '#E0E0E0'}
            
            # Create horizontal bar chart
            severity_labels = list(non_zero.keys())
            severity_values = list(non_zero.values())
            
            # Sort by severity level (High, Medium, Low, Unknown)
            severity_order = {'High': 0, 'Medium': 1, 'Low': 2, 'Unknown': 3}
            sorted_data = sorted(zip(severity_labels, severity_values), 
                                key=lambda x: severity_order.get(x[0], 4))
            
            labels = [x[0] for x in sorted_data]
            values = [x[1] for x in sorted_data]
            
            bars = ax_severity.barh(labels, values, 
                                   color=[colors.get(label, '#BDBDBD') for label in labels])
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax_severity.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                               str(int(width)), va='center')
            
            ax_severity.set_title('Accessibility Issues by Severity')
            ax_severity.set_xlabel('Number of Issues')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Web Accessibility Dashboard - Swiss Healthcare Websites', fontsize=16, y=0.98)
    
    # Save the dashboard
    file_path = os.path.join(output_dir, "accessibility_dashboard.png")
    plt.savefig(file_path)
    print(f"Saved accessibility dashboard to {file_path}")

def generate_insights(df, score_stats, domain_scores, issue_counts=None):
    """Generate meaningful insights based on the analysis"""
    print("\n===== KEY INSIGHTS =====")
    
    # Calculate key metrics
    pass_rate = (df['Accessibility_Score'] >= 90).mean() * 100
    avg_score = score_stats.loc['mean', 'Accessibility_Score']
    
    # Calculate average issues properly by excluding sites without issues data
    avg_issues = None
    if 'Accessibility_Issues_Count' in df.columns:
        # Only consider sites that have issue count data (not NaN)
        sites_with_issue_data = df['Accessibility_Issues_Count'].notna()
        if sites_with_issue_data.any():
            avg_issues = df.loc[sites_with_issue_data, 'Accessibility_Issues_Count'].mean()
            # Also calculate what percentage of sites have issue data
            percent_with_data = sites_with_issue_data.mean() * 100
    
    insights = [
        f"1. Only {pass_rate:.1f}% of analyzed healthcare websites pass accessibility standards (score ≥90).",
        f"2. The average accessibility score is {avg_score:.1f}/100, indicating significant room for improvement."
    ]
    
    if avg_issues is not None:
        insights.append(f"3. Websites have an average of {avg_issues:.1f} accessibility issues that need to be addressed (based on sites with available issue data).")
    
    # Domain insights
    if domain_scores is not None and len(domain_scores) > 1:
        score_range = domain_scores.max() - domain_scores.min()
        insights.append(f"4. There is a {score_range:.1f}-point difference between the best and worst performing domains.")
    
    # Insight about most common issue
    if issue_counts and len(issue_counts) > 0:
        # Convert to DataFrame for consistent handling
        issue_df = pd.DataFrame({'Issue': list(issue_counts.keys()), 'Count': list(issue_counts.values())})
        issue_df = issue_df.sort_values('Count', ascending=False)
        
        if not issue_df.empty:
            top_issue = issue_df.iloc[0]['Issue']
            top_issue_count = issue_df.iloc[0]['Count']
            insights.append(f"5. The most common accessibility issue is '{top_issue}', found on {top_issue_count} sites.")
    
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
    
    # Write to file
    insights_file = os.path.join(output_dir, "accessibility_insights_summary.txt")
    with open(insights_file, "w") as f:
        f.write(insights_text)
    
    print(f"\nSaved insights and recommendations to {insights_file}")
    
    return insights, insights_text

def create_insights_summary(df, score_stats, domain_scores, issue_counts):
    """Create a summary of insights and recommendations"""
    timestamp = datetime.now().strftime("%B %d, %Y")
    
    # Calculate average issues properly by excluding sites without issues data
    avg_issues = None
    if 'Accessibility_Issues_Count' in df.columns:
        # Only consider sites that have issue count data (not NaN)
        sites_with_issue_data = df['Accessibility_Issues_Count'].notna()
        if sites_with_issue_data.any():
            avg_issues = df.loc[sites_with_issue_data, 'Accessibility_Issues_Count'].mean()
    
    # Create the content
    summary = f"""# SWISS HEALTHCARE WEBSITE ACCESSIBILITY INSIGHTS

## OVERVIEW

This document presents an analysis of web accessibility for Swiss healthcare websites based on Google Lighthouse audits. The analysis included {len(df)} websites.

## KEY ACCESSIBILITY FINDINGS

1. **Accessibility Compliance**: Only {(df['Accessibility_Score'] >= 90).mean() * 100:.1f}% of healthcare websites pass accessibility standards (score ≥90/100).

2. **Average Performance**:
   - Average accessibility score: {score_stats.loc['mean', 'Accessibility_Score']:.1f}/100"""

    if avg_issues is not None:
        summary += f"\n   - Average number of accessibility issues: {avg_issues:.1f} per site (for sites with available issue data)"
    else:
        summary += "\n   - Average number of accessibility issues: No data available"

    summary += "\n\n3. **Top Accessibility Issues**:"

    # Add top issues if available
    if issue_counts and len(issue_counts) > 0:
        issue_df = pd.DataFrame({'Issue': list(issue_counts.keys()), 'Count': list(issue_counts.values())})
        issue_df = issue_df.sort_values('Count', ascending=False)
        top_issues = min(5, len(issue_df))
        for i in range(top_issues):
            issue = issue_df.iloc[i]['Issue']
            count = issue_df.iloc[i]['Count']
            summary += f"\n   - {issue} ({count} sites)"
    
    # Add keyboard focus section
    keyboard_cols = [col for col in df.columns if 'keyboard' in str(col).lower() or 'focus' in str(col).lower()]
    if keyboard_cols:
        has_keyboard_data = df[keyboard_cols].notna().any(axis=1)
        sites_with_data = has_keyboard_data.sum()
        testing_rate = sites_with_data / len(df) * 100
        
        summary += f"""

4. **Keyboard Accessibility**:
   - Only {testing_rate:.1f}% of sites were tested for keyboard focus accessibility"""

    # Add recommendations
    summary += """

## RECOMMENDATIONS

1. **Critical Improvements**:
   - Fix common accessibility issues identified in this analysis
   - Implement efficient keyboard navigation
   - Ensure proper image descriptions and ARIA attributes

2. **Accessibility Standards**:
   - Adopt and adhere to WCAG 2.1 AA standards
   - Implement accessibility testing in the development lifecycle
   - Test with assistive technologies

## CONCLUSION

The accessibility of Swiss healthcare websites requires significant improvement. With less than half meeting basic accessibility standards, there is a clear need for healthcare providers to address these issues.

Report Date: {timestamp}"""

    # Save to file
    summary_file = "insights_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    # Also save a copy to the output directory
    output_file = os.path.join(output_dir, "insights_summary.txt")
    with open(output_file, 'w') as f:
        f.write(summary)
    
    print(f"Saved insights summary to {summary_file} and {output_file}")
    
    return summary

def analyze_severity_distribution(df):
    """Analyze and visualize the severity distribution of accessibility issues"""
    print("\n===== SEVERITY DISTRIBUTION ANALYSIS =====")
    
    # Extract severity from keyboard focus column if available
    keyboard_focus_col = next((col for col in df.columns if 'keyboard' in col.lower() and 'focusable' in col.lower()), None)
    
    if keyboard_focus_col and df[keyboard_focus_col].notna().any():
        print(f"\nExtracting severity information from {keyboard_focus_col}")
        
        # Extract severity from each value
        severity_data = {}
        tested_sites = 0
        
        for val in df[keyboard_focus_col].dropna():
            tested_sites += 1
            val_str = str(val).lower()
            
            if 'severity: high' in val_str:
                severity = 'High'
            elif 'severity: medium' in val_str:
                severity = 'Medium'
            elif 'severity: low' in val_str:
                severity = 'Low'
            else:
                severity = 'Unknown'
            
            severity_data[severity] = severity_data.get(severity, 0) + 1
        
        if severity_data:
            print(f"\nSeverity distribution from {tested_sites} tested sites:")
            for severity, count in sorted(severity_data.items(), 
                                         key=lambda x: {'High': 0, 'Medium': 1, 'Low': 2, 'Unknown': 3}.get(x[0], 4)):
                percent = count / tested_sites * 100
                print(f"- {severity}: {count} issues ({percent:.1f}%)")
            
            # Create a more detailed visualization
            plt.figure(figsize=(14, 10))
            
            # Set up a 2x1 subplot grid
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # 1. Pie chart for severity distribution
            colors = {'High': '#FF5252', 'Medium': '#FFA726', 'Low': '#66BB6A', 'Unknown': '#E0E0E0'}
            sorted_data = sorted(severity_data.items(), 
                              key=lambda x: {'High': 0, 'Medium': 1, 'Low': 2, 'Unknown': 3}.get(x[0], 4))
            labels = [f"{k} ({v})" for k, v in sorted_data]
            sizes = [v for _, v in sorted_data]
            
            # Explode the high severity slice
            explode = [0.1 if k == 'High' else 0.05 if k == 'Medium' else 0 for k, _ in sorted_data]
            
            ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90,
                   colors=[colors.get(k, '#BDBDBD') for k, _ in sorted_data], shadow=True)
            ax1.axis('equal')
            ax1.set_title('Keyboard Focus Issues by Severity', fontsize=14)
            
            # 2. Bar chart showing severity counts
            bars = ax2.bar(
                [k for k, _ in sorted_data],
                [v for _, v in sorted_data],
                color=[colors.get(k, '#BDBDBD') for k, _ in sorted_data]
            )
            
            # Add count labels above bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=12)
            
            ax2.set_title('Severity Count Distribution', fontsize=14)
            ax2.set_xlabel('Severity Level', fontsize=12)
            ax2.set_ylabel('Number of Issues', fontsize=12)
            ax2.grid(axis='y', alpha=0.3)
            
            # Improve overall appearance
            plt.suptitle('Accessibility Issue Severity Analysis', fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save the plot
            file_path = os.path.join(output_dir, "severity_distribution.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Saved detailed severity distribution to {file_path}")
            
            return severity_data
    
    # If we didn't find severity in keyboard focus, check other issue columns
    issue_cols = [col for col in df.columns if 'issue' in col.lower() or 'aria' in col.lower() or 'form' in col.lower()]
    
    for col in issue_cols:
        if df[col].notna().any():
            sample_vals = df[col].dropna().head(5)
            for val in sample_vals:
                if isinstance(val, str) and ('severity' in val.lower() or 'high' in val.lower() or 'medium' in val.lower()):
                    # Found severity info in this column
                    return analyze_embedded_severity(df, col)
    
    # If we didn't find embedded severity, look for explicit severity columns
    potential_severity_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if ('severity' in col_lower or 
            'impact' in col_lower or 
            'priority' in col_lower or
            ('issue' in col_lower and ('type' in col_lower or 'level' in col_lower))):
            potential_severity_cols.append(col)
    
    if potential_severity_cols:
        severity_col = potential_severity_cols[0]  # Use the first one
        print(f"\nAnalyzing severity from column: {severity_col}")
        
        if df[severity_col].dtype == 'object':
            # If it's a string column, count by category
            severity_counts = df[severity_col].value_counts()
            
            # Print counts
            print("\nSeverity Distribution:")
            for severity, count in severity_counts.items():
                percent = count / severity_counts.sum() * 100
                print(f"- {severity}: {count} issues ({percent:.1f}%)")
            
            # Create a visualization of the severity distribution
            plt.figure(figsize=(12, 8))
            
            # Create bar chart
            ax = severity_counts.plot(kind='bar', color='skyblue')
            plt.title('Issue Severity Distribution')
            plt.xlabel('Severity')
            plt.ylabel('Count')
            
            # Add value labels
            for i, v in enumerate(severity_counts):
                ax.text(i, v + 0.1, str(v), ha='center')
            
            plt.tight_layout()
            
            # Save visualization
            file_path = os.path.join(output_dir, "severity_distribution.png")
            plt.savefig(file_path)
            print(f"Saved severity distribution to {file_path}")
            
            return severity_counts.to_dict()
    
    # If we couldn't find severity information, create a proxy from accessibility score
    print("\nNo explicit severity information found. Creating a proxy visualization using Accessibility Score.")
    
    # Create a proxy visualization using accessibility score
    if 'Accessibility_Score' in df.columns:
        # Create bins for accessibility score
        bins = [0, 50, 70, 90, 100]
        labels = ['Critical (0-50)', 'High (51-70)', 'Medium (71-90)', 'Low (91-100)']
        
        df['SeverityProxy'] = pd.cut(df['Accessibility_Score'], bins=bins, labels=labels, right=True)
        severity_counts = df['SeverityProxy'].value_counts().sort_index()
        
        # Print counts
        print("\nAccessibility Score as Severity Proxy:")
        for severity, count in severity_counts.items():
            percent = count / len(df) * 100
            print(f"- {severity}: {count} sites ({percent:.1f}%)")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Create bar chart with custom colors
        colors = ['crimson', 'orange', 'gold', 'lightgreen']
        ax = severity_counts.plot(kind='bar', color=colors[:len(severity_counts)])
        plt.title('Sites by Accessibility Severity Level (Based on Score)')
        plt.xlabel('Severity Level')
        plt.ylabel('Number of Sites')
        
        # Add value labels
        for i, v in enumerate(severity_counts):
            ax.text(i, v + 0.5, str(v), ha='center')
        
        plt.tight_layout()
        
        # Save visualization
        file_path = os.path.join(output_dir, "severity_distribution.png")
        plt.savefig(file_path)
        print(f"Saved proxy severity distribution to {file_path}")
        
        return severity_counts.to_dict()
    
    print("No severity information available and no proxy could be created.")
    return None

def analyze_embedded_severity(df, column):
    """Analyze severity embedded in issue column values"""
    print(f"\nExtracting embedded severity from {column}")
    
    severity_counts = {'High': 0, 'Medium': 0, 'Low': 0, 'Unknown': 0}
    total_issues = 0
    
    for val in df[column].dropna():
        if pd.isna(val):
            continue
        
        val_str = str(val).lower()
        total_issues += 1
        
        if 'severity: high' in val_str or 'high severity' in val_str:
            severity_counts['High'] += 1
        elif 'severity: medium' in val_str or 'medium severity' in val_str:
            severity_counts['Medium'] += 1
        elif 'severity: low' in val_str or 'low severity' in val_str:
            severity_counts['Low'] += 1
        else:
            severity_counts['Unknown'] += 1
    
    if total_issues > 0:
        print(f"\nExtracted severity from {total_issues} issue descriptions")
        print("\nSeverity Distribution:")
        for severity, count in severity_counts.items():
            percent = count / total_issues * 100
            print(f"- {severity}: {count} issues ({percent:.1f}%)")
        
        # Create a visualization of the severity distribution
        plt.figure(figsize=(12, 8))
        
        # Create pie chart
        colors = {'High': 'crimson', 'Medium': 'orange', 'Low': 'lightgreen', 'Unknown': 'lightgray'}
        
        # Filter out zero counts to avoid empty slices
        non_zero_counts = {k: v for k, v in severity_counts.items() if v > 0}
        if non_zero_counts:
            labels = [f"{k} ({v})" for k, v in non_zero_counts.items()]
            sizes = list(non_zero_counts.values())
            
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                    colors=[colors.get(k, 'gray') for k in non_zero_counts.keys()],
                    explode=[0.1 if k == 'High' else (0.05 if k == 'Medium' else 0) for k in non_zero_counts.keys()])
            
            plt.axis('equal')
            plt.title(f'Severity Distribution: {column.replace("_", " ")}')
            
            # Save visualization
            file_path = os.path.join(output_dir, "severity_distribution.png")
            plt.savefig(file_path)
            print(f"Saved severity distribution to {file_path}")
            
            return severity_counts
    
    return None

def main():
    """Main function to execute accessibility analysis"""
    # Load data
    df = load_data()
    
    # Filter out records with missing scores
    df_filtered = df.dropna(subset=['Accessibility_Score'], how='any')
    
    # Run analyses
    score_stats, pass_rates = analyze_scores(df_filtered)
    domain_scores = analyze_domains(df_filtered)
    issue_counts = analyze_accessibility_issues(df_filtered)
    keyboard_focus_results = analyze_keyboard_focus_accessibility(df_filtered)
    
    # Analyze severity distribution
    severity_distribution = analyze_severity_distribution(df_filtered)
    
    # Create dashboard visualization
    create_accessibility_dashboard(df_filtered)
    
    # Generate insights
    insights, insights_text = generate_insights(df_filtered, score_stats, domain_scores, issue_counts)
    
    # Create insights summary
    create_insights_summary(df_filtered, score_stats, domain_scores, issue_counts)

if __name__ == "__main__":
    main() 