import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import matplotlib.gridspec as gridspec

# Set plot style and configuration with red and dark blue color palette
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Set custom color palette with red and dark blue
red_blue_palette = ["#8B0000", "#00008B", "#D21F3C", "#0000CD", "#FF0000", "#0000FF"]
sns.set_palette(red_blue_palette)

# Create output directory for visualization
output_dir = "final_accessibility_analysis"
os.makedirs(output_dir, exist_ok=True)

def load_data(file_path='lighthouse_accessibility_data.xlsx'):
    """Load and prepare the data for analysis"""
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    print(f"Data loaded: {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Display basic information - reduce verbosity
    # print("\nData Overview:")
    # print(f"- Score columns: {[col for col in df.columns if '_Score' in col]}")
    # print(f"- Issue columns: {len([col for col in df.columns if 'A11y_Issue' in col or 'not_have' in col or 'lack' in col])} columns")
    
    # Simplify missing values reporting - only show if significant
    missing = df.isnull().sum()
    if missing.sum() > 0 and missing.sum() / (df.shape[0] * df.shape[1]) > 0.1:
        print("\nSignificant missing values detected")
    
    return df

def analyze_scores(df):
    """Analyze the main accessibility and performance scores"""
    print("\n===== SCORE ANALYSIS =====")
    
    # Basic statistics for scores
    score_cols = ['Performance_Score', 'Accessibility_Score', 'Best_Practices_Score', 'SEO_Score']
    score_stats = df[score_cols].describe()
    
    # Reduce detailed output
    # print("\nScore Statistics:")
    # print(score_stats)
    
    # Calculate pass rates (≥90 is considered passing)
    pass_rates = {col: (df[col] >= 90).mean() * 100 for col in score_cols}
    print("\nPass Rates (≥90):")
    for score, rate in pass_rates.items():
        print(f"- {score}: {rate:.1f}%")
    
    # Visualize score distributions
    plt.figure(figsize=(14, 10))
    
    for i, col in enumerate(score_cols):
        plt.subplot(2, 2, i+1)
        sns.histplot(df[col].dropna(), kde=True, bins=20, color=red_blue_palette[i % len(red_blue_palette)])
        plt.axvline(x=90, color='#D21F3C', linestyle='--', label='Pass threshold (90)')
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
    
    # Use a red-blue colormap
    sns.heatmap(score_corr, annot=True, mask=mask, vmin=-1, vmax=1, 
                cmap='RdBu_r', fmt='.2f', linewidths=1)
    
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
    
    # Top and bottom domains by accessibility - reduced output
    domain_scores = df.groupby('Domain')['Accessibility_Score'].mean().sort_values()
    
    # Limit output to just a summary
    print(f"Domain scores range from {domain_scores.min():.1f} to {domain_scores.max():.1f}")
    
    return domain_scores

def analyze_accessibility_issues(df):
    """Analyze accessibility issues to identify patterns and common problems"""
    print("\n===== ACCESSIBILITY ISSUES ANALYSIS =====")
    
    # Find issue columns - now the column structure has changed
    a11y_cols = []
    
    for col in df.columns:
        # Check for extracted issue columns from the new structure
        if (col.startswith('A11y_Issue_') or 
            'not_have' in col or 
            'lacks' in col or 
            'not_specified' in col or
            'Interactive_controls' in col):
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
            display_name = col.replace('A11y_Issue_', '')
            display_name = display_name.replace('_', ' ')
            issue_counts[display_name] = count
    
    # Sort issues by frequency
    sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Display only top 5 issues instead of 10
    if sorted_issues:
        top_n = min(5, len(sorted_issues))
        print(f"\nTop {top_n} most common issues:")
        for issue, count in sorted_issues[:top_n]:
            print(f"- {issue}: {count} instances")
    
    # Create visualization of common issues
    if sorted_issues:
        plt.figure(figsize=(14, 10))
        
        # Get data for plotting
        issues = [issue for issue, _ in sorted_issues[:10]]  # Top 10 for visualization
        counts = [count for _, count in sorted_issues[:10]]
        
        # Create horizontal bar chart
        bars = plt.barh([issue[:40] + '...' if len(issue) > 40 else issue for issue in issues][::-1], 
                        counts[::-1], color='#8B0000')
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 1
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, str(int(width)),
                    va='center', color='#00008B')
        
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
                colors=['#8B0000','#00008B'], textprops={'color': 'white'})
                
        plt.axis('equal')
        plt.title('Accessibility Pass/Fail Distribution')
        
        file_path = os.path.join(output_dir, "pass_fail_distribution.png")
        plt.savefig(file_path)
        print(f"Saved pass/fail distribution to {file_path}")
    
    # Return the dictionary rather than the sorted list for compatibility with existing functions
    return issue_counts

def analyze_keyboard_focus_accessibility(df):
    """Analyze keyboard focus accessibility issues"""
    print("\n===== KEYBOARD FOCUS ACCESSIBILITY ANALYSIS =====")
    
    # Find keyboard focus columns - updated for the new structure
    keyboard_cols = [col for col in df.columns if 'keyboard' in col.lower() or 'focus' in col.lower()]
    
    if not keyboard_cols:
        print("No keyboard focus accessibility columns found in the dataset.")
        return None
    
    # Simplify output
    print(f"Found {len(keyboard_cols)} keyboard focus accessibility columns.")
    
    # Count sites with keyboard focus issues
    keyboard_focus_issues = df[keyboard_cols[0]].notna().sum() if keyboard_cols else 0
    total_tested = df.shape[0]
    
    print(f"\nKeyboard Focus: {keyboard_focus_issues} sites with issues ({keyboard_focus_issues/total_tested*100:.1f}%)")
    
    # Extract severity information from keyboard columns if available
    severity_data = None
    for col in keyboard_cols:
        if 'Interactive_controls' in col and df[col].notna().sum() > 0:
            severity_data = analyze_embedded_severity(df, col)
            break
    
    # Removing keyboard focus plot as requested
    # No longer creating the keyboard focus accessibility visualization
    
    return {'tested': total_tested, 'issues': keyboard_focus_issues, 'severity': severity_data}

def create_accessibility_dashboard(df):
    """Create a simplified dashboard with key accessibility insights"""
    print("\n===== CREATING ACCESSIBILITY DASHBOARD =====")
    
    # Create figure with subplots - reduce to 2x2 grid
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # 1. Score Distribution Plot
    ax_scores = fig.add_subplot(gs[0, 0])
    
    # Get score columns
    score_cols = [col for col in df.columns if 'Score' in col and col != 'AccessibilityResult']
    for i, col in enumerate(score_cols):
        sns.kdeplot(df[col].dropna(), ax=ax_scores, label=col.replace('_Score', ''), 
                   color=red_blue_palette[i % len(red_blue_palette)])
    
    ax_scores.axvline(x=90, color='#D21F3C', linestyle='--', label='Pass threshold (90)')
    ax_scores.set_title('Score Distributions')
    ax_scores.set_xlabel('Score')
    ax_scores.set_ylabel('Density')
    ax_scores.legend()
    
    # 2. Pass/Fail Pie Chart
    ax_pass_fail = fig.add_subplot(gs[0, 1])
    
    if 'AccessibilityResult' in df.columns:
        pass_counts = df['AccessibilityResult'].value_counts()
        ax_pass_fail.pie(pass_counts, labels=pass_counts.index, autopct='%1.1f%%', startangle=90,
                colors=['#8B0000','#00008B'], textprops={'color': 'white'})
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
            top_issues = sorted_issues[:5]  # Top 5 issues
            
            issues = [issue[:30] + '...' if len(issue) > 30 else issue for issue, _ in top_issues]
            counts = [count for _, count in top_issues]
            
            bars = ax_issues.barh(issues[::-1], counts[::-1], color='#8B0000')
            
            # Add count labels
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 1
                ax_issues.text(label_x_pos, bar.get_y() + bar.get_height()/2, str(int(width)),
                       va='center', color='#00008B')
            
            ax_issues.set_title('Most Common Accessibility Issues')
            ax_issues.set_xlabel('Number of Sites')
    
    # 4. Severity Distribution
    ax_severity = fig.add_subplot(gs[1, 1])
    
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
            colors = {'High': '#8B0000', 'Medium': '#D21F3C', 'Low': '#00008B', 'Unknown': '#808080'}
            
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
            
            # Add value labels - make them white when on dark bars
            for bar in bars:
                width = bar.get_width()
                ax_severity.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                               str(int(width)), va='center', color='#00008B')
            
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
    """Generate key insights from the analysis"""
    print("\n===== KEY INSIGHTS =====")
    
    insights = []
    
    # 1. Overall Accessibility Score and Pass Rate
    avg_score = score_stats.loc['mean', 'Accessibility_Score']
    pass_rate = (df['Accessibility_Score'] >= 90).mean() * 100
    
    insights.append(f"The average accessibility score is {avg_score:.1f}/100, with {pass_rate:.1f}% of sites passing the accessibility standard (≥90).")
    
    # 2. Most common accessibility issues
    if issue_counts and len(issue_counts) > 0:
        issue_df = pd.DataFrame({'Issue': list(issue_counts.keys()), 'Count': list(issue_counts.values())})
        issue_df = issue_df.sort_values('Count', ascending=False)
        
        top_issue = issue_df.iloc[0]['Issue']
        top_count = issue_df.iloc[0]['Count']
        
        insights.append(f"The most common accessibility issue is '{top_issue}', found on {top_count} sites.")
        
        if len(issue_df) > 1:
            second_issue = issue_df.iloc[1]['Issue']
            second_count = issue_df.iloc[1]['Count']
            insights.append(f"The second most common issue is '{second_issue}', found on {second_count} sites.")
    
    # 3. Keyboard accessibility insights
    keyboard_cols = [col for col in df.columns if 'keyboard' in col.lower() or 'focus' in col.lower()]
    keyboard_issue_count = df[keyboard_cols[0]].notna().sum() if keyboard_cols and len(keyboard_cols) > 0 else 0
    
    if keyboard_issue_count > 0:
        keyboard_percent = keyboard_issue_count / len(df) * 100
        insights.append(f"{keyboard_issue_count} sites ({keyboard_percent:.1f}%) have keyboard accessibility issues.")
    
    # 4. Best and worst performers
    if len(domain_scores) > 0:
        best_domain = domain_scores.idxmax()
        best_score = domain_scores.max()
        worst_domain = domain_scores.idxmin()
        worst_score = domain_scores.min()
        
        insights.append(f"The best performing domain is {best_domain} with an accessibility score of {best_score:.1f}.")
        insights.append(f"The worst performing domain is {worst_domain} with an accessibility score of {worst_score:.1f}.")
    
    # 5. Score comparisons
    for score_type in ['Performance_Score', 'Best_Practices_Score', 'SEO_Score']:
        if score_type in score_stats.columns:
            score_avg = score_stats.loc['mean', score_type]
            comparison = "higher than" if score_avg > avg_score else "lower than"
            insights.append(f"The average {score_type.replace('_', ' ')} ({score_avg:.1f}) is {comparison} the accessibility score.")
    
    # 6. Score distribution insights
    min_score = score_stats.loc['min', 'Accessibility_Score']
    max_score = score_stats.loc['max', 'Accessibility_Score']
    median_score = score_stats.loc['50%', 'Accessibility_Score']
    
    score_range = max_score - min_score
    insights.append(f"Accessibility scores range from {min_score:.1f} to {max_score:.1f} (range of {score_range:.1f} points), with a median of {median_score:.1f}.")
    
    # 7. Potential severity insights
    high_severity_col = [col for col in df.columns if 'high' in col.lower() and 'severity' in col.lower()]
    if high_severity_col and len(high_severity_col) > 0:
        high_severity_count = df[high_severity_col[0]].notna().sum()
        high_severity_percent = high_severity_count / len(df) * 100
        insights.append(f"{high_severity_count} sites ({high_severity_percent:.1f}%) have high severity accessibility issues.")
    
    # Print insights
    for i, insight in enumerate(insights):
        print(f"{i+1}. {insight}")
    
    # Create insights text
    insights_text = "# Key Accessibility Insights\n\n"
    for i, insight in enumerate(insights):
        insights_text += f"{i+1}. {insight}\n"
    
    # Save to file
    file_path = os.path.join(output_dir, "accessibility_insights_summary.txt")
    with open(file_path, 'w') as f:
        f.write(insights_text)
    
    print(f"\nSaved accessibility insights to {file_path}")
    
    return insights, insights_text

def create_insights_summary(df, score_stats, domain_scores, issue_counts):
    """Create a comprehensive summary of accessibility insights and recommendations"""
    print("\n===== CREATING INSIGHTS SUMMARY =====")
    
    # Calculate key metrics
    sites_count = len(df)
    domains_count = len(domain_scores)
    avg_accessibility = score_stats.loc['mean', 'Accessibility_Score']
    pass_rate = (df['Accessibility_Score'] >= 90).mean() * 100
    
    # Calculate average issues per site
    if 'Accessibility_Issues_Count' in df.columns:
        # Use the pre-calculated count if available
        avg_issues = df['Accessibility_Issues_Count'].mean() if df['Accessibility_Issues_Count'].notna().any() else 'N/A'
    else:
        # Otherwise calculate from issue columns
        issue_cols = [col for col in df.columns if ('A11y_Issue' in col) or ('not_have' in col) or ('lacks' in col)]
        if issue_cols:
            # Count non-null values in any issue column
            has_issues = df[issue_cols].notna().any(axis=1)
            
            if has_issues.sum() > 0:
                # Count total issues and calculate average
                total_issues = df[issue_cols].notna().sum().sum()
                sites_with_issues = has_issues.sum()
                avg_issues = total_issues / sites_with_issues
            else:
                avg_issues = 'N/A'
        else:
            avg_issues = 'N/A'
    
    # Generate summary text
    summary = f"""# Healthcare Website Accessibility Analysis Summary

## Overview
- **Websites Analyzed:** {sites_count}
- **Unique Domains:** {domains_count}
- **Average Accessibility Score:** {avg_accessibility:.1f}/100
- **Pass Rate (≥90):** {pass_rate:.1f}%
- **Average number of accessibility issues:** {avg_issues if isinstance(avg_issues, str) else f"{avg_issues:.1f}"} per site

## Key Findings

"""
    
    # Add findings about common issues
    if issue_counts and len(issue_counts) > 0:
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        top_5_issues = sorted_issues[:5]
        
        summary += "### Common Accessibility Issues\n"
        for issue, count in top_5_issues:
            percentage = (count / sites_count) * 100
            summary += f"- **{issue}:** Found on {count} sites ({percentage:.1f}%)\n"
    
    # Add findings about keyboard accessibility
    keyboard_cols = [col for col in df.columns if 'keyboard' in col.lower() or 'focus' in col.lower()]
    if keyboard_cols and len(keyboard_cols) > 0:
        keyboard_focus_issues = df[keyboard_cols[0]].notna().sum()
        if keyboard_focus_issues > 0:
            kb_percentage = (keyboard_focus_issues / sites_count) * 100
            summary += f"\n### Keyboard Focus Accessibility\n"
            summary += f"- {keyboard_focus_issues} sites ({kb_percentage:.1f}%) have keyboard accessibility issues\n"
    
    # Add domain highlights
    if len(domain_scores) > 0:
        summary += "\n### Domain Performance\n"
        
        # Top domains
        top_domains = domain_scores.sort_values(ascending=False).head(3)
        summary += "#### Top Performers\n"
        for domain, score in top_domains.items():
            summary += f"- **{domain}:** {score:.1f}/100\n"
        
        # Bottom domains
        bottom_domains = domain_scores.sort_values().head(3)
        summary += "\n#### Needs Improvement\n"
        for domain, score in bottom_domains.items():
            summary += f"- **{domain}:** {score:.1f}/100\n"
    
    # Add recommendations
    summary += """
## Recommendations

### High Priority
1. **Fix keyboard accessibility issues** - Ensure all interactive elements are keyboard accessible
2. **Address high severity issues** - Focus on high impact issues that affect many users
3. **Ensure proper image alt text** - All images should have appropriate alternative text

### Medium Priority
1. **Improve ARIA attributes** - Ensure ARIA attributes are used correctly
2. **Fix form labeling** - All form elements should have proper labels
3. **Address color contrast issues** - Ensure text has sufficient contrast with background colors

### Low Priority
1. **Improve document structure** - Use proper heading hierarchy
2. **Add landmark regions** - Improve navigation with proper landmarks
3. **Review link text** - Ensure all links have descriptive text

## Next Steps
1. Share these findings with development teams
2. Prioritize fixes based on severity and impact
3. Implement an accessibility testing process
4. Create a timeline for addressing critical issues
5. Perform follow-up testing after remediation
"""
    
    # Save the summary
    file_path = os.path.join(output_dir, "insights_summary.txt")
    with open(file_path, 'w') as f:
        f.write(summary)
    
    print(f"Saved insights summary to {file_path}")
    
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
            # Simplify output to just key information
            print(f"\nSeverity from {tested_sites} sites: {', '.join([f'{k}: {v}' for k, v in severity_data.items()])}")
            
            # Create a single figure with a pie chart
            plt.figure(figsize=(12, 9))
            
            # Define colors for severity levels using red and blue palette
            colors = {'High': '#8B0000', 'Medium': '#D21F3C', 'Low': '#00008B', 'Unknown': '#808080'}
            sorted_data = sorted(severity_data.items(), 
                               key=lambda x: {'High': 0, 'Medium': 1, 'Low': 2, 'Unknown': 3}.get(x[0], 4))
            labels = [f"{k} ({v})" for k, v in sorted_data]
            sizes = [v for _, v in sorted_data]
            
            # Explode the high severity slice
            explode = [0.1 if k == 'High' else 0.05 if k == 'Medium' else 0 for k, _ in sorted_data]
            
            plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90,
                   colors=[colors.get(k, '#BDBDBD') for k, _ in sorted_data], shadow=True, 
                   textprops={'color': 'white'})
            plt.axis('equal')
            plt.title('Keyboard Focus Issues by Severity', fontsize=16)
            
            # Improve overall appearance
            plt.tight_layout()
            
            # Save the plot
            file_path = os.path.join(output_dir, "severity_distribution.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Saved severity distribution to {file_path}")
            
            return severity_data
    
    # Simplified output for proxy approach
    print("Creating severity distribution visualization...")
    
    if 'Accessibility_Score' in df.columns:
        # Create bins for accessibility score
        bins = [0, 50, 70, 90, 100]
        labels = ['Critical (0-50)', 'High (51-70)', 'Medium (71-90)', 'Low (91-100)']
        
        df['SeverityProxy'] = pd.cut(df['Accessibility_Score'], bins=bins, labels=labels, right=True)
        severity_counts = df['SeverityProxy'].value_counts().sort_index()
        
        # Create visualization - skip verbose output
        plt.figure(figsize=(12, 8))
        
        # Create bar chart with custom colors
        colors = ['#8B0000', '#D21F3C', '#0000CD', '#00008B']
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
        print(f"Saved severity distribution to {file_path}")
        
        return severity_counts.to_dict()
    
    return None

def analyze_embedded_severity(df, column):
    """Analyze severity embedded in issue column values"""
    print(f"Extracting embedded severity from {column}")
    
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
        print(f"Extracted severity from {total_issues} issue descriptions")
        
        # Simplify output - just a single summary line
        print("Severity Distribution: " + 
              ", ".join([f"{k}: {v} ({v/total_issues*100:.1f}%)" for k, v in severity_counts.items() if v > 0]))
        
        # Create a visualization of the severity distribution
        plt.figure(figsize=(12, 8))
        
        # Create pie chart with red and blue color scheme
        colors = {'High': '#8B0000', 'Medium': '#D21F3C', 'Low': '#00008B', 'Unknown': '#808080'}
        
        # Filter out zero counts to avoid empty slices
        non_zero_counts = {k: v for k, v in severity_counts.items() if v > 0}
        if non_zero_counts:
            labels = [f"{k} ({v})" for k, v in non_zero_counts.items()]
            sizes = list(non_zero_counts.values())
            
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                    colors=[colors.get(k, 'gray') for k in non_zero_counts.keys()],
                    explode=[0.1 if k == 'High' else (0.05 if k == 'Medium' else 0) for k in non_zero_counts.keys()],
                    textprops={'color': 'white'})
            
            plt.axis('equal')
            plt.title(f'Severity Distribution: {column.replace("_", " ")}')
            
            # Save visualization
            file_path = os.path.join(output_dir, "severity_distribution.png")
            plt.savefig(file_path)
            print(f"Saved severity distribution to {file_path}")
            
            return severity_counts
    
    return None

def analyze_accessibility_pass_rates_over_time(df):
    """Analyze and visualize accessibility pass rates over time if timestamp data is available"""
    print("\n===== ACCESSIBILITY PASS RATES ANALYSIS =====")
    
    # Check if there's a timestamp or date column
    date_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'timestamp'])]
    
    if not date_cols:
        print("No date/time columns found for temporal analysis")
        
        # Create alternative visualization - pass rates by score category
        plt.figure(figsize=(12, 8))
        
        # Get all score columns
        score_cols = [col for col in df.columns if col.endswith('_Score')]
        pass_rates = {}
        
        # Calculate pass rates for different thresholds
        thresholds = [70, 80, 90, 95]
        
        for col in score_cols:
            category = col.replace('_Score', '')
            pass_rates[category] = [100 * (df[col] >= threshold).mean() for threshold in thresholds]
        
        # Plot as grouped bar chart
        categories = list(pass_rates.keys())
        x = np.arange(len(thresholds))
        width = 0.8 / len(categories)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, (category, rates) in enumerate(pass_rates.items()):
            offset = width * i - (width * (len(categories) - 1) / 2)
            bars = ax.bar(x + offset, rates, width, label=category, 
                         color=red_blue_palette[i % len(red_blue_palette)])
            
            # Add percentage labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', 
                       color='#00008B', fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'≥{threshold}' for threshold in thresholds])
        ax.set_ylabel('Pass Rate (%)')
        ax.set_xlabel('Score Threshold')
        ax.set_title('Pass Rates by Score Category and Threshold')
        ax.set_ylim(0, 100)
        ax.legend(title='Score Category')
        
        plt.tight_layout()
        file_path = os.path.join(output_dir, "pass_rates_by_threshold.png")
        plt.savefig(file_path)
        print(f"Saved pass rates visualization to {file_path}")
        
        # Create a visualization of pass vs. fail by score category
        plt.figure(figsize=(14, 8))
        
        # Standard threshold for passing is 90
        pass_fail_data = []
        
        for col in score_cols:
            category = col.replace('_Score', '')
            pass_count = (df[col] >= 90).sum()
            fail_count = (df[col] < 90).sum()
            pass_fail_data.append({
                'Category': category,
                'Pass': pass_count,
                'Fail': fail_count
            })
        
        # Convert to DataFrame for easier plotting
        pass_fail_df = pd.DataFrame(pass_fail_data)
        
        # Create stacked bar chart
        ax = pass_fail_df.plot(x='Category', y=['Pass', 'Fail'], kind='bar', stacked=True,
                             color=['#00008B', '#8B0000'], figsize=(12, 8))
        
        # Add percentage annotations
        for i, row in enumerate(pass_fail_df.itertuples()):
            total = row.Pass + row.Fail
            pass_pct = 100 * row.Pass / total
            fail_pct = 100 * row.Fail / total
            
            # Pass percentage (on Pass section)
            ax.text(i, row.Pass / 2, f'{pass_pct:.1f}%', 
                   ha='center', va='center', color='white', fontweight='bold')
            
            # Fail percentage (on Fail section)
            ax.text(i, row.Pass + (row.Fail / 2), f'{fail_pct:.1f}%',
                   ha='center', va='center', color='white', fontweight='bold')
        
        plt.title('Pass vs. Fail by Score Category (Threshold ≥90)')
        plt.xlabel('Score Category')
        plt.ylabel('Number of Sites')
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, "pass_fail_by_category.png")
        plt.savefig(file_path)
        print(f"Saved pass/fail by category visualization to {file_path}")
        
        return pass_rates
    
    # If we have date information, continue with temporal analysis
    # ... potential code for temporal analysis if needed in future ...
    
    return None

def analyze_accessibility_issues_relationships(df):
    """Analyze relationships between different accessibility issues"""
    print("\n===== ACCESSIBILITY ISSUES RELATIONSHIPS =====")
    
    # Find all accessibility issue columns
    issue_cols = []
    for col in df.columns:
        # Check for various patterns indicating issue columns
        if (col.startswith('A11y_Issue_') or 
            'not_have' in col or 
            'lacks' in col or 
            'not_specified' in col or
            'Interactive_controls' in col):
            issue_cols.append(col)
    
    if len(issue_cols) <= 1:
        print("Not enough issue columns for relationship analysis")
        return None
    
    # Create a correlation matrix of issues (binary: present=1, absent=0)
    issue_matrix = pd.DataFrame()
    
    for col in issue_cols:
        # Convert to binary values (NaN = 0, any value = 1)
        display_name = col.replace('A11y_Issue_', '').replace('_', ' ')
        if len(display_name) > 30:
            display_name = display_name[:25] + '...'
        issue_matrix[display_name] = df[col].notna().astype(int)
    
    # Calculate correlation
    issue_corr = issue_matrix.corr()
    
    # Visualize the correlation matrix
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(issue_corr, dtype=bool))
    
    # Use a red-blue colormap for correlations
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Plot heatmap
    sns.heatmap(issue_corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title('Correlations Between Accessibility Issues', fontsize=16)
    plt.tight_layout()
    
    file_path = os.path.join(output_dir, "issue_correlations.png")
    plt.savefig(file_path)
    print(f"Saved issue correlations to {file_path}")
    
    # Create a co-occurrence network visualization - simplified version
    plt.figure(figsize=(14, 10))
    
    # Count co-occurrences between top issues
    top_issues = issue_matrix.sum().sort_values(ascending=False).head(5).index.tolist()
    co_occurrence = pd.DataFrame(index=top_issues, columns=top_issues, data=0)
    
    for i, issue1 in enumerate(top_issues):
        for issue2 in top_issues[i:]:  # Only upper triangle
            if issue1 != issue2:
                # Count sites with both issues
                both = ((issue_matrix[issue1] == 1) & (issue_matrix[issue2] == 1)).sum()
                co_occurrence.loc[issue1, issue2] = both
                co_occurrence.loc[issue2, issue1] = both
    
    # Create a directed graph visualization
    from matplotlib.patches import FancyArrowPatch
    
    # Define node positions in a circle
    import math
    n_nodes = len(top_issues)
    radius = 0.8
    node_positions = {}
    node_colors = {}
    
    for i, issue in enumerate(top_issues):
        angle = 2 * math.pi * i / n_nodes
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        node_positions[issue] = (x, y)
        node_colors[issue] = red_blue_palette[i % len(red_blue_palette)]
    
    # Plot nodes
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Add edges (connections)
    max_co = co_occurrence.max().max()
    min_width = 1
    max_width = 8
    
    for issue1 in top_issues:
        for issue2 in top_issues:
            if issue1 != issue2:
                co_value = co_occurrence.loc[issue1, issue2]
                if co_value > 0:
                    # Scale line width by co-occurrence value
                    width = min_width + ((co_value / max_co) * (max_width - min_width))
                    ax.add_patch(FancyArrowPatch(
                        node_positions[issue1], node_positions[issue2],
                        arrowstyle='-', mutation_scale=20, 
                        linewidth=width, alpha=0.7, color='#808080'))
    
    # Plot nodes over edges
    for issue in top_issues:
        x, y = node_positions[issue]
        # Plot node
        count = issue_matrix[issue].sum()
        size = 1000 * (count / issue_matrix.shape[0])  # Size based on prevalence
        ax.scatter(x, y, s=size, color=node_colors[issue], alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add labels with count
        ax.text(x, y, f"{issue}\n({count})", ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
    
    ax.set_title('Co-occurrence of Top 5 Accessibility Issues', fontsize=16)
    ax.axis('off')
    
    file_path = os.path.join(output_dir, "issue_co_occurrence.png")
    plt.savefig(file_path)
    print(f"Saved issue co-occurrence visualization to {file_path}")
    
    return issue_corr

def analyze_accessibility_scores_by_issue(df):
    """Analyze how accessibility issues affect overall scores"""
    print("\n===== ACCESSIBILITY SCORE BY ISSUE ANALYSIS =====")
    
    # Get issue columns
    issue_cols = []
    for col in df.columns:
        if (col.startswith('A11y_Issue_') or 
            'not_have' in col or 
            'lacks' in col or 
            'not_specified' in col or
            'Interactive_controls' in col):
            issue_cols.append(col)
    
    if not issue_cols:
        print("No issue columns found for score impact analysis")
        return None
    
    # Get top 5 most common issues
    issue_counts = {}
    for col in issue_cols:
        display_name = col.replace('A11y_Issue_', '').replace('_', ' ')
        count = df[col].notna().sum()
        if count > 0:
            issue_counts[col] = {'name': display_name, 'count': count}
    
    sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1]['count'], reverse=True)
    top_issues = [item[0] for item in sorted_issues[:5]]
    
    if not top_issues:
        print("No issues found with sufficient data")
        return None
    
    # Create box plots showing how each issue affects accessibility score
    plt.figure(figsize=(14, 10))
    
    issue_score_data = []
    
    # For each top issue, compare scores for sites with and without the issue
    for col in top_issues:
        display_name = issue_counts[col]['name']
        
        # Sites with the issue
        has_issue = df[col].notna()
        
        # Get median scores for sites with and without the issue
        with_issue_median = df.loc[has_issue, 'Accessibility_Score'].median()
        without_issue_median = df.loc[~has_issue, 'Accessibility_Score'].median()
        
        # Store data for plotting
        issue_score_data.append({
            'Issue': display_name[:30] + '...' if len(display_name) > 30 else display_name,
            'With Issue': df.loc[has_issue, 'Accessibility_Score'].tolist(),
            'Without Issue': df.loc[~has_issue, 'Accessibility_Score'].tolist(),
            'With_Median': with_issue_median,
            'Without_Median': without_issue_median,
            'Impact': without_issue_median - with_issue_median
        })
    
    # Sort by impact
    issue_score_data.sort(key=lambda x: x['Impact'], reverse=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Positions for the box pairs
    positions = np.arange(len(issue_score_data)) * 3
    width = 0.8
    
    # Plot box plots
    for i, data in enumerate(issue_score_data):
        # With issue box (red)
        if data['With Issue']:
            with_box = ax.boxplot([data['With Issue']], positions=[positions[i] - width/2],
                             widths=width, patch_artist=True,
                             boxprops=dict(facecolor='#8B0000', color='black'),
                             medianprops=dict(color='white', linewidth=2),
                             flierprops=dict(marker='o', markerfacecolor='#D21F3C'))
        
        # Without issue box (blue)
        if data['Without Issue']:
            without_box = ax.boxplot([data['Without Issue']], positions=[positions[i] + width/2],
                                widths=width, patch_artist=True,
                                boxprops=dict(facecolor='#00008B', color='black'),
                                medianprops=dict(color='white', linewidth=2),
                                flierprops=dict(marker='o', markerfacecolor='#0000CD'))
        
        # Add median value labels
        ax.text(positions[i] - width/2, data['With_Median'],
               f"{data['With_Median']:.1f}", ha='center', va='bottom',
               color='white', fontweight='bold')
        
        ax.text(positions[i] + width/2, data['Without_Median'],
               f"{data['Without_Median']:.1f}", ha='center', va='bottom',
               color='white', fontweight='bold')
        
        # Add impact score
        ax.text(positions[i], min(data['With_Median'], data['Without_Median']) - 5,
               f"Impact: {data['Impact']:.1f} pts", ha='center', 
               color='#D21F3C' if data['Impact'] > 0 else '#00008B',
               fontweight='bold')
    
    # Set labels and title
    ax.set_ylabel('Accessibility Score')
    ax.set_title('Impact of Accessibility Issues on Overall Score', fontsize=16)
    
    # Set x-tick labels
    ax.set_xticks(positions)
    ax.set_xticklabels([data['Issue'] for data in issue_score_data], rotation=45, ha='right')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#8B0000', edgecolor='black', label='Sites with Issue'),
        Patch(facecolor='#00008B', edgecolor='black', label='Sites without Issue')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Set y-axis limits
    ax.set_ylim(40, 105)
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, "issue_score_impact.png")
    plt.savefig(file_path)
    print(f"Saved issue score impact visualization to {file_path}")
    
    return issue_score_data

def main():
    """Main function to execute accessibility analysis"""
    # Load data - updated to use the new file
    df = load_data()
    
    # Filter out records with missing scores
    df_filtered = df.dropna(subset=['Accessibility_Score'], how='any')
    
    # Run existing analyses
    score_stats, pass_rates = analyze_scores(df_filtered)
    domain_scores = analyze_domains(df_filtered)
    issue_counts = analyze_accessibility_issues(df_filtered)
    keyboard_focus_results = analyze_keyboard_focus_accessibility(df_filtered)
    severity_distribution = analyze_severity_distribution(df_filtered)
    
    # Run new analyses
    pass_rate_analysis = analyze_accessibility_pass_rates_over_time(df_filtered)
    issue_relationships = analyze_accessibility_issues_relationships(df_filtered)
    issue_impact = analyze_accessibility_scores_by_issue(df_filtered)
    
    # Create dashboard visualization
    create_accessibility_dashboard(df_filtered)
    
    # Generate insights - but with reduced verbosity
    print("\n===== CREATING FINAL REPORTS =====")
    insights, insights_text = generate_insights(df_filtered, score_stats, domain_scores, issue_counts)
    
    # Create insights summary
    create_insights_summary(df_filtered, score_stats, domain_scores, issue_counts)
    
    print("\nAnalysis complete. Results saved to " + output_dir)

if __name__ == "__main__":
    main() 