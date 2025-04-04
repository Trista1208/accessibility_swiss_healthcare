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

# Set custom color palette with red and dark blue
red_blue_palette = ["#8B0000", "#00008B", "#D21F3C", "#0000CD", "#FF0000", "#0000FF"]
sns.set_palette(red_blue_palette)

# Create output directory for visualization
output_dir = "final_accessibility_analysis"
os.makedirs(output_dir, exist_ok=True)

def analyze_accessibility_data(df):
    """Analyze accessibility scores and issues"""
    print("\n===== ACCESSIBILITY ANALYSIS =====")
    
    # Basic statistics
    print("\nAccessibility Score Statistics:")
    print(df['Accessibility'].describe())
    
    # Pass/Fail distribution with enhanced visualization
    pass_rate = (df['Accessibility'] >= 90).mean() * 100
    print(f"\nPass Rate (≥90): {pass_rate:.1f}%")
    
    # Create enhanced pass/fail pie chart
    plt.figure(figsize=(12, 12))
    pass_counts = (df['Accessibility'] >= 90).value_counts()
    colors = ['#8B0000', '#00008B']
    
    wedges, _ = plt.pie(
        pass_counts,
        colors=colors,
        startangle=90,
        explode=(0.05, 0.05)
    )
    
    # Add simplified labels with PASS/FAIL and percentage only
    for i, wedge in enumerate(wedges):
        pct = 100 * pass_counts[i] / len(df)
        status = "PASS" if i == 1 else "FAIL"
        
        ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        
        # Status label with percentage
        plt.annotate(
            f"{status}\n{pct:.1f}%", 
            xy=(x * 0.6, y * 0.6),
            ha='center', va='center',
            fontsize=20, fontweight='bold',
            color='white'
        )
    
    plt.title('Accessibility Pass/Fail Distribution\n(≥90 score is passing)', fontsize=16, pad=20)
    
    file_path = os.path.join(output_dir, "accessibility_pass_fail.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    
    # Create score distribution plot
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df['Accessibility'].dropna(), bins=20, color='#8B0000', kde=True)
    plt.axvline(x=90, color='#00008B', linestyle='--', label='Pass threshold (90)')
    plt.title('Distribution of Accessibility Scores')
    plt.xlabel('Score')
    plt.ylabel('Number of Sites')
    plt.legend()
    
    file_path = os.path.join(output_dir, "score_distribution.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    
    # Analyze issues with enhanced visualization
    print("\nIssue Statistics:")
    issue_stats = df[['Total_Issues', 'High_Severity_Issues']].describe()
    print(issue_stats)
    
    # Create enhanced issues boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['Total_Issues', 'High_Severity_Issues']], palette=['#8B0000', '#00008B'])
    plt.title('Distribution of Accessibility Issues')
    plt.ylabel('Number of Issues')
    
    # Add mean value annotations with white text and background
    means = df[['Total_Issues', 'High_Severity_Issues']].mean()
    for i, mean_val in enumerate(means):
        plt.text(i, mean_val, f'Mean: {mean_val:.1f}', 
                ha='center', va='bottom', fontweight='bold', color='white',
                bbox=dict(facecolor='#00008B', edgecolor='none', alpha=0.7, pad=2))
    
    file_path = os.path.join(output_dir, "issues_distribution.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    
    return pass_rate, issue_stats

def analyze_domain_performance(df):
    """Analyze accessibility by domain"""
    print("\n===== DOMAIN PERFORMANCE =====")
    
    domain_scores = df.groupby('Domain')['Accessibility'].agg(['mean', 'count']).round(1)
    domain_scores = domain_scores.sort_values('mean', ascending=False)
    
    print("\nTop 5 Performing Domains:")
    print(domain_scores.head())
    print("\nBottom 5 Performing Domains:")
    print(domain_scores.tail())
    
    return domain_scores

def analyze_specific_issues(df):
    """Analyze specific accessibility issues"""
    print("\n===== SPECIFIC ISSUES ANALYSIS =====")
    
    # Find columns related to specific issues
    issue_cols = [col for col in df.columns if any(term in col.lower() for term in [
        'document language', 'links', 'form elements', '[alt]',
        'keyboard', 'focus', 'aria', 'wcag'
    ])]
    
    if not issue_cols:
        print("No specific issue columns found")
        return None
    
    # Calculate issue frequencies
    issue_freq = {}
    for col in issue_cols:
        count = df[col].notna().sum()
        if count > 0:
            issue_freq[col] = {
                'count': count,
                'percentage': (count/len(df)*100)
            }
    
    # Sort and display results
    sorted_issues = sorted(issue_freq.items(), key=lambda x: x[1]['count'], reverse=True)
    print("\nMost Common Issues:")
    for issue, stats in sorted_issues[:10]:
        print(f"- {issue}: {stats['count']} sites ({stats['percentage']:.1f}%)")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    issues = [issue[:40] + '...' if len(issue) > 40 else issue for issue, _ in sorted_issues[:10]]
    counts = [stats['count'] for _, stats in sorted_issues[:10]]
    
    bars = plt.barh(issues[::-1], counts[::-1], color='#8B0000')
    
    # Add count labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{int(width)} ({width/len(df)*100:.1f}%)',
                va='center', color='#00008B', fontweight='bold')
    
    plt.xlabel('Number of Sites Affected')
    plt.title('Most Common Accessibility Issues')
    plt.tight_layout()
    
    file_path = os.path.join(output_dir, "specific_issues.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    
    return issue_freq

def create_accessibility_dashboard(df):
    """Create a comprehensive dashboard with key accessibility insights"""
    print("\n===== CREATING ACCESSIBILITY DASHBOARD =====")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # 1. Score Distribution Plot
    ax_scores = fig.add_subplot(gs[0, 0])
    sns.histplot(data=df['Accessibility'].dropna(), bins=20, ax=ax_scores, color='#8B0000', kde=True)
    ax_scores.axvline(x=90, color='#00008B', linestyle='--', label='Pass threshold (90)')
    ax_scores.set_title('Accessibility Score Distribution')
    ax_scores.set_xlabel('Score')
    ax_scores.set_ylabel('Number of Sites')
    ax_scores.legend()
    
    # 2. Pass/Fail Distribution
    ax_pass_fail = fig.add_subplot(gs[0, 1])
    pass_counts = (df['Accessibility'] >= 90).value_counts()
    colors = ['#8B0000', '#00008B']
    
    wedges, _ = ax_pass_fail.pie(
        pass_counts,
        colors=colors,
        startangle=90,
        explode=(0.05, 0.05)
    )
    
    # Add simplified labels with PASS/FAIL and percentage only
    for i, wedge in enumerate(wedges):
        pct = 100 * pass_counts[i] / len(df)
        status = "PASS" if i == 1 else "FAIL"
        
        ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        
        # Status label with percentage
        ax_pass_fail.annotate(
            f"{status}\n{pct:.1f}%", 
            xy=(x * 0.6, y * 0.6),
            ha='center', va='center',
            fontsize=20, fontweight='bold',
            color='white'
        )
    
    ax_pass_fail.set_title('Pass/Fail Distribution')
    
    # 3. Issues Distribution
    ax_issues = fig.add_subplot(gs[1, 0])
    sns.boxplot(data=df[['Total_Issues', 'High_Severity_Issues']], 
                palette=['#8B0000', '#00008B'], ax=ax_issues)
    ax_issues.set_title('Distribution of Issues')
    ax_issues.set_ylabel('Number of Issues')
    
    # Add mean annotations with white text and background
    means = df[['Total_Issues', 'High_Severity_Issues']].mean()
    for i, mean_val in enumerate(means):
        ax_issues.text(i, mean_val, f'Mean: {mean_val:.1f}', 
                      ha='center', va='bottom', fontweight='bold', color='white',
                      bbox=dict(facecolor='#00008B', edgecolor='none', alpha=0.7, pad=2))
    
    # 4. Score vs Issues Scatter
    ax_scatter = fig.add_subplot(gs[1, 1])
    scatter = sns.scatterplot(data=df, x='Total_Issues', y='Accessibility', 
                   hue='High_Severity_Issues', palette='RdBu_r', ax=ax_scatter)
    ax_scatter.set_title('Accessibility Score vs Number of Issues')
    
    # Move legend to the right side
    ax_scatter.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Adjust layout to accommodate the legend
    plt.subplots_adjust(right=0.85)
    
    # 5. Most Common Issues
    ax_issues = fig.add_subplot(gs[2, :])
    issue_cols = [col for col in df.columns if any(term in col.lower() for term in [
        'document language', 'links', 'form elements', '[alt]',
        'keyboard', 'focus', 'aria', 'wcag'
    ])]
    
    issue_counts = {}
    for col in issue_cols:
        count = df[col].notna().sum()
        if count > 0:
            issue_counts[col] = count
    
    sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    issues = [issue[:40] + '...' if len(issue) > 40 else issue for issue, _ in sorted_issues]
    counts = [count for _, count in sorted_issues]
    
    bars = ax_issues.barh(issues[::-1], counts[::-1], color='#8B0000')
    
    # Add count and percentage labels
    for bar in bars:
        width = bar.get_width()
        ax_issues.text(width + 1, bar.get_y() + bar.get_height()/2, 
                      f'{int(width)} ({width/len(df)*100:.1f}%)',
                      va='center', color='#00008B', fontweight='bold')
    
    ax_issues.set_title('Most Common Accessibility Issues')
    ax_issues.set_xlabel('Number of Sites Affected')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Web Accessibility Dashboard - Swiss Healthcare Websites', fontsize=16, y=0.98)
    
    # Save the dashboard
    file_path = os.path.join(output_dir, "accessibility_dashboard.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

def analyze_severity_distribution(df):
    """Analyze and visualize the severity distribution of accessibility issues"""
    print("\n===== SEVERITY DISTRIBUTION ANALYSIS =====")
    
    plt.figure(figsize=(12, 12))
    severity_data = {
        'High': df['High_Severity_Issues'].sum(),
        'Other': df['Total_Issues'].sum() - df['High_Severity_Issues'].sum()
    }
    
    colors = ['#8B0000', '#00008B']
    wedges, _ = plt.pie(
        severity_data.values(),
        colors=colors,
        startangle=90,
        explode=(0.05, 0.05)
    )
    
    # Add custom labels
    for i, (severity, count) in enumerate(severity_data.items()):
        pct = 100 * count / sum(severity_data.values())
        
        ang = (wedges[i].theta2 - wedges[i].theta1) / 2. + wedges[i].theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        
        # Severity label with percentage and count
        plt.annotate(
            f"{severity.upper()}\n{count} issues\n{pct:.1f}%", 
            xy=(x * 0.6, y * 0.6),
            ha='center', va='center',
            fontsize=20, fontweight='bold',
            color='white'
        )
    
    # Add total count in title
    total_issues = sum(severity_data.values())
    plt.title(f'Distribution of Issue Severity\nTotal Issues: {total_issues}', 
              fontsize=16, pad=20)
    
    file_path = os.path.join(output_dir, "severity_distribution.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    
    return severity_data

def create_insights_summary(df, score_stats, domain_scores, issue_counts):
    """Create a comprehensive summary of accessibility insights and recommendations"""
    print("\n===== CREATING INSIGHTS SUMMARY =====")
    
    summary = "# Healthcare Website Accessibility Analysis Summary\n\n"
    
    # Overview Statistics
    summary += "## Overview\n"
    summary += f"- **Websites Analyzed:** {len(df)}\n"
    summary += f"- **Unique Domains:** {len(domain_scores)}\n"
    summary += f"- **Average Accessibility Score:** {df['Accessibility'].mean():.1f}/100\n"
    summary += f"- **Pass Rate (≥90):** {(df['Accessibility'] >= 90).mean()*100:.1f}%\n"
    summary += f"- **Average Issues per Site:** {df['Total_Issues'].mean():.1f}\n"
    summary += f"- **Average High Severity Issues:** {df['High_Severity_Issues'].mean():.1f}\n\n"
    
    # Score Distribution
    summary += "## Score Distribution\n"
    summary += f"- **Minimum Score:** {df['Accessibility'].min():.1f}\n"
    summary += f"- **Maximum Score:** {df['Accessibility'].max():.1f}\n"
    summary += f"- **Median Score:** {df['Accessibility'].median():.1f}\n"
    summary += f"- **Standard Deviation:** {df['Accessibility'].std():.1f}\n\n"
    
    # Common Issues
    if issue_counts:
        summary += "## Common Accessibility Issues\n"
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1]['count'], reverse=True)
        for issue, stats in sorted_issues[:10]:
            summary += f"- **{issue}:** {stats['count']} sites ({stats['percentage']:.1f}%)\n"
        summary += "\n"
    
    # Domain Performance
    if len(domain_scores) > 0:
        summary += "## Domain Performance\n"
        summary += "\nTop 5 Performing Domains:\n"
        top_domains = domain_scores.sort_values('mean', ascending=False).head()
        for idx, row in top_domains.iterrows():
            summary += f"- **{idx}:** {row['mean']:.1f}/100\n"
        
        summary += "\nDomains Needing Improvement:\n"
        bottom_domains = domain_scores.sort_values('mean').head()
        for idx, row in bottom_domains.iterrows():
            summary += f"- **{idx}:** {row['mean']:.1f}/100\n"
        summary += "\n"
    
    # Recommendations
    summary += "## Recommendations\n\n"
    
    summary += "### High Priority\n"
    summary += "1. **Address high severity issues** - Focus on high impact issues that affect many users\n"
    summary += "2. **Fix form accessibility** - Ensure all form elements have proper labels\n"
    summary += "3. **Add image alt text** - All images should have appropriate alternative text\n"
    summary += "4. **Improve ARIA implementation** - Ensure ARIA attributes are used correctly\n"
    summary += "5. **Fix color contrast** - Ensure text has sufficient contrast with background colors\n\n"
    
    summary += "### Medium Priority\n"
    summary += "1. **Improve document structure** - Use proper heading hierarchy\n"
    summary += "2. **Add landmark regions** - Improve navigation with proper landmarks\n"
    summary += "3. **Review link text** - Ensure all links have descriptive text\n"
    summary += "4. **Test with screen readers** - Verify content is accessible with assistive technologies\n"
    summary += "5. **Validate HTML** - Ensure proper semantic markup\n\n"
    
    summary += "### Long-term Strategy\n"
    summary += "1. **Implement accessibility testing** in development workflow\n"
    summary += "2. **Regular audits** - Schedule periodic accessibility assessments\n"
    summary += "3. **Staff training** - Provide accessibility best practices training\n"
    summary += "4. **User testing** - Conduct testing with users who rely on assistive technologies\n"
    summary += "5. **Documentation** - Create accessibility guidelines for content creators\n\n"
    
    summary += "## Next Steps\n"
    summary += "1. Share findings with development teams\n"
    summary += "2. Prioritize fixes based on severity and impact\n"
    summary += "3. Create remediation timeline\n"
    summary += "4. Implement automated accessibility testing\n"
    summary += "5. Schedule follow-up assessment\n"
    
    # Save summary
    file_path = os.path.join(output_dir, "insights_summary.txt")
    with open(file_path, 'w') as f:
        f.write(summary)
    
    print(f"Saved insights summary to {file_path}")
    
    return summary

def analyze_score_correlations(df):
    """Analyze and visualize correlations between different scores"""
    print("\n===== SCORE CORRELATION ANALYSIS =====")
    
    # Calculate correlations between scores and issues
    score_cols = ['Accessibility', 'Total_Issues', 'High_Severity_Issues']
    corr = df[score_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, annot=True, mask=mask, vmin=-1, vmax=1, 
                cmap='RdBu_r', fmt='.2f', linewidths=1)
    
    plt.title("Score and Issue Correlations")
    plt.tight_layout()
    
    file_path = os.path.join(output_dir, "score_correlations.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    
    return corr

def analyze_pass_rates(df):
    """Analyze and visualize pass rates at different thresholds"""
    print("\n===== PASS RATES ANALYSIS =====")
    
    # Calculate pass rates for different thresholds
    thresholds = [70, 80, 90, 95]
    pass_rates = {}
    
    for threshold in thresholds:
        rate = (df['Accessibility'] >= threshold).mean() * 100
        pass_rates[threshold] = rate
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(pass_rates.keys(), pass_rates.values(), color='#8B0000')
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.1f}%',
                ha='center', va='bottom',
                fontweight='bold', color='#00008B')
    
    plt.title('Pass Rates at Different Score Thresholds')
    plt.xlabel('Score Threshold')
    plt.ylabel('Pass Rate (%)')
    plt.xticks(thresholds, [f'≥{t}' for t in thresholds])
    
    file_path = os.path.join(output_dir, "pass_rates.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    
    return pass_rates

def analyze_issue_relationships(df):
    """Analyze relationships between different accessibility issues"""
    print("\n===== ISSUE RELATIONSHIPS ANALYSIS =====")
    
    # Find issue columns
    issue_cols = [col for col in df.columns if any(term in col.lower() for term in [
        'document language', 'links', 'form elements', '[alt]',
        'keyboard', 'focus', 'aria', 'wcag'
    ])]
    
    if not issue_cols:
        return None
    
    # Create binary issue matrix
    issue_matrix = pd.DataFrame()
    for col in issue_cols:
        display_name = col.replace('Accessibility_', '').replace('_', ' ')
        if len(display_name) > 30:
            display_name = display_name[:25] + '...'
        issue_matrix[display_name] = df[col].notna().astype(int)
    
    # Calculate correlation
    issue_corr = issue_matrix.corr()
    
    # Create heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(issue_corr, dtype=bool))
    
    sns.heatmap(issue_corr, mask=mask, cmap='RdBu_r', vmax=1, vmin=-1, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5)
    
    plt.title('Correlations Between Accessibility Issues')
    plt.tight_layout()
    
    file_path = os.path.join(output_dir, "issue_correlations.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    
    return issue_corr

def analyze_score_impact(df):
    """Analyze how different issues impact accessibility scores"""
    print("\n===== SCORE IMPACT ANALYSIS =====")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top 5 most common issues
    issue_cols = [col for col in df.columns if any(term in col.lower() for term in [
        'document language', 'links', 'form elements', '[alt]',
        'keyboard', 'focus', 'aria', 'wcag'
    ])]
    
    issue_counts = {}
    for col in issue_cols:
        count = df[col].notna().sum()
        if count > 0:
            display_name = col.replace('Accessibility_', '').replace('_', ' ')
            if len(display_name) > 30:
                display_name = display_name[:25] + '...'
            issue_counts[display_name] = {'count': count, 'column': col}
    
    sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
    
    # Create simpler plot with bar chart instead of box plots
    plt.figure(figsize=(12, 8))
    
    # Calculate mean scores for each issue
    x_pos = np.arange(len(sorted_issues))
    with_issue_means = []
    without_issue_means = []
    issue_names = []
    
    for issue_name, data in sorted_issues:
        col = data['column']
        with_issue = df[df[col].notna()]['Accessibility'].mean()
        without_issue = df[df[col].isna()]['Accessibility'].mean()
        with_issue_means.append(with_issue)
        without_issue_means.append(without_issue)
        issue_names.append(issue_name)
    
    # Plot grouped bar chart
    bar_width = 0.35
    plt.bar(x_pos - bar_width/2, with_issue_means, bar_width, 
            label='Sites with Issue', color='#8B0000')
    plt.bar(x_pos + bar_width/2, without_issue_means, bar_width,
            label='Sites without Issue', color='#00008B')
    
    # Add value labels in white
    for i in range(len(x_pos)):
        # Label for sites with issue (on red bar)
        plt.text(x_pos[i] - bar_width/2, with_issue_means[i],
                f'{with_issue_means[i]:.1f}',
                ha='center', va='bottom',
                color='white', fontweight='bold')
        
        # Label for sites without issue (on blue bar)
        plt.text(x_pos[i] + bar_width/2, without_issue_means[i],
                f'{without_issue_means[i]:.1f}',
                ha='center', va='bottom',
                color='white', fontweight='bold')
        
        # Add impact score below
        impact = without_issue_means[i] - with_issue_means[i]
        plt.text(x_pos[i], min(with_issue_means[i], without_issue_means[i]) - 5,
                f'Impact: {impact:.1f}',
                ha='center', va='top',
                color='white', fontweight='bold',
                bbox=dict(facecolor='#8B0000' if impact < 0 else '#00008B', 
                         edgecolor='none', alpha=0.7, pad=2))
    
    plt.ylabel('Average Accessibility Score')
    plt.title('Impact of Issues on Accessibility Score')
    plt.xticks(x_pos, issue_names, rotation=45, ha='right')
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with lower DPI
    file_path = os.path.join(output_dir, "score_impact.png")
    plt.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close()

def analyze_metric_comparisons(df):
    """Create a simple comparison plot for different metrics"""
    print("\n===== METRIC COMPARISONS ANALYSIS =====")
    
    # Metrics to compare
    metrics = ['Accessibility', 'Performance', 'SEO', 'Best Practices']
    
    # Calculate statistics
    stats = []
    for metric in metrics:
        # Convert to numeric and cap at 100
        scores = pd.to_numeric(df[metric], errors='coerce')
        scores = scores.clip(upper=100)
        mean = scores.mean()
        pass_rate = (scores >= 90).mean() * 100
        stats.append({
            'Metric': metric,
            'Average Score': mean,
            'Pass Rate': pass_rate
        })
    
    # Print summary
    print("\nSummary Statistics:")
    for stat in stats:
        print(f"\n{stat['Metric']}:")
        print(f"- Average Score: {stat['Average Score']:.1f}")
        print(f"- Pass Rate: {stat['Pass Rate']:.1f}%")
    
    # Create basic plot
    plt.figure(figsize=(12, 6))
    x = range(len(metrics))
    
    # Plot average scores
    bars1 = plt.bar([i - 0.2 for i in x], 
                    [s['Average Score'] for s in stats],
                    0.4,
                    label='Average Score',
                    color='#8B0000')
    
    # Plot pass rates
    bars2 = plt.bar([i + 0.2 for i in x], 
                    [s['Pass Rate'] for s in stats],
                    0.4,
                    label='Pass Rate (%)',
                    color='#00008B')
    
    # Add value labels on bars in black
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.1f}%',
                ha='center', va='bottom',
                color='black', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.1f}%',
                ha='center', va='bottom',
                color='black', fontweight='bold')
    
    # Add labels and title
    plt.xticks(x, metrics, rotation=15)
    plt.ylabel('Score / Pass Rate')
    plt.title('Score and Pass Rate Comparison')
    
    # Move legend to the right side
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Add grid and set y-axis limit
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Adjust layout to accommodate the legend
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "metric_comparisons.png"), bbox_inches='tight')
    plt.close()

def main():
    """Main function to execute the analysis"""
    print("Loading data...")
    df = pd.read_csv('lighthouse_data_cleaned.csv')
    print(f"Loaded {len(df)} records")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analyses
    pass_rate, issue_stats = analyze_accessibility_data(df)
    domain_scores = analyze_domain_performance(df)
    issue_freq = analyze_specific_issues(df)
    
    # Create comprehensive visualizations
    create_accessibility_dashboard(df)
    severity_data = analyze_severity_distribution(df)
    
    # Additional analyses and visualizations
    score_corr = analyze_score_correlations(df)
    pass_rates = analyze_pass_rates(df)
    issue_corr = analyze_issue_relationships(df)
    analyze_score_impact(df)
    analyze_metric_comparisons(df)
    
    # make insights summary
    insights = create_insights_summary(df, issue_stats, domain_scores, issue_freq)
    
    print("\nAnalysis complete. Results saved to " + output_dir)

if __name__ == "__main__":
    main() 