import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from scipy import stats
import re
import shutil
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
    """Analyze accessibility issues to identify patterns and common problems"""
    print("\n===== ACCESSIBILITY ISSUES ANALYSIS =====")
    
    # Find issue columns, which could be of different formats in different datasets
    a11y_cols = []
    
    # Look for columns with known issue-related prefixes
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
    
    # Convert the columns to numeric where possible
    df_numeric = df.copy()
    
    for col in a11y_cols:
        # Try to convert to numeric, setting non-numeric values to NaN
        if pd.api.types.is_object_dtype(df[col]):
            try:
                # For columns with format like "0.72, Severity: Medium", extract the first number
                if df[col].notna().any() and ',' in str(df[col].iloc[0]):
                    df_numeric[col] = df[col].apply(
                        lambda x: float(str(x).split(',')[0]) if pd.notna(x) and ',' in str(x) else np.nan
                    )
                else:
                    # If not in the special format, just try to convert directly
                    df_numeric[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                # If conversion fails, exclude this column from numeric analysis
                a11y_cols.remove(col)
    
    # Count the total number of unique issues
    issue_count = len(a11y_cols)
    print(f"\nTotal unique issue types: {issue_count}")
    
    if issue_count == 0:
        print("No valid issue metrics found for analysis.")
        return None
    
    # For numeric issue columns, calculate means
    numeric_cols = [col for col in a11y_cols if pd.api.types.is_numeric_dtype(df_numeric[col])]
    
    # Calculate average values for numeric columns
    if numeric_cols:
        issue_means = df_numeric[numeric_cols].mean().sort_values(ascending=False)
        print("\nAverage values for each issue type:")
        for issue, mean_val in issue_means.items():
            if not pd.isna(mean_val):
                print(f"- {issue.replace('A11y_Issue_', '')}: {mean_val:.2f}")
    
    # For categorical issue columns, count occurrence
    # First, find columns that represent presence/absence of issues
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
        
        # Save visualization
        file_path = os.path.join(output_dir, "common_issues.png")
        plt.savefig(file_path)
        print(f"Saved common issues analysis to {file_path}")
    
    # Analyze specific issue columns for more detailed insights
    if len(a11y_cols) > 0:
        print(f"\nAnalyzing {len(a11y_cols)} specific accessibility issue columns")
        
        # Create visualizations for issue distributions
        create_issue_distribution_visualizations(df, a11y_cols)
    
    return issue_counts

def create_issue_distribution_visualizations(df, issue_columns):
    """
    Create visualizations showing the distribution of accessibility issues
    
    Args:
        df: DataFrame with accessibility data
        issue_columns: List of issue column names to visualize
    """
    # Visualize issue distributions
    plt.figure(figsize=(14, 8))
    
    # Prepare data for visualization
    issue_data = pd.DataFrame()
    
    for col in issue_columns:
        # Skip if not enough non-null values
        if df[col].notna().sum() < 3:
            continue
            
        # For severity-based columns, try to extract just the numeric score
        if df[col].dtype == 'object' and df[col].notna().any() and ',' in str(df[col].iloc[df[col].first_valid_index()]):
            # Try to convert from format like "0.72, Severity: Medium"
            try:
                values = df[col].dropna().apply(lambda x: float(str(x).split(',')[0]) if pd.notna(x) else np.nan)
                if not values.empty:
                    # Use a shortened column name
                    short_name = col.replace('Accessibility_', '').replace('A11y_Issue_', '')
                    short_name = short_name[:20] + '...' if len(short_name) > 20 else short_name
                    issue_data[short_name] = values
            except:
                pass
        elif pd.api.types.is_numeric_dtype(df[col]):
            # For already numeric columns
            short_name = col.replace('Accessibility_', '').replace('A11y_Issue_', '')
            short_name = short_name[:20] + '...' if len(short_name) > 20 else short_name
            issue_data[short_name] = df[col]
    
    # If we have enough data to plot
    if not issue_data.empty and issue_data.shape[1] > 0:
        # Create boxplot or violin plot depending on data volume
        if issue_data.shape[1] > 5:
            sns.boxplot(data=issue_data)
            plt.title("Distribution of Accessibility Issue Scores")
        else:
            # For fewer columns, violin plots give more detail
            sns.violinplot(data=issue_data)
            plt.title("Distribution of Accessibility Issue Scores (Violin Plot)")
            
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, "issue_distributions.png")
        plt.savefig(file_path)
        print(f"Saved issue distributions to {file_path}")
    
    # Create pass/fail distribution visualization
    if 'AccessibilityResult' in df.columns:
        plt.figure(figsize=(8, 8))
        pass_counts = df['AccessibilityResult'].value_counts()
        
        # Create pie chart
        plt.pie(pass_counts, labels=pass_counts.index, autopct='%1.1f%%', startangle=90,
                colors=['#ff9999','#66b3ff'])
        plt.title("Accessibility Pass/Fail Distribution")
        plt.axis('equal')
        
        file_path = os.path.join(output_dir, "pass_fail_distribution.png")
        plt.savefig(file_path)
        print(f"Saved pass/fail distribution to {file_path}")

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
    
    # Count sites with keyboard focus data
    has_keyboard_data = df[keyboard_focus_cols].notna().any(axis=1)
    sites_with_data = has_keyboard_data.sum()
    percent_with_data = sites_with_data / len(df) * 100
    
    print(f"\nSites with keyboard focus accessibility data: {sites_with_data} out of {len(df)} ({percent_with_data:.1f}%)")
    
    if sites_with_data == 0:
        print("No keyboard focus accessibility data available for analysis.")
        return None
    
    # Extract severity information if available
    interactive_col = next((col for col in keyboard_focus_cols if 'interactive' in col.lower() and 'focusable' in col.lower()), None)
    
    if interactive_col and df[interactive_col].notna().any():
        # Count by severity
        severity_data = df[df[interactive_col].notna()][interactive_col].apply(
            lambda x: x.split(',')[1].strip() if ',' in str(x) else 'Unknown Severity'
        ).value_counts()
        
        print("\nSeverity distribution for keyboard focusability issues:")
        for severity, count in severity_data.items():
            print(f"  - {severity}: {count} sites ({count/sites_with_data*100:.1f}% of tested sites)")
        
        # High severity issues deserve special attention
        high_severity = severity_data.get('Severity: High', 0)
        if high_severity > 0:
            print(f"\n⚠️ Warning: {high_severity} sites have high severity keyboard focus issues.")
            print("These issues severely impact users relying on keyboard navigation.")
    
    # Check WCAG compliance information if available
    wcag_col = next((col for col in keyboard_focus_cols if 'wcag' in col.lower()), None)
    
    if wcag_col and df[wcag_col].notna().any():
        wcag_criteria = df[df[wcag_col].notna()][wcag_col].value_counts()
        
        print("\nWCAG Success Criteria violated:")
        for criterion, count in wcag_criteria.items():
            print(f"  - {criterion}: {count} sites")
    
    # Calculate overall keyboard focus accessibility insights
    print("\nKey keyboard focus accessibility insights:")
    print(f"1. {percent_with_data:.1f}% of sites have keyboard focus accessibility data")
    
    if interactive_col and df[interactive_col].notna().any():
        print(f"2. All sites with keyboard focus data ({sites_with_data}) have keyboard focusability issues")
        
        if 'Severity: High' in severity_data:
            high_percent = severity_data['Severity: High'] / sites_with_data * 100
            print(f"3. {high_percent:.1f}% of tested sites have high severity keyboard focus issues")
        
        not_tested = len(df) - sites_with_data
        print(f"4. {not_tested} sites ({(not_tested/len(df))*100:.1f}%) were not tested for keyboard focus accessibility")
    
    return {
        'tested_sites': sites_with_data,
        'percent_tested': percent_with_data,
        'has_keyboard_issues': sites_with_data if interactive_col else 0
    }

def create_accessibility_dashboard(df):
    """Create a comprehensive dashboard with key accessibility insights"""
    print("\n===== CREATING ACCESSIBILITY DASHBOARD =====")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 14))
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
    else:
        ax_pass_fail.text(0.5, 0.5, 'No AccessibilityResult data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 3. Top Issues Bar Chart
    ax_issues = fig.add_subplot(gs[1, :])
    
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
            top_issues = sorted_issues[:10]  # Top 10 issues
            
            issues = [issue[:30] for issue, _ in top_issues]
            counts = [count for _, count in top_issues]
            
            bars = ax_issues.barh(issues[::-1], counts[::-1], color='coral')
            
            # Add count labels
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 1
                ax_issues.text(label_x_pos, bar.get_y() + bar.get_height()/2, str(int(width)),
                       va='center')
            
            ax_issues.set_title('Top 10 Most Common Accessibility Issues')
            ax_issues.set_xlabel('Number of Sites')
    else:
        ax_issues.text(0.5, 0.5, 'No accessibility issue data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 4. Accessibility Score vs Number of Issues Scatter Plot
    ax_correlation = fig.add_subplot(gs[2, 0])
    
    if 'Accessibility_Score' in df.columns and 'Accessibility_Issues_Count' in df.columns:
        issues_col = 'Accessibility_Issues_Count'
        
        # Create scatter plot
        sns.regplot(x='Accessibility_Score', y=issues_col, data=df.dropna(subset=['Accessibility_Score', issues_col]), 
                   ax=ax_correlation, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
        
        ax_correlation.set_title('Accessibility Score vs. Number of Issues')
        ax_correlation.set_xlabel('Accessibility Score')
        ax_correlation.set_ylabel('Number of Issues')
    else:
        ax_correlation.text(0.5, 0.5, 'No accessibility score or issue count data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 5. Keyboard Focus Issues
    ax_keyboard = fig.add_subplot(gs[2, 1])
    
    # Find keyboard focus columns
    keyboard_cols = [col for col in df.columns if 'keyboard' in col.lower() or 'focus' in col.lower()]
    
    if keyboard_cols:
        # Count sites with keyboard data
        has_keyboard_data = df[keyboard_cols].notna().any(axis=1)
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
        
        # Add text annotation about issues if available
        if sites_with_data > 0:
            # Try to find the interactive controls column
            interactive_col = next((col for col in keyboard_cols if 'interactive' in col.lower() and 'focusable' in col.lower()), None)
            
            if interactive_col and df[interactive_col].notna().any():
                # Count by severity
                def get_severity(val):
                    if pd.isna(val):
                        return np.nan
                    val_str = str(val)
                    if 'high' in val_str.lower():
                        return 'High'
                    elif 'medium' in val_str.lower():
                        return 'Medium'
                    elif 'low' in val_str.lower():
                        return 'Low'
                    else:
                        return 'Unknown'
                
                # Add text annotation
                high_count = df[interactive_col].apply(get_severity).value_counts().get('High', 0)
                ax_keyboard.annotate(f"⚠️ {high_count} sites have high severity\nkeyboard focus issues",
                             xy=(0, 0), xytext=(0, 0), textcoords='offset points',
                             bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
                             ha='center', va='center', fontsize=10)
    else:
        ax_keyboard.text(0.5, 0.5, 'No keyboard focus data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Web Accessibility Dashboard - Swiss Healthcare Websites', fontsize=16, y=0.98)
    
    # Save the dashboard
    file_path = os.path.join(output_dir, "accessibility_dashboard.png")
    plt.savefig(file_path)
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
    
    # Insight 6: Most common issue
    if issue_counts and len(issue_counts) > 0:
        # Convert to DataFrame for consistent handling
        issue_df = pd.DataFrame({'Issue': list(issue_counts.keys()), 'Count': list(issue_counts.values())})
        issue_df = issue_df.sort_values('Count', ascending=False)
        
        if not issue_df.empty:
            top_issue = issue_df.iloc[0]['Issue']
            top_issue_count = issue_df.iloc[0]['Count']
            insights.append(f"6. The most common accessibility issue is '{top_issue}', found on {top_issue_count} sites.")
    
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

def create_comprehensive_insights_summary(df, score_stats, domain_scores, issue_counts, pass_rates=None):
    """Create a comprehensive summary of insights and recommendations"""
    timestamp = datetime.now().strftime("%B %d, %Y")
    
    # Convert issue_counts to DataFrame for consistent handling
    if issue_counts and len(issue_counts) > 0:
        issue_df = pd.DataFrame({'Issue': list(issue_counts.keys()), 'Count': list(issue_counts.values())})
        issue_df = issue_df.sort_values('Count', ascending=False)
    else:
        issue_df = pd.DataFrame()
    
    # Create the content
    summary = f"""# SWISS HEALTHCARE WEBSITE ACCESSIBILITY INSIGHTS
==================================================

## OVERVIEW

This document presents a comprehensive analysis of web accessibility for Swiss healthcare websites based on Google Lighthouse audits. The analysis included {len(df)} websites after excluding websites that were PDFs or non-HTML content that couldn't be properly analyzed by the Lighthouse tool.

## KEY ACCESSIBILITY FINDINGS

1. **Accessibility Compliance**: Only {(df['Accessibility_Score'] >= 90).mean() * 100:.1f}% of healthcare websites pass accessibility standards (score ≥90/100).

2. **Average Performance**:
   - Average accessibility score: {score_stats.loc['mean', 'Accessibility_Score']:.1f}/100
   - Average number of accessibility issues: {df['Accessibility_Issues_Count'].mean() if 'Accessibility_Issues_Count' in df.columns else 'N/A'} per site

3. **Domain Performance Discrepancy**:
   - {score_stats.loc['max', 'Accessibility_Score'] - score_stats.loc['min', 'Accessibility_Score']:.1f}-point gap between best and worst domains
   - {(df['Accessibility_Issues_Count'] == 0).sum() if 'Accessibility_Issues_Count' in df.columns else 0} websites were found to have zero accessibility issues

4. **Top Accessibility Issues**:"""

    # Add top issues if available
    if not issue_df.empty:
        top_issues = min(6, len(issue_df))
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

5. **Keyboard Accessibility**:
   - Only {testing_rate:.1f}% of sites were tested for keyboard focus accessibility
   - {100.0 - (df[df[keyboard_cols].notna().any(axis=1)][keyboard_cols].isna().all(axis=1).mean() * 100) if sites_with_data > 0 else 0.0:.1f}% pass rate among tested sites for keyboard focus criteria
   - Most keyboard focus issues were not tested, representing a significant gap"""
    
    # Add performance metrics
    if pass_rates:
        summary += f"""

6. **Performance Metrics**:
   - Performance: Only {pass_rates.get('Performance_Score', 0):.1f}% pass rate
   - Best Practices: {pass_rates.get('Best_Practices_Score', 0):.1f}% pass rate
   - SEO: {pass_rates.get('SEO_Score', 0):.1f}% pass rate"""

    # Add recommendations
    summary += """

## RECOMMENDATIONS

1. **Critical Improvements**:
   - Fix performance issues like Time to Interactive and First Contentful Paint
   - Add explicit image dimensions
   - Implement efficient caching policies
   - Optimize JavaScript loading and execution

2. **Accessibility Standards**:
   - Adopt and adhere to WCAG 2.1 AA standards
   - Implement accessibility testing in the development lifecycle
   - Provide accessibility training for development teams

3. **Targeted Testing**:
   - Increase testing coverage for keyboard focus accessibility
   - Ensure all interactive elements are keyboard accessible
   - Test with assistive technologies

4. **Monitoring & Maintenance**:
   - Implement regular accessibility audits
   - Set benchmarks for improvement over time
   - Include user testing with people with disabilities

## METHODOLOGY NOTE

Websites that were PDFs or non-HTML content were excluded from the analysis because they couldn't be properly analyzed by the Lighthouse tool. The final analysis was conducted on the remaining websites.

## CONCLUSION

The accessibility of Swiss healthcare websites requires significant improvement. With less than half meeting basic accessibility standards, there is a clear need for healthcare providers to address these issues. Improving accessibility ensures that all users, including those with disabilities, have equal access to vital healthcare information and services.

==================================================
Report Date: {timestamp}"""

    # Save to file
    summary_file = "insights_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    # Also save a copy to the output directory
    output_file = os.path.join(output_dir, "insights_summary.txt")
    with open(output_file, 'w') as f:
        f.write(summary)
    
    print(f"Saved comprehensive insights summary to {summary_file} and {output_file}")
    
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