import pandas as pd
import numpy as np
import os
from datetime import datetime

def clean_lighthouse_data(input_file='lighthouse_scores_extracted.xlsx', threshold_empty=0.8):
    """
    Clean lighthouse data by removing unnecessary rows and columns with too many empty values
    while preserving the most important information.
    
    Args:
        input_file: Path to the lighthouse_scores_extracted.xlsx file
        threshold_empty: Threshold for removing columns with empty values (0-1)
    """
    print(f"Reading data from {input_file}...")
    df = pd.read_excel(input_file)
    
    print(f"Original data shape: {df.shape}")
    
    # Step 1: Remove columns with too many empty values
    empty_ratios = df.isna().mean()
    cols_to_keep = empty_ratios[empty_ratios < threshold_empty].index.tolist()
    df_cleaned = df[cols_to_keep].copy()
    
    print(f"After removing sparse columns: {df_cleaned.shape}")
    print(f"Removed {df.shape[1] - df_cleaned.shape[1]} columns with over {threshold_empty*100}% empty values")
    
    # Step 2: Remove duplicate rows based on URL if present
    if 'URL' in df_cleaned.columns:
        before_dedup = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates(subset=['URL'], keep='first')
        print(f"Removed {before_dedup - len(df_cleaned)} duplicate URLs")
    
    # Step 3: Clean score columns and ensure proper format
    score_columns = [col for col in df_cleaned.columns if col in ['Performance', 'Accessibility', 'Best Practices', 'SEO']]
    for col in score_columns:
        if col in df_cleaned.columns:
            # Convert scores to numeric, replacing 'N/A' with NaN
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            # Cap values at 100
            df_cleaned[col] = df_cleaned[col].apply(lambda x: min(x, 100) if pd.notna(x) else x)
    
    # Step 4: Create accessibility-specific columns if not present
    if 'Accessibility' in df_cleaned.columns:
        df_cleaned['AccessibilityResult'] = df_cleaned['Accessibility'].apply(lambda x: 'Pass' if x >= 90 else 'Fail')
    
    # Step 5: Extract domain names from URLs for easier analysis
    if 'URL' in df_cleaned.columns and 'Domain' not in df_cleaned.columns:
        df_cleaned['Domain'] = df_cleaned['URL'].apply(
            lambda url: url.split('//')[1].split('/')[0] if pd.notna(url) and '//' in url else 
                        (url.split('/')[0] if pd.notna(url) else np.nan)
        )
    
    # Step 6: Restructure accessibility issues for better analysis
    # Identify issue columns - these are columns beyond the basic metadata that contain scores or issue information
    issue_columns = []
    issue_text_present = False
    
    # Check if 'Accessibility Issues' column exists and has content
    if 'Accessibility Issues' in df_cleaned.columns and df_cleaned['Accessibility Issues'].notna().any():
        issue_text_present = True
    else:
        # Look for columns that might represent specific accessibility issues
        for col in df_cleaned.columns:
            if col not in ['Rank', 'Title', 'URL', 'Performance', 'Accessibility', 'Best Practices', 
                           'SEO', 'Domain', 'AccessibilityResult']:
                if df_cleaned[col].notna().any():
                    issue_columns.append(col)
        
        # Create the combined issues column if we found issue columns
        if issue_columns:
            df_cleaned['Accessibility Issues'] = ""
            
            # Combine all issues, properly formatting each one
            for idx, row in df_cleaned.iterrows():
                issues = []
                for col in issue_columns:
                    if pd.notna(row[col]) and row[col]:
                        # Format as "Issue Name (Score: value)"
                        issues.append(f"{col} (Score: {row[col]})")
                
                if issues:
                    df_cleaned.at[idx, 'Accessibility Issues'] = "; ".join(issues)
            
            issue_text_present = True
    
    # Step 7: Add summary columns for issue counts
    if issue_text_present:
        df_cleaned['Total_Issues'] = df_cleaned['Accessibility Issues'].apply(
            lambda x: len(str(x).split(';')) if pd.notna(x) and x != '' else 0
        )
    
    # Step 8: Select and rename only the most important columns
    # Define core columns to keep
    core_columns = [
        'Rank', 'Title', 'URL', 'Domain', 
        'Performance', 'Accessibility', 'Best Practices', 'SEO',
        'AccessibilityResult', 'Total_Issues'
    ]
    
    # Add accessibility issues column if it exists
    if 'Accessibility Issues' in df_cleaned.columns:
        core_columns.append('Accessibility Issues')
    
    # Add up to 10 most common accessibility issue columns
    accessibility_issue_cols = []
    for col in df_cleaned.columns:
        if col not in core_columns and df_cleaned[col].notna().mean() > 0.1:  # Column has values in at least 10% of rows
            accessibility_issue_cols.append((col, df_cleaned[col].notna().mean()))
    
    # Sort by frequency (most common first) and take top 10
    accessibility_issue_cols.sort(key=lambda x: x[1], reverse=True)
    top_issue_cols = [col[0] for col in accessibility_issue_cols[:10]]
    
    # Final column selection
    columns_to_keep = core_columns + top_issue_cols
    columns_to_keep = [col for col in columns_to_keep if col in df_cleaned.columns]
    
    # Keep only selected columns
    df_final = df_cleaned[columns_to_keep].copy()
    
    # Step 9: Create readable column names and rename non-descriptive ones
    columns_mapping = {
        'Performance': 'Performance_Score',
        'Accessibility': 'Accessibility_Score',
        'Best Practices': 'Best_Practices_Score',
        'SEO': 'SEO_Score',
        'Accessibility Issues': 'Accessibility_Issues_Details',
        'Total_Issues': 'Accessibility_Issues_Count'
    }
    
    # Add mappings for accessibility issue columns
    for col in top_issue_cols:
        if col in df_final.columns:
            # Clean up column name to be more readable
            clean_name = col.replace('.', '')
            # Convert to title case and add prefix if needed
            if 'Accessibility' not in clean_name and 'a11y' not in clean_name.lower():
                clean_name = f"A11y_Issue_{clean_name}"
            clean_name = clean_name.replace(' ', '_').replace('-', '_')
            columns_mapping[col] = clean_name
    
    # Apply column renaming
    df_final = df_final.rename(columns=columns_mapping)
    
    # Step 10: Remove columns with all zeros or insignificant variance
    cols_to_drop = []
    
    for col in df_final.columns:
        # Skip non-numeric columns and essential metadata columns
        essential_cols = ['Rank', 'Title', 'URL', 'Domain', 'AccessibilityResult']
        if col in essential_cols or not pd.api.types.is_numeric_dtype(df_final[col]):
            continue
            
        # Check for all zeros
        if (df_final[col] == 0).all():
            cols_to_drop.append((col, "all zeros"))
            continue
            
        # Check for >95% zeros
        zero_ratio = (df_final[col] == 0).mean()
        if zero_ratio > 0.95:
            cols_to_drop.append((col, f"{zero_ratio*100:.1f}% zeros"))
            continue
            
        # Check for low variance (all values nearly the same)
        if df_final[col].nunique() <= 1:
            cols_to_drop.append((col, "no variance"))
            continue
            
        # Check for near-constant values
        std = df_final[col].std()
        mean = df_final[col].mean()
        if mean > 0 and std/mean < 0.01:  # Coefficient of variation < 1%
            cols_to_drop.append((col, "minimal variance"))
    
    # Drop identified columns
    if cols_to_drop:
        print("\nRemoving columns with insignificant data:")
        for col, reason in cols_to_drop:
            print(f"  - {col}: {reason}")
        
        df_final = df_final.drop([col for col, _ in cols_to_drop], axis=1)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'lighthouse_scores_optimized_{timestamp}.xlsx'
    
    # Save the cleaned data
    print(f"Saving optimized data to {output_file}...")
    df_final.to_excel(output_file, index=False)
    
    # Create a second summary file with just the most critical information
    summary_cols = ['Rank', 'Title', 'URL', 'Domain', 
                   'Performance_Score', 'Accessibility_Score', 'Best_Practices_Score', 'SEO_Score',
                   'AccessibilityResult', 'Accessibility_Issues_Count']
    summary_cols = [col for col in summary_cols if col in df_final.columns]
    
    df_summary = df_final[summary_cols].copy()
    
    # Also check the summary file for zero or near-zero columns
    summary_cols_to_drop = []
    for col in df_summary.columns:
        # Skip non-numeric columns and essential metadata columns
        essential_cols = ['Rank', 'Title', 'URL', 'Domain', 'AccessibilityResult']
        if col in essential_cols or not pd.api.types.is_numeric_dtype(df_summary[col]):
            continue
            
        # Check for all zeros
        if (df_summary[col] == 0).all():
            summary_cols_to_drop.append((col, "all zeros"))
            continue
            
        # Check for >95% zeros
        zero_ratio = (df_summary[col] == 0).mean()
        if zero_ratio > 0.95:
            summary_cols_to_drop.append((col, f"{zero_ratio*100:.1f}% zeros"))
    
    # Drop identified columns from summary
    if summary_cols_to_drop:
        print("\nRemoving columns with insignificant data from summary file:")
        for col, reason in summary_cols_to_drop:
            print(f"  - {col}: {reason}")
        
        df_summary = df_summary.drop([col for col, _ in summary_cols_to_drop], axis=1)
    
    summary_file = f'lighthouse_scores_summary_{timestamp}.xlsx'
    df_summary.to_excel(summary_file, index=False)
    
    # Print summary
    print("\nCleaning Summary:")
    print(f"Original dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Intermediate cleaned dimensions: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
    print(f"Final optimized dimensions: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
    print(f"Summary dimensions: {df_summary.shape[0]} rows × {df_summary.shape[1]} columns")
    
    print(f"\nFinal columns: {', '.join(df_final.columns)}")
    
    if 'Accessibility_Score' in df_final.columns:
        pass_rate = (df_final['Accessibility_Score'] >= 90).mean() * 100
        print(f"Accessibility pass rate (≥90): {pass_rate:.1f}%")
    
    return df_final, output_file, summary_file

if __name__ == "__main__":
    cleaned_df, output_file, summary_file = clean_lighthouse_data('lighthouse_scores_extracted.xlsx')
    print(f"\nData cleaning completed! Files created:")
    print(f"1. {output_file} - Optimized data with specific column names")
    print(f"2. {summary_file} - Summary with just the most essential information") 