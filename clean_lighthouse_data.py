import pandas as pd
import numpy as np
import os
from datetime import datetime

def clean_lighthouse_data(input_file='lighthouse_results_detailed_20250321_135559.xlsx', threshold_empty=0.8):
    """
    Clean lighthouse data by removing unnecessary rows and columns with too many empty values
    while preserving the most important information.
    
    Args:
        input_file: Path to the detailed lighthouse results file
        threshold_empty: Threshold for removing columns with empty values (0-1)
    """
    print(f"Reading data from {input_file}...")
    
    # Handle different file extensions
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        df = pd.read_excel(input_file)
    
    print(f"Original data shape: {df.shape}")
    
    # Step 1: Identify keyboard and focus related columns to preserve
    keyboard_focus_cols = [col for col in df.columns if 'keyboard' in str(col).lower() or 'focus' in str(col).lower()]
    print(f"Found {len(keyboard_focus_cols)} keyboard/focus related columns to preserve:")
    for col in keyboard_focus_cols:
        print(f"  - {col}")
    
    # Step 2: Remove columns with too many empty values, but preserve keyboard/focus columns
    empty_ratios = df.isna().mean()
    
    # Columns to keep: either has low percentage of missing values OR is a keyboard/focus column
    cols_to_keep = [col for col in df.columns if (empty_ratios[col] < threshold_empty) or (col in keyboard_focus_cols)]
    
    df_cleaned = df[cols_to_keep].copy()
    
    print(f"After removing sparse columns: {df_cleaned.shape}")
    print(f"Removed {df.shape[1] - df_cleaned.shape[1]} columns with over {threshold_empty*100}% empty values")
    
    # Step 3: Remove duplicate rows based on URL if present
    if 'URL' in df_cleaned.columns:
        before_dedup = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates(subset=['URL'], keep='first')
        print(f"Removed {before_dedup - len(df_cleaned)} duplicate URLs")
    
    # Step 4: Clean score columns and ensure proper format
    # Check for typical score column names in both formats
    possible_score_columns = [
        # Original format
        'Performance', 'Accessibility', 'Best Practices', 'SEO',
        # Alternative format
        'performance', 'accessibility', 'best-practices', 'seo',
        'Performance Score', 'Accessibility Score', 'Best Practices Score', 'SEO Score'
    ]
    
    score_columns = [col for col in possible_score_columns if col in df_cleaned.columns]
    
    for col in score_columns:
        if col in df_cleaned.columns:
            # Convert scores to numeric, replacing 'N/A' with NaN
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            # Cap values at 100
            df_cleaned[col] = df_cleaned[col].apply(lambda x: min(x, 100) if pd.notna(x) else x)
    
    # Step 5: Create accessibility-specific columns if not present
    accessibility_col = next((col for col in score_columns if 'accessibility' in col.lower()), None)
    
    if accessibility_col:
        df_cleaned['AccessibilityResult'] = df_cleaned[accessibility_col].apply(lambda x: 'Pass' if x >= 90 else 'Fail')
    
    # Step 6: Extract domain names from URLs for easier analysis
    if 'URL' in df_cleaned.columns and 'Domain' not in df_cleaned.columns:
        df_cleaned['Domain'] = df_cleaned['URL'].apply(
            lambda url: url.split('//')[1].split('/')[0] if pd.notna(url) and '//' in url else 
                        (url.split('/')[0] if pd.notna(url) else np.nan)
        )
    
    # Step 7: Restructure accessibility issues for better analysis
    # Check for various issue column formats
    possible_issue_columns = [
        'Accessibility Issues', 'accessibility-issues', 'A11y Issues',
        'Issues', 'Failures', 'Errors'
    ]
    
    issue_column = next((col for col in possible_issue_columns if col in df_cleaned.columns), None)
    issue_text_present = False
    
    # Check if an issue column exists and has content
    if issue_column and df_cleaned[issue_column].notna().any():
        issue_text_present = True
        # Rename to standardized name
        df_cleaned.rename(columns={issue_column: 'Accessibility Issues'}, inplace=True)
    else:
        # Look for columns that might represent specific accessibility issues
        issue_columns = []
        for col in df_cleaned.columns:
            if col not in ['Rank', 'Title', 'URL', 'Performance', 'Accessibility', 'Best Practices', 
                          'SEO', 'Domain', 'AccessibilityResult'] + keyboard_focus_cols + score_columns:
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
    
    # Step 8: Add summary columns for issue counts
    if issue_text_present and 'Accessibility Issues' in df_cleaned.columns:
        df_cleaned['Total_Issues'] = df_cleaned['Accessibility Issues'].apply(
            lambda x: len(str(x).split(';')) if pd.notna(x) and x != '' else 0
        )
    
    # Step 9: Select and rename only the most important columns
    # Identify standard column names
    # Find which of the possible column formats exists in the data
    standard_columns = {
        'Rank': next((col for col in ['Rank', 'rank', '#', 'Position'] if col in df_cleaned.columns), None),
        'Title': next((col for col in ['Title', 'title', 'Website', 'Page Title', 'Name'] if col in df_cleaned.columns), None),
        'URL': next((col for col in ['URL', 'url', 'Website URL', 'Link'] if col in df_cleaned.columns), None),
        'Performance': next((col for col in ['Performance', 'performance', 'Performance Score'] if col in df_cleaned.columns), None),
        'Accessibility': next((col for col in ['Accessibility', 'accessibility', 'Accessibility Score'] if col in df_cleaned.columns), None),
        'Best Practices': next((col for col in ['Best Practices', 'best-practices', 'Best Practices Score'] if col in df_cleaned.columns), None),
        'SEO': next((col for col in ['SEO', 'seo', 'SEO Score'] if col in df_cleaned.columns), None),
    }
    
    # Create a mapping for renaming
    rename_mapping = {v: k for k, v in standard_columns.items() if v is not None}
    
    # Apply renaming if needed
    if rename_mapping:
        df_cleaned.rename(columns=rename_mapping, inplace=True)
    
    # Define core columns to keep
    core_columns = [
        'Rank', 'Title', 'URL', 'Domain', 
        'Performance', 'Accessibility', 'Best Practices', 'SEO',
        'AccessibilityResult', 'Total_Issues'
    ]
    
    # Filter to only include columns that exist
    core_columns = [col for col in core_columns if col in df_cleaned.columns]
    
    # Add accessibility issues column if it exists
    if 'Accessibility Issues' in df_cleaned.columns:
        core_columns.append('Accessibility Issues')
    
    # Add keyboard focus columns
    core_columns.extend(keyboard_focus_cols)
    
    # Add up to 10 most common accessibility issue columns that aren't already in core columns
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
    
    # Step 10: Create readable column names and rename non-descriptive ones
    columns_mapping = {
        'Performance': 'Performance_Score',
        'Accessibility': 'Accessibility_Score',
        'Best Practices': 'Best_Practices_Score',
        'SEO': 'SEO_Score',
        'Accessibility Issues': 'Accessibility_Issues_Details',
        'Total_Issues': 'Accessibility_Issues_Count'
    }
    
    # Add mappings for keyboard focus columns
    for col in keyboard_focus_cols:
        if col in df_final.columns:
            clean_name = str(col).replace('.', '')
            # Convert the keyboard focus column names to be more consistent
            clean_name = f"Accessibility_Keyboard_Focus_{clean_name.replace(' ', '_').replace('-', '_')}"
            columns_mapping[col] = clean_name
    
    # Add mappings for accessibility issue columns
    for col in top_issue_cols:
        if col in df_final.columns and col not in keyboard_focus_cols:
            # Clean up column name to be more readable
            clean_name = str(col).replace('.', '')
            # Convert to title case and add prefix if needed
            if 'Accessibility' not in clean_name and 'a11y' not in clean_name.lower():
                clean_name = f"A11y_Issue_{clean_name}"
            clean_name = clean_name.replace(' ', '_').replace('-', '_')
            columns_mapping[col] = clean_name
    
    # Apply column renaming
    df_final = df_final.rename(columns=columns_mapping)
    
    # Step 11: Remove columns with all zeros or insignificant variance, but KEEP keyboard focus columns
    keyboard_focus_renamed = [columns_mapping.get(col, col) for col in keyboard_focus_cols if col in df_final.columns]
    
    # Print keyboard focus columns after renaming for tracking
    print("\nKeyboard focus columns after renaming:")
    for col in keyboard_focus_renamed:
        print(f"  - {col}")
    
    # Mark columns to drop, EXCLUDING keyboard focus columns
    cols_to_drop = []
    
    for col in df_final.columns:
        # Skip non-numeric columns, essential metadata columns, and keyboard/focus columns
        essential_cols = ['Rank', 'Title', 'URL', 'Domain', 'AccessibilityResult'] + keyboard_focus_renamed
        if col in essential_cols or col in keyboard_focus_renamed or not pd.api.types.is_numeric_dtype(df_final[col]):
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
    
    # Print identified columns to drop but clearly indicate keyboard focus columns will be preserved
    print("\nIdentified columns with insignificant data:")
    for col, reason in cols_to_drop:
        print(f"  - {col}: {reason}")
    
    # Print keyboard focus column statistics
    print("\nKeyboard focus column statistics (these columns will be preserved regardless of variance):")
    for col in keyboard_focus_cols:
        if col in df_cleaned.columns:
            # Get column stats
            values_count = df_cleaned[col].value_counts(dropna=False)
            print(f"  - {col}:")
            print(f"    Values: {values_count.to_dict()}")
    
    # Drop identified columns, carefully excluding keyboard focus columns
    cols_to_actually_drop = [col for col, _ in cols_to_drop if col not in keyboard_focus_renamed]
    
    if cols_to_actually_drop:
        print(f"\nRemoving {len(cols_to_actually_drop)} columns with insignificant data (preserving keyboard focus columns):")
        for col in cols_to_actually_drop:
            print(f"  - {col}")
        
        df_final = df_final.drop(cols_to_actually_drop, axis=1)
    
    # Make sure keyboard focus columns are kept even after the drop operation
    for col in keyboard_focus_cols:
        # Get the renamed column name
        renamed_col = columns_mapping.get(col, col)
        # If the column was dropped despite our checks, put it back
        if renamed_col not in df_final.columns and col in df_cleaned.columns:
            print(f"Restoring accidentally dropped column: {renamed_col}")
            df_final[renamed_col] = df_cleaned[col]
    
    # Generate output filename
    output_file = 'lighthouse_scores_optimized.xlsx'
    
    # Save the cleaned data
    print(f"Saving optimized data to {output_file}...")
    df_final.to_excel(output_file, index=False)
    
    # Create a second summary file with just the most critical information
    summary_cols = ['Rank', 'Title', 'URL', 'Domain', 
                   'Performance_Score', 'Accessibility_Score', 'Best_Practices_Score', 'SEO_Score',
                   'AccessibilityResult', 'Accessibility_Issues_Count']
    
    # Add keyboard focus columns to summary
    summary_cols.extend(keyboard_focus_renamed)
    
    summary_cols = [col for col in summary_cols if col in df_final.columns]
    
    df_summary = df_final[summary_cols].copy()
    
    # Also check the summary file for zero or near-zero columns, but preserve keyboard focus columns
    summary_cols_to_drop = []
    for col in df_summary.columns:
        # Skip non-numeric columns, essential metadata columns, and keyboard focus columns
        essential_cols = ['Rank', 'Title', 'URL', 'Domain', 'AccessibilityResult'] + keyboard_focus_renamed
        if col in essential_cols or not pd.api.types.is_numeric_dtype(df_summary[col]) or col in keyboard_focus_renamed:
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
    
    summary_file = 'lighthouse_scores_summary.xlsx'
    df_summary.to_excel(summary_file, index=False)
    
    # Print summary
    print("\nCleaning Summary:")
    print(f"Original dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Intermediate cleaned dimensions: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
    print(f"Final optimized dimensions: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
    print(f"Summary dimensions: {df_summary.shape[0]} rows × {df_summary.shape[1]} columns")
    
    print(f"\nFinal columns: {', '.join(df_final.columns)}")
    
    # Count keyboard focus columns retained
    keyboard_focus_cols_final = [col for col in df_final.columns if 'keyboard' in col.lower() or 'focus' in col.lower()]
    print(f"\nKeyboard focus columns preserved: {len(keyboard_focus_cols_final)}")
    for col in keyboard_focus_cols_final:
        print(f"  - {col}")
    
    if 'Accessibility_Score' in df_final.columns:
        pass_rate = (df_final['Accessibility_Score'] >= 90).mean() * 100
        print(f"Accessibility pass rate (≥90): {pass_rate:.1f}%")
    
    return df_final, output_file, summary_file

if __name__ == "__main__":
    # Try to find the detailed results file based on various patterns
    data_files = []
    
    for file in os.listdir('.'):
        if file.endswith('.xlsx') or file.endswith('.csv'):
            if 'detailed' in file.lower() or 'result' in file.lower():
                data_files.append((file, os.path.getmtime(file)))
    
    # Also look in original_files_backup
    if os.path.exists('original_files_backup'):
        for file in os.listdir('original_files_backup'):
            if file.endswith('.xlsx') or file.endswith('.csv'):
                if 'detailed' in file.lower() or 'result' in file.lower():
                    full_path = os.path.join('original_files_backup', file)
                    data_files.append((full_path, os.path.getmtime(full_path)))
    
    if data_files:
        # Sort by modification time (newest first)
        data_files.sort(key=lambda x: x[1], reverse=True)
        input_file = data_files[0][0]
        print(f"Found data file: {input_file}")
    else:
        # Use default if no file found
        input_file = 'original_files_backup/lighthouse_results_detailed_20250321_135559.xlsx'
        print(f"Using default file: {input_file}")
    
    # Process the data
    cleaned_df, output_file, summary_file = clean_lighthouse_data(input_file)
    print(f"\nData cleaning completed! Files created:")
    print(f"1. {output_file} - Optimized data with specific column names")
    print(f"2. {summary_file} - Summary with just the most essential information") 