import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

def clean_lighthouse_data(input_file='lighthouse_results_original.csv'):
    """
    Clean lighthouse data by:
    1. Loading the original CSV file
    2. Removing unnecessary columns
    3. Handling missing values appropriately
    4. Keeping original issue counts and accessibility scores
    
    Args:
        input_file: Path to the lighthouse results file (CSV)
    """
    print(f"Reading data from {input_file}...")
    
    # Load the data
    df = pd.read_csv(input_file)
    original_shape = df.shape
    print(f"Original data shape: {original_shape[0]} rows, {original_shape[1]} columns")
    
    # Define core columns to keep
    core_columns = [
        'Rank', 'Title', 'URL', 'Domain',
        'Performance', 'Accessibility', 'Best Practices', 'SEO',
        'AccessibilityResult', 'Total_Issues', 'High_Severity_Issues',
        'Accessibility Issues'  # Original detailed issues text
    ]
    
    # Add accessibility-specific issue columns
    accessibility_columns = [col for col in df.columns if any(term in col for term in [
        'do not have', 'not specified', 'lacks', 'not keyboard',
        'wcag', 'priority', 'details', 'aria'
    ])]
    
    # Combine all columns we want to keep
    columns_to_keep = list(set(core_columns + accessibility_columns))
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    
    # Keep only selected columns
    df_cleaned = df[columns_to_keep].copy()
    print(f"Kept {len(columns_to_keep)} relevant columns")
    
    # Handle missing values appropriately
    # 1. For numeric scores
    score_columns = ['Performance', 'Accessibility', 'Best Practices', 'SEO']
    for col in score_columns:
        if col in df_cleaned.columns:
            # Convert to numeric, keeping original values
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    
    # 2. For issue counts
    count_columns = ['Total_Issues', 'High_Severity_Issues']
    for col in count_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].fillna(0).astype(int)
    
    # 3. Ensure AccessibilityResult is based on actual score
    if 'Accessibility' in df_cleaned.columns:
        df_cleaned['AccessibilityResult'] = df_cleaned['Accessibility'].apply(
            lambda x: 'Pass' if pd.notna(x) and x >= 90 else 'Fail'
        )
    
    # Remove duplicate URLs if any exist
    if 'URL' in df_cleaned.columns:
        before_dedup = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates(subset=['URL'], keep='first')
        if before_dedup > len(df_cleaned):
            print(f"Removed {before_dedup - len(df_cleaned)} duplicate URLs")
    
    # Generate summary statistics
    print("\nData Cleaning Summary:")
    print(f"- Final shape: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")
    if 'Accessibility' in df_cleaned.columns:
        print(f"- Average accessibility score: {df_cleaned['Accessibility'].mean():.1f}")
        print(f"- Pass rate: {(df_cleaned['Accessibility'] >= 90).mean()*100:.1f}%")
    if 'Total_Issues' in df_cleaned.columns:
        print(f"- Average issues per site: {df_cleaned['Total_Issues'].mean():.1f}")
    if 'High_Severity_Issues' in df_cleaned.columns:
        print(f"- Average high severity issues: {df_cleaned['High_Severity_Issues'].mean():.1f}")
    
    # Save cleaned data
    output_file = 'lighthouse_data_cleaned.csv'
    df_cleaned.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to {output_file}")
    
    return df_cleaned

def process_accessibility_issues(df):
    """
    Process accessibility issues from 'Accessibility_Issues_Details' column and extract
    information about each issue into separate columns.
    
    Args:
        df: DataFrame containing the 'Accessibility_Issues_Details' column
        
    Returns:
        DataFrame with additional columns for each issue type
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Check if the required column exists
    if 'Accessibility_Issues_Details' not in result_df.columns:
        print("Column 'Accessibility_Issues_Details' not found in the DataFrame")
        return result_df
    
    # Dictionary to store all extracted issues and their scores/severity
    all_issues = {}
    
    # Process each row in the DataFrame
    for idx, row in result_df.iterrows():
        # Skip rows with NaN values in the Accessibility_Issues_Details column
        if pd.isna(row['Accessibility_Issues_Details']):
            continue
            
        # Split the text by semicolon to separate different issues
        issues = row['Accessibility_Issues_Details'].split(';')
        
        for issue in issues:
            issue = issue.strip()
            if not issue:
                continue
                
            # Use regex to extract the issue name, score, and severity
            match = re.match(r'(.*?)\s*\(Score:\s*([\d.]+),\s*Severity:\s*(\w+)\)', issue)
            if match:
                issue_name = match.group(1).strip()
                score = match.group(2)
                severity = match.group(3)
                
                # Create the combined value with score and severity
                value = f"{score};{severity}"
                
                # Create or update the dictionary entry for this issue
                if issue_name not in all_issues:
                    all_issues[issue_name] = {}
                
                all_issues[issue_name][idx] = value
    
    # Create a new column for each unique issue type
    for issue_name, values in all_issues.items():
        column_name = issue_name.replace(' ', '_').replace('`', '').replace('\'', '').replace('"', '')
        # Limit column name length to avoid overly long names
        if len(column_name) > 63:
            column_name = column_name[:60] + '...'
        result_df[column_name] = pd.Series(values)
    
    # Drop the original Accessibility_Issues_Details column after extracting the issues
    result_df = result_df.drop(columns=['Accessibility_Issues_Details'])
    print(f"Extracted {len(all_issues)} unique accessibility issues and dropped Accessibility_Issues_Details column")
    
    return result_df

if __name__ == "__main__":
    # Run the cleaning process
    clean_lighthouse_data()

# The following code is the standalone version that matches exactly what was requested
# It can be used directly if needed:

"""
def process_accessibility_issues(df):

    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Check if the required column exists
    if 'Accessibility_Issues_Details' not in result_df.columns:
        print("Column 'Accessibility_Issues_Details' not found in the DataFrame")
        return result_df
    
    # Dictionary to store all extracted issues and their scores/severity
    all_issues = {}
    
    # Process each row in the DataFrame
    for idx, row in result_df.iterrows():
        # Skip rows with NaN values in the Accessibility_Issues_Details column
        if pd.isna(row['Accessibility_Issues_Details']):
            continue
            
        # Split the text by semicolon to separate different issues
        issues = row['Accessibility_Issues_Details'].split(';')
        
        for issue in issues:
            issue = issue.strip()
            if not issue:
                continue
                
            # Use regex to extract the issue name, score, and severity
            match = re.match(r'(.*?)\s*\(Score:\s*([\d.]+),\s*Severity:\s*(\w+)\)', issue)
            if match:
                issue_name = match.group(1).strip()
                score = match.group(2)
                severity = match.group(3)
                
                # Create the combined value with score and severity
                value = f"{score};{severity}"
                
                # Create or update the dictionary entry for this issue
                if issue_name not in all_issues:
                    all_issues[issue_name] = {}
                
                all_issues[issue_name][idx] = value
    
    # Create a new column for each unique issue type
    for issue_name, values in all_issues.items():
        column_name = issue_name.replace(' ', '_').replace('`', '').replace('\'', '').replace('"', '')
        result_df[column_name] = pd.Series(values)
    
    return result_df

input_file = "lighthouse_scores_optimized.xlsx"
df = pd.read_excel(input_file)
result_df = process_accessibility_issues(df)
result_df.to_excel('output_file2.xlsx', index=False)
""" 
