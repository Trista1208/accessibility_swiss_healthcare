import pandas as pd
import re
from datetime import datetime

def extract_lighthouse_scores(xlsx_file):
    df = pd.read_excel(xlsx_file)
    columns = df.columns
    
    # Create a new DataFrame with the first 7 columns from the original DataFrame
    result_df = df.iloc[:, :7].copy()
    
    # Dictionary to store unique titles and their scores
    title_scores = {}
    
    # Process data starting from the 8th column (index 7)
    for i in range(7, len(columns)):
        for row_idx, value in enumerate(df.iloc[:, i]):
            # Check if the value is a string and not NaN
            if isinstance(value, str) and value:
                # Extract the title (text before brackets) and score
                match = re.search(r'(.*?)\s*\(Score: (.*?)\)', value)
                if match:
                    title = match.group(1).strip()  # Text before brackets, stripped of whitespace
                    score = match.group(2)  # Score value
                    
                    # Initialize the scores list for this title if it doesn't exist
                    if title not in title_scores:
                        title_scores[title] = [None] * len(df)
                    
                    # Store the score for this row
                    title_scores[title][row_idx] = score
    
    # Add the extracted scores to the result DataFrame
    for title, scores in title_scores.items():
        result_df[title] = scores
    
    return result_df



def clean_dataframe(df, threshold=0.3):
    """
    Thoroughly clean DataFrame by removing columns with:
    1. All zero values
    2. Majority (>= 30%) None/NaN/empty values
    3. Columns with no meaningful data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame to be cleaned
    threshold : float, optional (default=0.3)
        Threshold for percentage of invalid values to drop a column
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned DataFrame with problematic columns removed
    """
    # Create a copy of the DataFrame to avoid modifying the original
    cleaned_df = df.copy()
    
    # Comprehensive empty/invalid value check
    def is_invalid(value):
        # Check for None, NaN, zero, empty string, and whitespace-only string
        if value is None:
            return True
        
        # Check for numpy NaN
        if pd.isna(value):
            return True
        
        # Check for zero (including float and int zero)
        if isinstance(value, (int, float)) and value == 0:
            return True
        
        # Check for empty or whitespace-only string
        if isinstance(value, str) and value.strip() == '':
            return True
        
        return False
    
    # Identify columns to drop
    columns_to_drop = []
    
    for col in cleaned_df.columns:
        # Convert the column to a series for easier manipulation
        series = cleaned_df[col]
        
        # Calculate the percentage of invalid values
        invalid_percentage = series.apply(is_invalid).mean()
        
        # Check if ALL values are zero
        all_zero = (series.apply(lambda x: x == 0 if isinstance(x, (int, float)) else False)).all()
        
        # Add column to drop list if:
        # 1. Over threshold of invalid values, or
        # 2. All values are zero
        if invalid_percentage >= threshold or all_zero:
            columns_to_drop.append(col)
    
    # Remove identified columns
    cleaned_df.drop(columns=columns_to_drop, inplace=True)
    
    return cleaned_df


def remove_bad_columns(df, threshold=0.3):
    """
    Remove columns where more than a specified threshold of values are 0, 'None', empty strings, or NaN/None.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    threshold (float): Proportion threshold (default 0.5) for column removal
    
    Returns:
    pd.DataFrame: DataFrame with qualifying columns removed
    """
    df_clean = df.copy()
    cols_to_drop = []
    
    for col in df_clean.columns:
        col_series = df_clean[col]
        # Create boolean mask for bad values
        bad_mask = (
            (col_series == 0) | 
            (col_series == '0')|
            (col_series == 'None') | 
            (col_series == '') | 
            col_series.isna()
        )
        bad_count = bad_mask.sum()
        
        # Check if bad values exceed threshold
        if (bad_count / len(col_series)) > threshold:
            cols_to_drop.append(col)
    
    return df_clean.drop(columns=cols_to_drop)
if __name__ == "__main__":
    input_file = "lighthouse_results - organiszed.xlsx"
    
    result_df = extract_lighthouse_scores(input_file)
    clean_df = remove_bad_columns(result_df)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS
    filename = f"lighthouse_scores_extracted_{current_time}.xlsx"

    clean_df.to_excel(filename, index=False)
        
