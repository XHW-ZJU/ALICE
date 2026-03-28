import os
import pandas as pd
import numpy as np


def setup_pandas_display():
    """Configure pandas display settings."""
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)     # Show all rows
    pd.set_option('max_colwidth', 200)          # Set maximum column width to 200


def normalize_column(df, column_name):
    """Normalize a column to range [0, 1]."""
    min_val = df[column_name].min()
    max_val = df[column_name].max()
    return (df[column_name] - min_val) / (max_val - min_val)


def calculate_ea_score(row):
    """Calculate the EA Score for a row."""
    return (3.5 * np.sin(row['Target_LY6C1_Enri_norm'] * 1.85) +
            3.5 * np.sin(row['Target_LY6A_Enri_norm'] * 2.7) +
            row['Prod_Fit_norm'] * 0.1 +
            row['FE_score_norm'] * 0.15)


def process_data(input_file, output_file):
    """Main function to process the data."""
    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Normalize columns
    for column in ['Target_LY6A_Enri', 'Target_LY6C1_Enri', 'Prod_Fit', 'FE_score']:
        df[f'{column}_norm'] = normalize_column(df, column)

    # Calculate EA Score
    df['EA_Score'] = df.apply(calculate_ea_score, axis=1)

    # Sort by EA Score and reset index
    sorted_df = df.sort_values(by='EA_Score', ascending=False).reset_index(drop=True)
    sorted_df = sorted_df[sorted_df['EA_Score'] >= 6.7]
    # Save to CSV
    sorted_df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")


def main(in_path: str, out_path: str):
    # Setup pandas display
    setup_pandas_display()

    # Define input and output paths
    output_file = os.path.join(out_path, 'EA_Top.csv')

    # Create output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)

    # Process the data
    process_data(in_path, output_file)


if __name__ == "__main__":
    main('../output/EA/Prod_Fit_Top100.csv', '../output/EA/')