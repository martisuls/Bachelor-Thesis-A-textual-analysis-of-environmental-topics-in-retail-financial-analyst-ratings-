import pandas as pd
import sys

def merge_csv_files(file1_path, file2_path, file3_path, output_path, merge_type='inner'):
    """
    Merge three CSV files on 'id' column and save to a new CSV file.

    Parameters:
    - file1_path: Path to the first CSV file
    - file2_path: Path to the second CSV file
    - file3_path: Path to the third CSV file
    - output_path: Path where the merged CSV will be saved
    - merge_type: Type of merge ('inner', 'outer', 'left', 'right')
                 Default is 'inner' (only matching IDs)
    """
    try:
        # Read the CSV files
        print(f"Reading {file1_path}...")
        df1 = pd.read_csv(file1_path)

        print(f"Reading {file2_path}...")
        df2 = pd.read_csv(file2_path)

        print(f"Reading {file3_path}...")
        df3 = pd.read_csv(file3_path)

        # Check if 'id' column exists in all dataframes
        if 'id' not in df1.columns:
            raise ValueError(f"'id' column not found in {file1_path}")
        if 'id' not in df2.columns:
            raise ValueError(f"'id' column not found in {file2_path}")
        if 'id' not in df3.columns:
            raise ValueError(f"'id' column not found in {file3_path}")

        # Merge the dataframes on 'id' column
        print(f"Merging first two files on 'id' column using {merge_type} join...")
        merged_df = pd.merge(df1, df2, on='id', how=merge_type)

        print(f"Merging with third file on 'id' column using {merge_type} join...")
        merged_df = pd.merge(merged_df, df3, on='id', how=merge_type)

        # Rename publishOn column to date if it exists
        if 'publishOn' in merged_df.columns:
            merged_df.rename(columns={'publishOn': 'date'}, inplace=True)
            print("Renamed 'publishOn' column to 'date'")

        # Save to output file
        merged_df.to_csv(output_path, index=False)
        print(f"Successfully merged! Output saved to: {output_path}")
        print(f"Total rows in merged file: {len(merged_df)}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Automatically run with specified files
    file1 = 'environmental_metrics_outputGPT.csv'
    file2 = 'articles_datesformat_filtered.csv'
    file3 = 'matched_companies_filtered.csv'
    output = 'final_data_all.csv'
    merge_type = 'inner'

    merge_csv_files(file1, file2, file3, output, merge_type)
