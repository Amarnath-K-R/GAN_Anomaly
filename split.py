import pandas as pd

def split_data(input_file, normal_file, anomalous_file):
    # Read the dataset
    df = pd.read_csv(input_file)
    
    # Split into normal (tsunami = 0) and anomalous (tsunami = 1)
    normal_df = df[df['tsunami'] == 0]
    anomalous_df = df[df['tsunami'] == 1]
    
    # Save the two datasets to separate CSV files
    normal_df.to_csv(normal_file, index=False)
    anomalous_df.to_csv(anomalous_file, index=False)
    print(f"Normal data saved to {normal_file} and Anomalous data saved to {anomalous_file}")

# Usage
split_data("earthquake.csv", "Normal.csv", "Anomalous.csv")
