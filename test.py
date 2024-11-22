import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from TsunoGan import EnhancedAnoGAN, TsunamiDetector  # Importing your classes

# Constants for the dataset
NUM_FEATURES = 6  # Columns: magnitude, cdi, mmi, tsunami, sig, dmin
SEQUENCE_LENGTH = 1  # Adjust if handling sequential data
LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 200

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Select relevant numerical columns
    data = df[['magnitude', 'cdi', 'mmi', 'dmin', 'gap', 'depth']].fillna(0).values

    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)

    # Convert to tensors
    tensor_data = torch.tensor(normalized_data, dtype=torch.float32)
    return TensorDataset(tensor_data), scaler

# Create DataLoader
def create_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Main function to train and evaluate the model
def main():
    # Load and preprocess the data
    dataset, scaler = load_and_preprocess_data("earthquake.csv")
    dataloader = create_dataloader(dataset, BATCH_SIZE)

    # Initialize the tsunami detector
    detector = TsunamiDetector(
        sequence_length=SEQUENCE_LENGTH,
        latent_dim=LATENT_DIM,
        num_features=NUM_FEATURES
    )

    # Train the model
    print("Training the TsunoGAN...")
    detector.train(dataloader, epochs=EPOCHS)

    # Generate synthetic data and evaluate
    detector.anogan.generator.eval()
    with torch.no_grad():
        z = torch.randn(10, LATENT_DIM).to(detector.device)  # Generate 10 synthetic samples
        synthetic_data = detector.anogan.generator(z)

    # Transform synthetic data back to the original scale for interpretation
    synthetic_data = synthetic_data.view(-1, NUM_FEATURES).cpu().numpy()
    synthetic_data = scaler.inverse_transform(synthetic_data)

    print("Generated Synthetic Data:")
    print(synthetic_data)

    # Example evaluation (comparing synthetic vs real data)
    real_data = dataset[:10][0].numpy()  # Get the first 10 samples from the real dataset
    real_data = scaler.inverse_transform(real_data)

    mse = ((synthetic_data - real_data) ** 2).mean()
    print(f"Mean Squared Error between real and synthetic data: {mse:.4f}")

if __name__ == "__main__":
    main()
