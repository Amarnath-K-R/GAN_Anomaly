import torch
import torch.nn as nn
import torch.optim as optim

# Define Generator
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.model(x)

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

# Hyperparameters
input_dim = 100
hidden_dim = 128
data_dim = 28*28  # For example, using flattened MNIST images
batch_size = 64
epochs = 100

# Initialize models
G = Generator(input_dim, hidden_dim, data_dim)
D = Discriminator(data_dim, hidden_dim)

# Loss and optimizers
criterion = nn.BCELoss()
G_optimizer = optim.Adam(G.parameters(), lr=0.0002)
D_optimizer = optim.Adam(D.parameters(), lr=0.0002)

# Training Loop
for epoch in range(epochs):
    for real_data in data_loader:  # Assuming data_loader provides batches of normal data
        
        # Discriminator training
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # Train with real data
        D_optimizer.zero_grad()
        real_output = D(real_data)
        D_real_loss = criterion(real_output, real_labels)
        
        # Train with fake data
        noise = torch.randn(batch_size, input_dim)
        fake_data = G(noise)
        fake_output = D(fake_data)
        D_fake_loss = criterion(fake_output, fake_labels)
        
        # Total discriminator loss
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_optimizer.step()
        
        # Generator training
        G_optimizer.zero_grad()
        noise = torch.randn(batch_size, input_dim)
        fake_data = G(noise)
        fake_output = D(fake_data)
        G_loss = criterion(fake_output, real_labels)
        G_loss.backward()
        G_optimizer.step()
        
    print(f'Epoch [{epoch+1}/{epochs}], D Loss: {D_loss.item()}, G Loss: {G_loss.item()}')

# Anomaly Detection (using reconstruction error)
def anomaly_score(test_data):
    reconstructed_data = G(D(test_data))
    reconstruction_error = torch.mean((test_data - reconstructed_data) ** 2, dim=1)
    return reconstruction_error
