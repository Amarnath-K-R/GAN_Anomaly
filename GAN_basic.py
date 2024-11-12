import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# 1. Prepare the Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_dataset, dev_dataset = random_split(train_dataset, [int(len(train_dataset) * 0.83), int(len(train_dataset) * 0.17)])

# Data Loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2. Define the GAN Model
class Generator(nn.Module):
    def __init__(self, latent_dim, img_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim),
            nn.Tanh()  # Output values in [-1, 1] range
        )
        
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output a single probability value
        )
        
    def forward(self, x):
        return self.model(x)

# Initialize the models
latent_dim = 100
img_dim = 28 * 28
generator = Generator(latent_dim, img_dim)
discriminator = Discriminator(img_dim)

# Loss and Optimizers
criterion = nn.BCELoss()
lr = 0.0002
G_optimizer = optim.Adam(generator.parameters(), lr=lr)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# 3. Training Loop
epochs = 50
for epoch in range(epochs):
    for real_imgs, _ in train_loader:
        batch_size = real_imgs.size(0)
        
        # Flatten the images
        real_imgs = real_imgs.view(batch_size, -1)
        
        # Labels for real and fake images
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train Discriminator
        D_optimizer.zero_grad()
        
        # Real images
        real_outputs = discriminator(real_imgs)
        D_real_loss = criterion(real_outputs, real_labels)
        
        # Fake images
        noise = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(noise)
        fake_outputs = discriminator(fake_imgs.detach())
        D_fake_loss = criterion(fake_outputs, fake_labels)
        
        # Total discriminator loss
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_optimizer.step()
        
        # Train Generator
        G_optimizer.zero_grad()
        
        # Generate fake images and calculate loss
        fake_outputs = discriminator(fake_imgs)
        G_loss = criterion(fake_outputs, real_labels)  # Flip labels to fool discriminator
        G_loss.backward()
        G_optimizer.step()
        
    print(f'Epoch [{epoch+1}/{epochs}], D Loss: {D_loss.item():.4f}, G Loss: {G_loss.item():.4f}')

# 4. Generate and Visualize Samples
def generate_and_save_images(epoch, generator, latent_dim, n_images=16):
    noise = torch.randn(n_images, latent_dim)
    fake_imgs = generator(noise).view(-1, 1, 28, 28).cpu().detach() * 0.5 + 0.5  # Rescale to [0, 1]
    grid = torch.cat([img for img in fake_imgs], dim=2).view(28, -1)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.numpy(), cmap="gray")
    plt.axis("off")
    plt.show()

# Generate some final samples
generate_and_save_images(epochs, generator, latent_dim)
