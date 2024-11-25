import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Generator(nn.Module):
    def __init__(self, latent_dim, sequence_length,num_features):
        super(Generator, self).__init__()
        
        # Enhanced architecture with more layers and dropout
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, sequence_length*num_features),
            nn.Tanh()
        )

    def forward(self, z):
        generated = self.model(z)
        return generated.view(z.size(0), -1, self.num_features)

class Discriminator(nn.Module):
    def __init__(self, sequence_length,num_features):
        super(Discriminator, self).__init__()
        
        # Enhanced architecture with feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(sequence_length*num_features, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128), 
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, sequence):
        sequence = sequence.view(sequence.size(0), -1)  
        features = self.feature_extractor(sequence)
        return self.classifier(features), features

def wasserstein_loss(y_true, y_pred):
    return -torch.mean(y_true * y_pred)  # For both generator and discriminator

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=device)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates = interpolates.requires_grad_(True)

    # Compute discriminator scores for interpolates
    d_interpolates, _ = discriminator(interpolates)

    # Compute gradients with respect to interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Calculate gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty

def train_discriminator_with_gp(generator, discriminator, real_sequences, optimizer_d, lambda_gp, device):
    """
    Train the discriminator using Wasserstein loss and Gradient Penalty.
    """
    batch_size = real_sequences.size(0)
    real_sequences = real_sequences.to(device)

    # Generate fake samples
    z = torch.randn(batch_size, generator.latent_dim, device=device)
    fake_sequences = generator(z).detach()

    # Compute discriminator outputs
    real_validity, _ = discriminator(real_sequences)
    fake_validity, _ = discriminator(fake_sequences)

    # Compute Wasserstein loss components
    d_loss_real = -torch.mean(real_validity)
    d_loss_fake = torch.mean(fake_validity)

    # Compute gradient penalty
    gradient_penalty = compute_gradient_penalty(discriminator, real_sequences, fake_sequences, device)

    # Total loss for discriminator
    d_loss = d_loss_real + d_loss_fake + lambda_gp * gradient_penalty

    # Backpropagation and optimization
    optimizer_d.zero_grad()
    d_loss.backward()
    optimizer_d.step()

    return d_loss.item()


def train_generator_with_wasserstein(generator, discriminator, optimizer_g, device):
    """
    Train the generator using Wasserstein loss.
    """
    batch_size = next(generator.parameters()).device
    z = torch.randn(batch_size, generator.latent_dim, device=device)

    # Generate fake sequences
    fake_sequences = generator(z)

    # Compute generator loss (Wasserstein loss)
    fake_validity, _ = discriminator(fake_sequences)
    g_loss = -torch.mean(fake_validity)

    # Backpropagation and optimization
    optimizer_g.zero_grad()
    g_loss.backward()
    optimizer_g.step()

    return g_loss.item()




class EnhancedAnoGAN:
    def __init__(self, sequence_length, latent_dim=300, lambda_ano=0.1,num_features=6, device='cpu'):
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.lambda_ano = lambda_ano
        self.device = device

        self.generator = Generator(latent_dim, sequence_length,num_features).to(device)
        self.discriminator = Discriminator(sequence_length,num_features).to(device)

        self.adversarial_loss = nn.BCELoss()
        self.feature_matching_loss = nn.MSELoss()

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-4)


    def train_step(self, real_sequences):
        """
        Perform one training step for the generator and discriminator.
        """
        batch_size = real_sequences.size(0)
        real_sequences = real_sequences.to(self.device)

        # Train Discriminator with WGAN-GP
        d_loss = train_discriminator_with_gp(
            self.generator,
            self.discriminator,
            real_sequences,
            self.d_optimizer,
            lambda_gp=10,  # Set gradient penalty coefficient
            device=self.device
        )

        # Train Generator with Wasserstein loss
        g_loss = train_generator_with_wasserstein(
            self.generator,
            self.discriminator,
            self.g_optimizer,
            device=self.device
        )

        return {'g_loss': g_loss, 'd_loss': d_loss}



class TsunamiDetector:
    def __init__(self, sequence_length, latent_dim=300, lambda_ano=0.1,num_features=6, device='cpu'):
        self.anogan = EnhancedAnoGAN(sequence_length, latent_dim, lambda_ano,num_features, device)
        self.threshold = None
        self.device = device
        self.scaler = None
        
    def train(self, dataloader, epochs=200):
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            batches = 0
            
            for batch in dataloader:
                sequences = batch[0]
                losses = self.anogan.train_step(sequences)
                epoch_g_loss += losses['g_loss']
                epoch_d_loss += losses['d_loss']
                batches += 1
            
            
            print(f"Epoch {epoch}: G_loss: {epoch_g_loss/batches:.4f}, D_loss: {epoch_d_loss/batches:.4f}")

    def calibrate_threshold(self, validation_data, percentile=85):
        self.anogan.generator.eval()
        self.anogan.discriminator.eval()
        
        scores = []
        with torch.no_grad():
            for sequence in validation_data:
                sequence = sequence.to(self.device)
                sequence = sequence.unsqueeze(0)
                score = self._compute_anomaly_score(sequence)
                scores.append(score)
        
        self.threshold = np.percentile(scores, percentile)
        return self.threshold

    def _compute_anomaly_score(self, sequence):
        """Enhanced anomaly score computation with multiple metrics."""
        # Initialize z with gradients enabled
        torch.set_grad_enabled(True)

        self.anogan.generator.train()

        z = torch.randn(1, self.anogan.latent_dim, requires_grad=True, device=self.device)

        # Define optimizer for optimizing z
        optimizer = optim.Adam([z], lr=0.01)

        best_score = float('inf')

        # Iterate to optimize z
        for _ in range(100):  # Reduced iterations for performance
            optimizer.zero_grad()

            # Generate sequence using the current z
            generated = self.anogan.generator(z)  # `generated` should have grad enabled

            # Ensure computation graph is intact
            assert generated.requires_grad, "Generated sequence must have gradients enabled"

            # Calculate reconstruction loss
            reconstruction_loss = torch.mean((sequence - generated) ** 2)

            # Get discriminator features
            validity, gen_features = self.anogan.discriminator(generated)
            _, real_features = self.anogan.discriminator(sequence)

            # Calculate feature matching and discrimination losses
            feature_loss = torch.mean((gen_features - real_features) ** 2)
            discrimination_loss = torch.mean((validity - 1) ** 2)

            # Total loss for anomaly score
            total_loss = reconstruction_loss + 0.1 * feature_loss + 0.1 * discrimination_loss

            # Ensure loss is connected to `z`
            assert total_loss.requires_grad, "Total loss must have gradients enabled"

            # Track the best score
            if total_loss.item() < best_score:
                best_score = total_loss.item()

            # Backpropagation and optimization
            total_loss.backward()
            optimizer.step()

        return best_score



    def detect_tsunami(self, sequence, threshold_multiplier=1.0):
        """
        Enhanced tsunami detection with confidence score
        """
        if self.threshold is None:
            raise ValueError("Threshold not calibrated. Run calibrate_threshold first.")
        
        sequence = sequence.to(self.device).unsqueeze(0)
        
        anomaly_score = self._compute_anomaly_score(sequence)
        
        # Calculate confidence score (0 to 1)
        confidence = np.clip((anomaly_score - self.threshold) / self.threshold, 0, 1)




        
        is_anomaly = anomaly_score > (self.threshold * threshold_multiplier)
        
        return {
            'is_tsunami': is_anomaly,
            'confidence': confidence,
            'anomaly_score': anomaly_score,
            'threshold': self.threshold
        }


