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
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(1024, sequence_length*num_features),
            nn.Tanh()
        )

    def forward(self, z):
        generated = self.model(z)
        return generated.view(z.size(0), -1, 8)

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
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, sequence):
        sequence = sequence.view(sequence.size(0), -1)  
        features = self.feature_extractor(sequence)
        return self.classifier(features), features

class EnhancedAnoGAN:
    def __init__(self, sequence_length, latent_dim=100, lambda_ano=0.1,num_features=8, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.lambda_ano = lambda_ano
        self.device = device

        self.generator = Generator(latent_dim, sequence_length,num_features).to(device)
        self.discriminator = Discriminator(sequence_length,num_features).to(device)

        self.adversarial_loss = nn.BCELoss()
        self.feature_matching_loss = nn.MSELoss()

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train_step(self, real_sequences):
        batch_size = real_sequences.size(0)
        real_sequences = real_sequences.to(self.device)
        valid = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)

        # Train Generator
        self.g_optimizer.zero_grad()
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        generated_sequences = self.generator(z)
        
        validity, gen_features = self.discriminator(generated_sequences)
        _, real_features = self.discriminator(real_sequences)
        
        # Enhanced generator loss with feature matching
        g_loss = self.adversarial_loss(validity, valid) + \
                 0.1 * self.feature_matching_loss(gen_features, real_features.detach())
        
        g_loss.backward()
        self.g_optimizer.step()

        # Train Discriminator
        self.d_optimizer.zero_grad()
        real_validity, _ = self.discriminator(real_sequences)
        fake_validity, _ = self.discriminator(generated_sequences.detach())
        
        real_loss = self.adversarial_loss(real_validity, valid)
        fake_loss = self.adversarial_loss(fake_validity, fake)
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        self.d_optimizer.step()

        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item()
        }

class TsunamiDetector:
    def __init__(self, sequence_length, latent_dim=100, lambda_ano=0.1,num_features=8, device='cpu'):
        self.anogan = EnhancedAnoGAN(sequence_length, latent_dim, lambda_ano,num_features, device)
        self.threshold = None
        self.device = device
        self.scaler = None
        
    def train(self, dataloader, epochs=100):
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
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: G_loss: {epoch_g_loss/batches:.4f}, D_loss: {epoch_d_loss/batches:.4f}")

    def calibrate_threshold(self, validation_data, percentile=95):
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
        optimizer = optim.Adam([z], lr=0.001)

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
        confidence = 1.0 - np.exp(-max(0, anomaly_score - self.threshold) / self.threshold)
        
        is_anomaly = anomaly_score > (self.threshold * threshold_multiplier)
        
        return {
            'is_tsunami': is_anomaly,
            'confidence': confidence,
            'anomaly_score': anomaly_score,
            'threshold': self.threshold
        }


