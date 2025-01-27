import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler


# Define the generator model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.model(z)


# Define the discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# Load and preprocess the Iris dataset using sklearn
iris = load_iris()
data = iris.data  # Only use features (shape: (150, 4))

# Normalize the dataset
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
data = torch.tensor(data, dtype=torch.float32)

# GAN training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 10  # Latent space dimensionality
data_dim = data.shape[1]  # Number of features
batch_size = 32
epochs = 200
lr = 0.0002

# Create DataLoader
data_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

# Instantiate models
generator = Generator(input_dim=latent_dim, output_dim=data_dim).to(device)
discriminator = Discriminator(input_dim=data_dim).to(device)

# Define optimizers and loss function
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)
loss_fn = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    for real_batch, in data_loader:
        real_batch = real_batch.to(device)
        batch_size = real_batch.size(0)

        # Train discriminator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_data = generator(z)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        d_real = discriminator(real_batch)
        d_fake = discriminator(fake_data.detach())

        d_loss_real = loss_fn(d_real, real_labels)
        d_loss_fake = loss_fn(d_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # Train generator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_data = generator(z)
        g_loss = loss_fn(discriminator(fake_data), real_labels)

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Generate synthetic data
z = torch.randn(150, latent_dim).to(device)  # Generate the same number of samples as the Iris dataset
synthetic_data = generator(z).detach().cpu().numpy()
synthetic_data = scaler.inverse_transform(synthetic_data)  # Rescale back to the original range

print("Generated Data:", synthetic_data)



