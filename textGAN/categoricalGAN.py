import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# Load the Adult Income dataset
data = fetch_openml(name='adult', version=2, as_frame=True)
df = data.frame

# Separate features and target
X = df.drop(columns=['class'])
y = df['class']

# Identify categorical and numeric columns
cat_columns = X.select_dtypes(include='category').columns
num_columns = X.select_dtypes(include='number').columns

# Preprocess the dataset
# One-hot encode categorical variables
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_data = ohe.fit_transform(X[cat_columns])

# Normalize numeric data
scaler = MinMaxScaler()
num_data = scaler.fit_transform(X[num_columns])

# Assign unique column names for categorical and numeric features
cat_columns_expanded = [f"{col}_{i}" for col, n_vals in zip(cat_columns, ohe.categories_) for i in range(len(n_vals))]
num_columns_expanded = list(num_columns)

# Convert processed data into DataFrames
cat_df = pd.DataFrame(cat_data, columns=cat_columns_expanded)
num_df = pd.DataFrame(num_data, columns=num_columns_expanded)

# Combine all features into a single dataset
processed_data = torch.tensor(
    pd.concat([cat_df, num_df], axis=1).values,
    dtype=torch.float32
)

# GAN Training Parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 64  # Latent space dimensionality
data_dim = processed_data.shape[1]  # Number of features
batch_size = 64
epochs = 300
lr = 0.0002

# Create DataLoader
data_loader = DataLoader(TensorDataset(processed_data), batch_size=batch_size, shuffle=True)


# Define the Generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # Ensure values are in [0, 1]
        )

    def forward(self, z):
        return self.model(z)


# Define the Discriminator
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


# Instantiate models
generator = Generator(input_dim=latent_dim, output_dim=data_dim).to(device)
discriminator = Discriminator(input_dim=data_dim).to(device)

# Define optimizers and loss function
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)
loss_fn = nn.BCELoss()

# Training Loop
for epoch in range(epochs):
    for real_batch, in data_loader:
        real_batch = real_batch.to(device)
        batch_size = real_batch.size(0)

        # Train Discriminator
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

        # Train Generator
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
z = torch.randn(1000, latent_dim).to(device)
synthetic_data = generator(z).detach().cpu().numpy()

# Decode the generated data
# Split into categorical and numeric
cat_synthetic = synthetic_data[:, :cat_data.shape[1]]
num_synthetic = synthetic_data[:, cat_data.shape[1]:]

# Decode categorical data
cat_synthetic_decoded = ohe.inverse_transform(cat_synthetic)
num_synthetic_decoded = scaler.inverse_transform(num_synthetic)

# Combine results into a DataFrame
synthetic_df = pd.DataFrame(num_synthetic_decoded, columns=num_columns_expanded).join(
    pd.DataFrame(cat_synthetic_decoded, columns=cat_columns)
)

# synthetic_df.to_csv("synthetic_census.csv", index=False)

print("Synthetic Data Sample:")
print(synthetic_df.head())


