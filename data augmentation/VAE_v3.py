import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import math 
import matplotlib.pyplot as plt

def capture_reconstruction_sample(model, dataset, device):
    """Used to capture the progress of the model along the way"""
    model.no_grad()
    with torch.no_grad():
        # Creates a new vector of length (latent_dimension) 
        original = dataset[np.random.randint(len(dataset))].unsqueeze(0).to(device)
        # model takes teh vector and transforms it into a reconcustrcted image
        recon, _, _ = model(original)
        
        return (original.squeeze().cpu().numpy(), recon.squeeze().cpu().numpy())
    # enable further training
    model.train()

def plot_reconstruction_grid(samples):
    cols = 5
    rows = math.ceil(len(samples) / cols)
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 3, rows * 3)) # because we want rows for the original and reconstructed versions stacked on top of each other
    axes = axes.reshape(rows * 2, cols)

    # gemini did the display stuff, works pretty well so not gonna comment this part     
    for idx, (orig, recon) in enumerate(samples):
        row = (idx // cols) * 2
        col = idx % cols
        band_idx = orig.shape[0] // 2

        axes[row][col].imshow(orig[band_idx], cmap="jet")
        axes[row][col].set_title(f"Epoch {10 * (idx + 1)} Orig")
        axes[row][col].axis("off")

        axes[row + 1][col].imshow(recon[band_idx], cmap="jet")
        axes[row + 1][col].set_title(f"Epoch {10 * (idx + 1)} Recon")
        axes[row + 1][col].axis("off")

    for extra_row in range(rows * 2):
        for extra_col in range(len(axes[0])):
            if extra_row * cols + extra_col >= len(samples) * 2:
                axes[extra_row][extra_col].axis("off")

    plt.tight_layout()
    plt.show()

# Input data 
INPUT_SHAPE = (1, 15, 15, 15)  # 15 bands of 15 pixels in width and length (NEED TO CHANGE THIS AS WE GET PROPER DATA)
INPUT_DIM = (15, 15, 15) # used in the dataset portion, made a variable here so we know to change it (needs to be the same as above but without the 1) 
LATENT_DIM = 64 # the number of scalars that the encoder has to work with to describe the image (if latent dim was 1, they would describe the entire image with a single scalar, so lots and lots of information is loss. If it is too high (whatever 15 cubed is, for example), it will capture is too perfectly, leading to no variance when it is reconstructed by the decoder
BATCH_SIZE = 16 
LEARNING_RATE = 1e-3
EPOCHS = 120
FILE_PATH = r"C:\Users\liche\OneDrive\Desktop\PycharmProjects\NoNoise-\indian_pine_array.npy" 
DEVICE = torch.device("cpu") # no gpu :( never tested it on a gpu, so not sure if it'll work 

# dynamic dataset generator (read the obsidian note I made for it)
class HyperspectralLazyDataset(Dataset):
    # Essentially, a lazy data loader that stops me from needing to make hundreds of thousands of copies of the dataset (lots of memory), so I can just store the instructions to get to a datapoint (saves memory but sacrifices a bit of time, can be sped up with parallelism, but that's unnecessary because its already really fast) 

    def __init__(self, path_to_npy, crop_size=INPUT_DIM, patches_to_generate=1000):
        if not os.path.exists(path_to_npy):
            raise FileNotFoundError(f"{path_to_npy } can't be found")
        
        # Load the large array 
        self.full_data = np.load(path_to_npy).astype(np.float32)
        
        # Standardize to (B, H, W) if it's (H, W, B)
        if self.full_data.shape[0] > self.full_data.shape[2]:
            self.full_data = self.full_data.transpose(2, 0, 1)

        # Basic Min-Max Scaling to [0, 1]for VAE Sigmoid output
        self.full_data -= np.min(self.full_data) 
        self.full_data /= (np.max(self.full_data) + 1e-8)

        self.crop_size = crop_size # the size of the data being inputted (15 cubed)
        self.instructions = []

        # Currently only has functionality for cropping (no rotation or intensity shifting -> pretty important thing to work on        
        max_d = self.full_data.shape[0] - crop_size[0]
        max_h = self.full_data.shape[1] - crop_size[1]
        max_w = self.full_data.shape[2] - crop_size[2]
        
        for _ in range(patches_to_generate):
            d = np.random.randint(0, max_d + 1)
            h = np.random.randint(0, max_h + 1)
            w = np.random.randint(0, max_w + 1)
            self.instructions.append((d, h, w))

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        d, h, w = self.instructions[idx]
        patch = self.full_data[d:d+self.crop_size[0], 
                               h:h+self.crop_size[1], 
                               w:w+self.crop_size[2]]
        
        return torch.tensor(patch, dtype=torch.float32).unsqueeze(0)

class HyperspectralEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        # architecture defined in the notes -> will be in the github repo
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),
            nn.InstanceNorm3d(16), nn.LeakyReLU(0.2),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.InstanceNorm3d(32), nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm3d(64), nn.LeakyReLU(0.2)
        )
        with torch.no_grad():
            dummy_out = self.conv_layers(torch.zeros(1, *input_shape))
            self.feature_shape = dummy_out.shape[2:]
            self.flattened_dim = dummy_out.view(1, -1).size(1)

        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flattened_dim, latent_dim)

    def forward(self, x): # forward prop        
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc_mu(x), self.fc_log_var(x)

    # THE VAE REPARAMETERIZATION TRICK -> there will be notes on this in teh repo because this thing kinda makes no sense (very important for the training of VAEs and things with random sampling in general
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

# Architecture defined in the obsidian notes. Note how there is no reparameterization trick required here because no random sampling occurs at this step, so our losses can be backpropagated normally without reparameterization    
class HyperspectralDecoder(nn.Module):
    def __init__(self, latent_dim, feature_shape, flattened_dim):
        super().__init__()
        self.feature_shape = feature_shape
        self.fc = nn.Linear(latent_dim, flattened_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm3d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm3d(16), nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), 64, *self.feature_shape)
        return self.deconv(x)

# combines the two together    
class HyperspectralVAE(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.encoder = HyperspectralEncoder(input_shape, latent_dim)
        self.decoder = HyperspectralDecoder(latent_dim, self.encoder.feature_shape, self.encoder.flattened_dim)
        
    def forward(self, x):
        mu, log_var = self.encoder(x)
        # reparameterization needs to occur during training here because otherwise, backprop doesn't work 
        z = self.encoder.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        if recon_x.shape != x.shape:
            recon_x = nn.functional.interpolate(recon_x, size=x.shape[2:])
        return recon_x, mu, log_var

# The loss function -> MSE and KL divergence (want to add a couple more/swap them out for more domain specific losses) 
def vae_loss_fn(recon_x, x, mu, log_var):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') # performs a pixel by pixel comparison -> errors are added up 

    # KL divergence forces the encoder to make its latent space closely resemble the distribution specified previously (the normal distribution). If this is not done, the latent space can be very bumpy and sucky to calculate gradients (bad because we're using gradients) 
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) # not 100% how the math works, so notes might not be perfect for this part
    return recon_loss + kl_div # combines the loss (maybe add a scaling factor to both depending on which loss is more important -> might be able to get the training more stable with higher kl_div importance and better decoder performance with higher scaling factor for mse? (this seems like a pretty important thing to do because pretty big possibility that mse is completely swamping out kl-div at the start, making encoder training very choppy before it stabilizes, possibly into a local optima (maybe this is leading to loss plateauing?) (just speculation, but there is always more to do) 

# just visualization code written by gemini (not part of the vae) 
def visualize_reconstruction(model, dataset, device, epoch):
    model.eval()
    with torch.no_grad():
        # 1. Get a random sample from the dataset
        original = dataset[np.random.randint(len(dataset))].unsqueeze(0).to(device)
        
        # 2. Run it through the VAE
        recon, _, _ = model(original)
        
        # 3. Convert to numpy for plotting
        # Shape is [1, 1, Bands, H, W] -> remove first two dims
        original_np = original.squeeze().cpu().numpy()
        recon_np = recon.squeeze().cpu().numpy()
        
        # 4. Pick a middle band to visualize (e.g., band 7 of 15)
        band_idx = original_np.shape[0] // 2
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_np[band_idx], cmap='jet')
        axes[0].set_title(f"Original (Band {band_idx})")
        axes[1].imshow(recon_np[band_idx], cmap='jet')
        axes[1].set_title(f"Not Original (Band {band_idx})")
        
        plt.suptitle(f"Thing {epoch}")
        plt.show()
        model.train()


if __name__ == "__main__":
    print(f"--- Loading Indian Pines from {FILE_PATH} ---")
    ds = HyperspectralLazyDataset(FILE_PATH, crop_size=INPUT_SHAPE[1:], patches_to_generate=500)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) # the lazy data loading system from earlier 
    
    model = HyperspectralVAE(INPUT_SHAPE, LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    reconstruction_history = []

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            recon, mu, log_var = model(batch)
            loss = vae_loss_fn(recon, batch, mu, log_var)

            # standard gradient backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
                
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {epoch_loss/len(ds):.4f}")
            
        if (epoch + 1) % 10 == 0:
            #     visualize_reconstruction(model, ds, DEVICE, epoch + 1)  
            reconstruction_history.append(capture_reconstruction_sample(model, ds, DEVICE))

        visualize_reconstruction(model, ds, DEVICE, epoch + 1)  

        plot_reconstruction_grid(reconstruction_history)

        # NEED THIS TO LOAD THE MODEL AGAIN (for test_VAE.py) 
        torch.save({
            "decoder_state": model.decoder.state_dict(), # every learnable parameter of the decoder (weights, biases) (need this to recreate my model later) 
            "feature_shape": model.encoder.feature_shape, # need this to know what size the decoder takes 
            "flattened_dim": model.encoder.flattened_dim, # total count of values in a "layer", so it can reconstruct the model 
        }, "decoder_bundle.pth") # the weights and biases and some metadata, used to recreate the actual model (this contains the stuff that you plug into the shell of a model, that makes it the actual model) hopefully that makes sense...
