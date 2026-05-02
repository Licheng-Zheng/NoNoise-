# loads the model and recreates the model to be used

import torch
import matplotlib.pyplot as plt

from VAE_v3 import HyperspectralVAE, HyperspectralDecoder, HyperspectralLazyDataset
from VAE_v3 import LATENT_DIM, DEVICE

bundle = torch.load("decoder_bundle.pth", map_location=DEVICE)

decoder = HyperspectralDecoder(
    latent_dim=LATENT_DIM,
    feature_shape=bundle["feature_shape"],
    flattened_dim=bundle["flattened_dim"],
)
decoder.load_state_dict(bundle["decoder_state"])

# decoder.to(DEVICE)
# decoder.eval()

# latent = torch.randn(1, LATENT_DIM).to(DEVICE)
# print("Latent vector shape:", latent.shape)
# print("Latent vector:", latent)
# with torch.no_grad():
#     reconstruction = decoder(latent)

# recon_np = reconstruction.squeeze().cpu().numpy()
# band_idx = recon_np.shape[0] // 2 - 1 

# plt.figure(figsize=(5, 5))
# plt.imshow(recon_np[band_idx], cmap="jet")
# plt.title(f"Reconstruction (Band {band_idx})")
# plt.colorbar()
# plt.show()


decoder.to(DEVICE)
decoder.eval()

num_images = 15
grid_cols = 5
grid_rows = 3
fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 8))

for idx in range(num_images):
    latent = torch.randn(1, LATENT_DIM).to(DEVICE)
    with torch.no_grad():
        reconstruction = decoder(latent)

    recon_np = reconstruction.squeeze().cpu().numpy()
    band_idx = recon_np.shape[0] // 2 - 1

    row, col = divmod(idx, grid_cols)
    ax = axes[row, col]
    ax.imshow(recon_np[band_idx], cmap="jet")
    ax.set_title(f"Sample {idx + 1}")
    ax.axis("off")

for ax in axes.flat[num_images:]:
    ax.remove()

plt.tight_layout()
plt.show()
