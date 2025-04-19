import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import pennylane as qml
import numpy as np
import torch.nn.init as init # Added for init_weights
import csv # Added for CSV logging
from torchmetrics import StructuralSimilarityIndexMeasure # Added for SSIM
from torch_fidelity import calculate_metrics # Added for FID

# Setup
os.environ['OMP_NUM_THREADS'] = '24' # Adjust as needed
os.environ['MKL_NUM_THREADS'] = '24' # Adjust as needed
torch.set_num_threads(24) # Adjust as needed
torch.set_num_interop_threads(24) # Adjust as needed

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# === Helper function for initialization (from qasa.py) ===
def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)


# === Cosine Beta Schedule ===
def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64, device=DEVICE)
    alphas_cumprod = torch.cos(((steps / timesteps + s) / (1 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999).float()

# === Sinusoidal Timestep Embedding ===
def get_timestep_embedding(timesteps, embedding_dim=128):
    """Build sinusoidal embeddings. From Fair Diffusion."""
    assert embedding_dim % 2 == 0
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=DEVICE) * -emb)
    # Ensure timesteps is on the correct device before computation
    timesteps = timesteps.to(DEVICE)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb

# === Positional Encoding (from qasa.py) ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

# === Quantum Layer (Adapted from qasa.py) ===
class QASALayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_qubits=8, n_layers=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # --- Device Selection Logic ---
        # For backpropagation through the whole network (needed for training),
        # default.qubit often offers better compatibility than lightning.qubit.
        # Let's prioritize default.qubit when using backprop.
        diff_method_for_training = "backprop"
        try:
            # Attempt to use lightning.qubit first if available *and* if it supports the diff method (heuristically assuming it might not for backprop)
            # A safer approach for backprop is often default.qubit
            # self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
            # print("Attempting to use lightning.qubit...")
            # qml.qnode(self.dev, interface="torch", diff_method=diff_method_for_training) # Test if compatible
            # print("lightning.qubit seems compatible with backprop.")
            # Forcing default.qubit for stability with backprop:
            print("Forcing default.qubit for backprop compatibility.")
            self.dev = qml.device("default.qubit", wires=self.n_qubits)

        except qml.QuantumFunctionError:
            print(f"lightning.qubit incompatible with diff_method='{diff_method_for_training}'. Falling back to default.qubit.")
            self.dev = qml.device("default.qubit", wires=self.n_qubits)
        except qml.DeviceError:
            print("lightning.qubit not available. Falling back to default.qubit.")
            self.dev = qml.device("default.qubit", wires=self.n_qubits)

        print(f"Using Pennylane device: {self.dev.name} with {self.n_qubits} wires.")
        # --- End Device Selection Logic ---

        @qml.qnode(self.dev, interface="torch", diff_method=diff_method_for_training)
        def quantum_circuit(inputs, weights):
            # Encoding input features onto qubits (Angle encoding)
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
                qml.RZ(inputs[i], wires=i) # Added RZ based on qasa

            # Applying parameterized layers
            for l in range(self.n_layers): # Using n_layers parameter
                # Parameterized gates (example: RX, RZ, CNOT structure from qasa)
                for i in range(self.n_qubits):
                    qml.RX(weights[l, 0, i], wires=i) # Using weight index
                    qml.RZ(weights[l, 1, i], wires=i) # Using weight index

                # Entanglement between qubits
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, (i + 1)]) # Simplified CNOT structure

            # Measurement (Expectation value of Pauli Z)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # Adjust weight shape based on circuit structure (layers, params_per_layer, qubits)
        self.weight_shapes = {"weights": (self.n_layers, 2, self.n_qubits)} # 2 params (RX, RZ) per qubit per layer
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, self.weight_shapes)

        # Linear layers to project features to/from qubit dimension
        self.input_proj = nn.Linear(input_dim, self.n_qubits)
        self.output_proj = nn.Linear(self.n_qubits, output_dim)

        # Initialize linear layer weights (optional but good practice)
        nn.init.kaiming_uniform_(self.input_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.input_proj.bias)
        nn.init.kaiming_uniform_(self.output_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.output_proj.bias)


    def forward(self, x):
        # Project input features to the dimension of qubits
        # x shape: (batch_size, input_dim) - Here batch_size is B*N
        x_proj = self.input_proj(x) # Shape: [B*N, n_qubits]

        # --- Manual Batching to bypass TorchLayer internal reshape issue ---
        batch_size_internal = x_proj.shape[0] # This is B*N
        outputs = []
        # Iterate through each item in the effective batch (each patch token)
        for i in range(batch_size_internal):
            # Pass each input vector individually to the qlayer
            # qlayer expects input shape matching the QNode input signature (inputs[i] -> size n_qubits)
            single_input = x_proj[i] # Shape: [n_qubits]
            single_output = self.qlayer(single_input) # qlayer called with single input, returns [n_qubits] tensor
            outputs.append(single_output)

        # Stack the results along the batch dimension
        quantum_output = torch.stack(outputs, dim=0) # Shape: [B*N, n_qubits]
        # --- End Manual Batching ---

        # Run the quantum circuit for each item in the batch
        # The qlayer expects input shape (batch_size, n_qubits)
        # quantum_output = self.qlayer(x_proj) # Original line - relies on internal batching

        # Project quantum output back to the desired output dimension
        out = self.output_proj(quantum_output) # Shape: [B*N, output_dim]
        return out

# === Quantum Encoder Layer (Adapted from qasa.py to use QASALayer) ===
class QuantumEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_qubits, n_layers, dropout_rate=0.1):
        super().__init__()
        # Using standard MultiheadAttention
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=dropout_rate)
        # Integrate QASALayer here
        self.q_layer = QASALayer(input_dim=hidden_dim, output_dim=hidden_dim, n_qubits=n_qubits, n_layers=n_layers)
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim) # LayerNorm before QASA layer

        # Apply initialization
        self.ffn.apply(init_weights)
        self.attn.apply(init_weights) # May need specific init for MHA weights/biases if desired

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        # Self-Attention
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output) # Add & Norm 1

        # QASA Layer Integration
        x_norm = self.norm3(x) # Normalize before QASA
        b, seq_len, dim = x_norm.shape
        # QASALayer expects (batch_size * seq_len, dim) if processing each token independently
        # Or adapt QASALayer/pooling if sequence-level processing is desired
        # Here, we process each token's features independently through QASA
        x_flat = x_norm.reshape(b * seq_len, dim)
        q_out_flat = self.q_layer(x_flat)
        q_out = q_out_flat.reshape(b, seq_len, dim)
        # Residual connection for the quantum layer output
        x = x + q_out # Add residual connection after QASA layer

        # Feed-Forward Network
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output) # Add & Norm 2

        return x


# === Diffusion Transformer (DiT) with optional QASA Quantum Layer ===
class DiffusionQASATransformer(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, hidden_dim=256, num_layers=6, num_heads=4, time_emb_dim=128, dropout_rate=0.1, use_quantum=False, q_n_qubits=8, q_n_layers=4):
        super().__init__()
        self.use_quantum = use_quantum
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_patches = (img_size // patch_size) ** 2

        # 1. Patch Embedding
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Timestep Embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 3. Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02) # Initialize positional embedding

        # 4. Dropout
        self.pos_drop = nn.Dropout(dropout_rate)

        # 5. Transformer Encoder Blocks
        encoder_layers = []
        num_classical_layers = num_layers - 1 if use_quantum else num_layers
        for _ in range(num_classical_layers):
            encoder_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=4 * hidden_dim,
                    dropout=dropout_rate,
                    activation='gelu', # Use GELU activation consistent with ViT/DiT
                    batch_first=True, # Expect input as (batch, seq, feature)
                    norm_first=True   # Apply LayerNorm before Attention/FFN (common in modern Transformers)
                )
            )

        # Add Quantum Layer if enabled
        if use_quantum:
            print(f"Initializing QuantumEncoderLayer at the end with hidden_dim={hidden_dim}, n_qubits={q_n_qubits}, n_layers={q_n_layers}")
            encoder_layers.append(
                QuantumEncoderLayer(hidden_dim=hidden_dim, n_qubits=q_n_qubits, n_layers=q_n_layers, dropout_rate=dropout_rate)
            )
        else:
            print("Using only classical TransformerEncoderLayers.")

        self.encoder = nn.ModuleList(encoder_layers)

        # 6. Final LayerNorm
        self.norm_out = nn.LayerNorm(hidden_dim)

        # 7. Output Projection (to predict noise in patchified format)
        self.output_proj = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)

        # Initialize weights
        self.apply(init_weights) # Apply custom init where applicable
        self.patch_embed.apply(init_weights) # Ensure patch embed is initialized
        self.output_proj.apply(init_weights) # Ensure output projection is initialized


    def unpatchify(self, x):
        """
        x: (B, N, P*P*C) N = number of patches, P = patch_size, C = channels
        returns: (B, C, H, W)
        """
        B = x.shape[0]
        N = x.shape[1] # Should be self.num_patches
        P = self.patch_size
        C = self.in_channels
        H_patch = W_patch = self.img_size // self.patch_size
        assert N == H_patch * W_patch

        x = x.reshape(B, H_patch, W_patch, P, P, C)
        x = torch.einsum('bhwpqc->bchpwq', x) # Permute axes
        imgs = x.reshape(B, C, H_patch * P, W_patch * P)
        return imgs

    def forward(self, x, time):
        # x: (B, C, H, W) Input image (potentially noisy)
        # time: (B,) Timestep indices

        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Input image size ({H}x{W}) doesn't match model ({self.img_size}x{self.img_size})."

        # 1. Patch Embedding
        x = self.patch_embed(x) # (B, hidden_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, N, hidden_dim) where N = (H/P)*(W/P)

        # 2. Timestep Embedding
        t_emb = get_timestep_embedding(time, 128) # Use original timestep embedding dim
        t = self.time_mlp(t_emb) # (B, hidden_dim)
        t = t.unsqueeze(1) # (B, 1, hidden_dim)

        # 3. Add Positional and Timestep Embeddings
        x = x + self.pos_embed # Add positional embedding (broadcasts along batch)
        x = x + t # Add timestep embedding (broadcasts along sequence)

        # 4. Dropout
        x = self.pos_drop(x)

        # 5. Transformer Encoder Blocks
        for layer in self.encoder:
            x = layer(x) # Each layer takes (B, N, hidden_dim)

        # 6. Final LayerNorm
        x = self.norm_out(x)

        # 7. Output Projection
        x = self.output_proj(x) # (B, N, P*P*C)

        # 8. Unpatchify to image format
        noise_pred = self.unpatchify(x) # (B, C, H, W)

        return noise_pred


# === Gaussian Diffusion (Adapted from quantum_diffusion_mnist_v7) ===
class GaussianDiffusionQASA:
    def __init__(self, timesteps=1000, beta_schedule='cosine'):
        self.timesteps = timesteps

        if beta_schedule == 'cosine':
            self.beta = cosine_beta_schedule(timesteps).to(DEVICE)
        else: # Add linear or other schedules if needed
             raise ValueError(f"Unsupported beta schedule: {beta_schedule}")

        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bar) # Renamed for clarity

        # Precompute values for p_sample variance calculation
        # Ensure all tensors used in calculation are on the correct device
        alpha_bar_prev = torch.cat([torch.tensor([self.alpha_bar[0]], device=DEVICE), self.alpha_bar[:-1]])

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alpha)
        # self.posterior_variance = self.beta * (1. - torch.cat([torch.tensor([self.alpha_bar[0]]), self.alpha_bar[:-1]])) / (1. - self.alpha_bar) # Original line
        self.posterior_variance = self.beta * (1. - alpha_bar_prev) / (1. - self.alpha_bar)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min =1e-20))
        # self.posterior_mean_coef1 = self.beta * torch.sqrt(torch.cat([torch.tensor([self.alpha_bar[0]]), self.alpha_bar[:-1]])) / (1. - self.alpha_bar)
        self.posterior_mean_coef1 = self.beta * torch.sqrt(alpha_bar_prev) / (1. - self.alpha_bar)
        # self.posterior_mean_coef2 = (1. - torch.cat([torch.tensor([self.alpha_bar[0]]), self.alpha_bar[:-1]])) * torch.sqrt(self.alpha) / (1. - self.alpha_bar)
        self.posterior_mean_coef2 = (1. - alpha_bar_prev) * torch.sqrt(self.alpha) / (1. - self.alpha_bar)


    def _extract(self, a, t, x_shape):
        # Helper function to extract specific indices from a tensor
        batch_size = t.shape[0]
        out = a.gather(-1, t.to(a.device)) # Ensure t is on the same device as a
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        # Forward diffusion process: q(x_t | x_0)
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar_t = self._extract(self.sqrt_alpha_bar, t, x_start.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alpha_bar, t, x_start.shape)

        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    def predict_x0_from_noise(self, x_t, t, noise):
        # Predict x_0 from x_t and predicted noise
        sqrt_recip_alpha_bar_t = self._extract(1.0 / self.sqrt_alpha_bar, t, x_t.shape)
        sqrt_recipm1_alpha_bar_t = self._extract(torch.sqrt(1.0 / self.alpha_bar - 1), t, x_t.shape)
        return sqrt_recip_alpha_bar_t * x_t - sqrt_recipm1_alpha_bar_t * noise

    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        # Calculate the mean and variance of the reverse process p(x_{t-1} | x_t)
        noise_pred = model(x_t, t)
        x0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)

        if clip_denoised:
            x0_pred.clamp_(-1., 1.)

        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = posterior_mean_coef1_t * x0_pred + posterior_mean_coef2_t * x_t

        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped_t = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance_t, posterior_log_variance_clipped_t


    @torch.no_grad()
    def p_sample(self, model, x, t):
        # Sample x_{t-1} from the model
        posterior_mean, _, posterior_log_variance_clipped_t = self.p_mean_variance(model, x, t)
        noise = torch.randn_like(x)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *((1,) * (len(x.shape) - 1)))
        return posterior_mean + nonzero_mask * (0.5 * posterior_log_variance_clipped_t).exp() * noise

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=1):
        # Generate samples from the model
        shape = (batch_size, channels, image_size, image_size)
        img = torch.randn(shape, device=DEVICE)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", total=self.timesteps, leave=False):
            t = torch.full((batch_size,), i, device=DEVICE, dtype=torch.long)
            img = self.p_sample(model, img, t)
            # Optionally save intermediate steps
            # if i % 50 == 0:
            #     imgs.append(img.cpu().numpy())
        imgs.append(img.cpu().numpy()) # Save final image
        return imgs # Return list of all saved images (or just the final one)


# === Sampling Helper ===
@torch.no_grad()
def generate_samples(model, diffusion, epoch, digit_label, n_samples=5, save_dir="."):
    model.eval()
    # Get img_size from the model if possible, default to 28 for MNIST
    img_size = getattr(model, 'img_size', 28)
    # Generate samples using the diffusion model
    samples_list = diffusion.sample(model, image_size=img_size, batch_size=n_samples, channels=1)
    final_samples = samples_list[-1] # Get the final generated images

    plt.figure(figsize=(10, 2))
    for i in range(n_samples):
        img = (final_samples[i].squeeze() + 1) / 2 # Denormalize from [-1, 1] to [0, 1]
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(np.clip(img, 0, 1), cmap='gray') # Clip just in case
        plt.axis("off")
    plt.suptitle(f"Digit {digit_label} - Epoch {epoch} Samples", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"epoch{epoch}_samples.png")
    plt.savefig(save_path)
    print(f"Saved samples to {save_path}")
    plt.close()
    model.train() # Set model back to training mode
    # Return generated samples for potential metric calculation (optional, if needed outside)
    # return final_samples # Or return samples_list[-1].to(DEVICE)


# === Training ===
def train_pipeline(digit_label=1,
                   use_quantum=False,
                   q_n_qubits=8,
                   q_n_layers=4,
                   epochs=30,
                   batch_size=64,
                   lr=3e-4,
                   timesteps=1000,
                   ema_decay=0.999,
                   img_size=28, # Added image size parameter
                   patch_size=4, # Added patch size parameter
                   hidden_dim=256, # Added hidden dim parameter
                   num_layers=6, # Added num layers parameter
                   num_heads=4, # Added num heads parameter
                   dropout_rate=0.1, # Added dropout rate parameter
                   base_save_dir="diffusion_qasa"):

    save_dir = os.path.join(base_save_dir, f"mnist_{digit_label}" + ("_quantum" if use_quantum else "_classical"))
    os.makedirs(save_dir, exist_ok=True)
    print(f"Starting training for digit {digit_label}, Quantum: {use_quantum}")
    print(f"Saving results to: {save_dir}")
    if use_quantum:
        print(f"Quantum params: n_qubits={q_n_qubits}, n_layers={q_n_layers}")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), # Resize to specified image size
        transforms.ToTensor(), # Scales to [0, 1]
        transforms.Normalize((0.5,), (0.5,)) # Normalizes to [-1, 1]
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Filter dataset for the specific digit
    idx = dataset.targets == digit_label
    dataset.data, dataset.targets = dataset.data[idx], dataset.targets[idx]
    print(f"Training on {len(dataset)} images of digit '{digit_label}'")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) # Added num_workers

    # Initialize Model, Diffusion, Optimizer
    # model = DiffusionQASAUNet(...) # Removed UNet instantiation
    model = DiffusionQASATransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        time_emb_dim=128, # Keep consistent timestep embedding dim
        dropout_rate=dropout_rate,
        use_quantum=use_quantum,
        q_n_qubits=q_n_qubits,
        q_n_layers=q_n_layers
    ).to(DEVICE)

    # Optional: Print model parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")


    ema_model = copy.deepcopy(model) # For generating samples
    diffusion = GaussianDiffusionQASA(timesteps=timesteps, beta_schedule='cosine')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    best_loss = float('inf')

    # --- CSV Logger for FID/SSIM ---
    eval_log_path = os.path.join(save_dir, "metrics_eval.csv")
    with open(eval_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'fid', 'ssim']) # Add other metrics if needed
    print(f"Initialized evaluation metrics log at: {eval_log_path}")
    # ------------------------------

    # --- Prepare Real Data Subset for FID/SSIM ---
    num_real_samples_for_eval = 500 # Number of real samples for consistent FID/SSIM evaluation
    real_subset_indices = torch.randperm(len(dataset))[:num_real_samples_for_eval]
    # Create a DataLoader for the subset to easily get batches
    real_subset = torch.utils.data.Subset(dataset, real_subset_indices)
    real_loader_for_eval = DataLoader(real_subset, batch_size=batch_size, shuffle=False) # Use same batch size for SSIM
    # Pre-load all real images for FID calculation
    all_real_images_for_fid = []
    for real_batch, _ in real_loader_for_eval:
        all_real_images_for_fid.append(real_batch)
    all_real_images_for_fid = torch.cat(all_real_images_for_fid, dim=0).to(DEVICE)
    print(f"Prepared {all_real_images_for_fid.shape[0]} real samples for evaluation.")
    # ---------------------------------------------

    # Initialize SSIM Metric
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE) # Input range [0, 1]

    # Training Loop
    for epoch in range(epochs):
        epoch_loss_sum = 0.0
        model.train() # Ensure model is in training mode
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for step, (x, _) in enumerate(progress_bar):
            x = x.to(DEVICE) # Shape: (B, 1, 28, 28)
            optimizer.zero_grad()

            # Sample random timestep for each image in the batch
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=DEVICE).long()

            # Calculate loss (predict noise)
            noise = torch.randn_like(x)
            x_noisy = diffusion.q_sample(x_start=x, t=t, noise=noise)
            predicted_noise = model(x_noisy, t)
            loss = F.mse_loss(predicted_noise, noise)

            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()

            # --- EMA Update ---
            with torch.no_grad():
                # Use parameters() method directly
                for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                     ema_p.data.mul_(ema_decay).add_(model_p.data, alpha=1 - ema_decay)
            # --- End EMA Update ---

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_epoch_loss = epoch_loss_sum / len(loader)
        loss_history.append(avg_epoch_loss)
        print(f"[Epoch {epoch+1}] Average Loss: {avg_epoch_loss:.4f}")

        # Save sample images using EMA model & Calculate Metrics
        # if (epoch + 1) % 5 == 0 or epoch == epochs - 1: # Save every 5 epochs and last epoch
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1: # Evaluate every 10 epochs and last epoch (FID is slow)
             print(f"\n--- Evaluating Epoch {epoch + 1} ---")
             generate_samples(ema_model, diffusion, epoch + 1, digit_label, n_samples=10, save_dir=save_dir)

             # --- Calculate FID and SSIM ---
             ema_model.eval() # Ensure EMA model is in eval mode
             with torch.no_grad():
                # 1. Generate a larger batch of samples for evaluation
                num_gen_samples_for_eval = num_real_samples_for_eval # Match number of real samples
                print(f"Generating {num_gen_samples_for_eval} samples for FID/SSIM calculation...")
                generated_samples_list = diffusion.sample(ema_model, image_size=img_size, batch_size=num_gen_samples_for_eval, channels=1)
                generated_samples = torch.tensor(generated_samples_list[-1]).to(DEVICE) # Get final samples [N, C, H, W]

                # 2. Prepare images for metrics
                # FID: needs uint8, [0, 255], 3 channels (RGB)
                generated_fid = ((generated_samples + 1) / 2 * 255).byte().repeat(1, 3, 1, 1) # Denorm, scale, cast, repeat channels
                real_fid = ((all_real_images_for_fid + 1) / 2 * 255).byte().repeat(1, 3, 1, 1)

                # SSIM: needs float, [0, 1], can be 1 channel
                generated_ssim = (generated_samples + 1) / 2 # Denorm to [0, 1]
                real_ssim_batch = (all_real_images_for_fid[:generated_ssim.shape[0]] + 1) / 2 # Take a matching size batch

                # 3. Calculate FID
                fid_value = float('nan') # Default to NaN if calculation fails
                try:
                    print("Calculating FID...")
                    metrics_dict = calculate_metrics(
                        input1=generated_fid,
                        input2=real_fid,
                        cuda=DEVICE.type == 'cuda',
                        fid=True,
                        verbose=False
                    )
                    fid_value = metrics_dict['frechet_inception_distance']
                    print(f"FID calculated: {fid_value:.4f}")
                except Exception as e:
                    print(f"Error calculating FID: {e}")

                # 4. Calculate SSIM
                ssim_value = float('nan') # Default to NaN
                try:
                    print("Calculating SSIM...")
                    # Ensure batches match for SSIM calculation if numbers differ slightly
                    min_batch = min(generated_ssim.shape[0], real_ssim_batch.shape[0])
                    ssim_value = ssim_metric(generated_ssim[:min_batch], real_ssim_batch[:min_batch]).item()
                    print(f"SSIM calculated: {ssim_value:.4f}")
                except Exception as e:
                    print(f"Error calculating SSIM: {e}")

                # 5. Log metrics to CSV
                with open(eval_log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, fid_value, ssim_value])
                print(f"Logged evaluation metrics for epoch {epoch + 1}.")

             print(f"--- Evaluation Complete ---\n")
             # Set model back to training mode if needed (generate_samples already does this, but good practice)
             model.train()
             # ema_model is only used for eval, no need to set back?


        # Save checkpoint (best model based on loss)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': ema_model.state_dict(), # Save EMA model state
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                # Add model config to checkpoint
                'model_config': {
                     'img_size': img_size,
                     'patch_size': patch_size,
                     'in_channels': 1,
                     'hidden_dim': hidden_dim,
                     'num_layers': num_layers,
                     'num_heads': num_heads,
                     'time_emb_dim': 128,
                     'dropout_rate': dropout_rate,
                     'use_quantum': use_quantum,
                     'q_n_qubits': q_n_qubits if use_quantum else None,
                     'q_n_layers': q_n_layers if use_quantum else None,
                }
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"Saved new best model checkpoint at epoch {epoch+1} with loss {best_loss:.4f}")

    # --- End Training Loop ---

    # Save final model explicitly
    torch.save({
        'epoch': epochs,
        'model_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss,
        # Add model config to checkpoint
        'model_config': {
             'img_size': img_size,
             'patch_size': patch_size,
             'in_channels': 1,
             'hidden_dim': hidden_dim,
             'num_layers': num_layers,
             'num_heads': num_heads,
             'time_emb_dim': 128,
             'dropout_rate': dropout_rate,
             'use_quantum': use_quantum,
             'q_n_qubits': q_n_qubits if use_quantum else None,
             'q_n_layers': q_n_layers if use_quantum else None,
        }
    }, os.path.join(save_dir, "final_model.pth"))
    print("Saved final model checkpoint.")


    # Save loss history to txt
    with open(os.path.join(save_dir, "loss.txt"), "w") as f:
        for l in loss_history:
            f.write(f"{l}\n")

    # Plot loss curve
    plt.figure()
    plt.plot(range(1, epochs + 1), loss_history)
    plt.title(f"Training Loss (Digit {digit_label}, Quantum: {use_quantum})")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE Loss")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()
    print("Saved loss curve plot.")


if __name__ == '__main__':
    # --- Configuration ---
    DIGITS_TO_TRAIN = range(10) # Train for digits 0 through 9
    EPOCHS = 50             # Number of training epochs
    BATCH_SIZE = 64         # Reduced batch size for potentially larger DiT model
    LEARNING_RATE = 1e-4     # Often need lower LR for Transformers
    TIMESTEPS = 1000         # Number of diffusion timesteps
    EMA_DECAY = 0.999        # Exponential Moving Average decay for sampling model
    BASE_SAVE_DIR = "results/diffusion_qasa_dit_mnist" # Updated save directory name

    # --- DiT Model Specific Config ---
    IMG_SIZE = 28            # Input image size
    PATCH_SIZE = 4           # Patch size (must divide img_size)
    HIDDEN_DIM = 192         # Transformer hidden dimension (adjust based on resources)
    NUM_LAYERS = 6           # Number of Transformer layers
    NUM_HEADS = 4            # Number of attention heads
    DROPOUT_RATE = 0.1       # Dropout rate

    # Quantum Layer Specific Config (only used if use_quantum=True)
    Q_N_QUBITS = 8           # Number of qubits in the QASALayer
    Q_N_LAYERS = 4           # Number of layers in the QASALayer circuit

    # --- Run Training ---
    for digit in DIGITS_TO_TRAIN:
        print("-" * 50)
        # Train Classical DiT Version
        train_pipeline(
            digit_label=digit,
            use_quantum=False,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            timesteps=TIMESTEPS,
            ema_decay=EMA_DECAY,
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            dropout_rate=DROPOUT_RATE,
            base_save_dir=BASE_SAVE_DIR
        )

        print("-" * 50)
        # Train Quantum DiT Version
        train_pipeline(
            digit_label=digit,
            use_quantum=True,
            q_n_qubits=Q_N_QUBITS,
            q_n_layers=Q_N_LAYERS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE, # Quantum training might need smaller batch size
            lr=LEARNING_RATE,
            timesteps=TIMESTEPS,
            ema_decay=EMA_DECAY,
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            dropout_rate=DROPOUT_RATE,
            base_save_dir=BASE_SAVE_DIR
        )
        print("=" * 50)

    print("Training complete for all specified digits.")