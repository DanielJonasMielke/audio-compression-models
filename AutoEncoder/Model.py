import torch
import torch.nn as nn

class SimpleConvAutoencoder(nn.Module):
    """
    A simple 1D Convolutional Autoencoder for raw audio snippets.

    Assumes input audio is mono (1 channel) and normalized (e.g., to [-1, 1]).
    The architecture uses Conv1d for downsampling in the encoder and
    ConvTranspose1d for upsampling in the decoder.

    Input shape: [batch_size, 1, snippet_length]
    Output shape: [batch_size, 1, snippet_length] (reconstructed)
    """
    def __init__(self, snippet_length=44100):
        super().__init__()

        self.snippet_length = snippet_length

        # --- Encoder ---
        # Input: [B, 1, 44100]
        self.encoder = nn.Sequential(
            # formula to calculate output size of conv layer: O = [(W−K+2P)/S]+1
            # Layer 1
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=31, stride=4, padding=15), # Output: [B, 16, 11025]
            nn.ReLU(),
            # Layer 2
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=15, stride=4, padding=7), # Output: [B, 32, 2757]
            nn.ReLU(),
            # Layer 3
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=4, padding=3),   # Output: [B, 64, 690] -> This is our latent representation space
            nn.ReLU()
            # Note: Could add more layers or a final linear layer for a smaller bottleneck
        )

        # --- Decoder ---
        # Input: [B, 64, 690] (Output of Encoder)
        self.decoder = nn.Sequential(
            # Layer 1 (Transpose of Encoder Layer 3)
            # Needs output_padding calculation to match encoder output length precisely
            # Target length: 2757
            # Formula: L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding
            # L_out = (690 - 1)*4 - 2*3 + 7 + op = 689*4 - 6 + 7 + op = 2756 + 1 + op = 2757 + op
            # -> Needs output_padding = 0
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=7, stride=4, padding=3, output_padding=0), # Output: [B, 32, 2757]
            nn.ReLU(),

            # Layer 2 (Transpose of Encoder Layer 2)
            # Target length: 11025
            # L_out = (2757 - 1)*4 - 2*7 + 15 + op = 2756*4 - 14 + 15 + op = 11024 + 1 + op = 11025 + op
            # -> Needs output_padding = 0
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=15, stride=4, padding=7, output_padding=0), # Output: [B, 16, 11025]
            nn.ReLU(),

            # Layer 3 (Transpose of Encoder Layer 1)
            # Target length: 44100
            # L_out = (11025 - 1)*4 - 2*15 + 31 + op = 11024*4 - 30 + 31 + op = 44096 + 1 + op = 44097 + op
            # -> Needs output_padding = 3
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=31, stride=4, padding=15, output_padding=3), # Output: [B, 1, 44100]

            # Final Activation: Tanh
            # Tanh outputs values in [-1, 1]. This is suitable if your input audio
            # is normalized to the same range. If not, remove or change this.
            nn.Tanh()
        )

    def forward(self, x):
        """
        Passes the input audio snippet through the encoder and decoder.
        """
        # Check input shape consistency (optional but helpful)
        if x.shape[2] != self.snippet_length:
            print(f"Warning: Input length {x.shape[2]} does not match model's expected snippet_length {self.snippet_length}")
            # Handle potential mismatch if needed (e.g., padding/truncating), or raise error
            # For now, we assume the input length is correct

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Ensure output length matches input length exactly (important due to potential rounding)
        # If there's a small mismatch (e.g., off by one), you might need to slightly
        # adjust padding/output_padding or pad/trim the output here.
        if decoded.shape[2] != x.shape[2]:
             # Example: Trim if decoder output is slightly longer
             if decoded.shape[2] > x.shape[2]:
                 decoded = decoded[:, :, :x.shape[2]]
             # Example: Pad if decoder output is slightly shorter (less common with correct padding)
             elif decoded.shape[2] < x.shape[2]:
                 diff = x.shape[2] - decoded.shape[2]
                 # Simple zero padding on the right
                 decoded = nn.functional.pad(decoded, (0, diff))


        return decoded
