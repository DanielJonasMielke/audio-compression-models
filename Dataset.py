import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
import logging

AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_peak(waveform):
    # Normalize waveform to [-1, 1] based on peak absolute value
    max_abs_val = torch.max(torch.abs(waveform))
    if max_abs_val > 1e-6: # Avoid division by zero or near-zero
        return waveform / max_abs_val
    else:
        return waveform # Return as is if waveform is silent or near-silent

class AudioSnippetDataset(Dataset):
    def __init__(self, dataset_directory, snippet_length=44100, target_sr=44100, recursive=True, transform=None): # 1 second audio clips
        super().__init__()
        logging.info("// ------------------ INIT DATASET ------------------ //\n\n")
        self.dataset_directory = Path(dataset_directory) # Convert string to Path object
        self.snippet_length = snippet_length
        self.target_sr = target_sr
        self.recursive = recursive
        self.transform = transform

        logging.info(f"Target snippet length: {self.snippet_length} samples")
        logging.info(f"Target sampling rate: {self.target_sr} Hz / {self.target_sr / 1000}kHz") # (usually 44.1kHz)

        # --- Find audio files ---
        logging.info(f"Searching for audio files in: {self.dataset_directory}")
        search_pattern = "**/*" if recursive else "*"  # Search recursively or just top-level
        all_files = self.dataset_directory.glob(search_pattern)

        # Filter for files with the correct extensions (case-insensitive)
        self.file_paths = sorted([
            p for p in all_files
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        ])

        if not self.file_paths:
            logging.warning(f"No audio files with extensions {AUDIO_EXTENSIONS} found in {self.dataset_directory}")
        else:
            logging.info(f"Found {len(self.file_paths)} audio files.")

        # --- Prepare snippet information ---
        self.snippet_info = [] # contains tuples if (file_idx, sample_start_idx)
        self._prepare_snippets()

        assert self.snippet_info , f"No snippets were created. Check audio files are long enough and readable in {self.dataset_directory}"

        logging.info(f"Dataset initialized with {len(self.snippet_info)} snippets.")
        logging.info("\n\n// ------------------ CLOSE INIT DATASET ------------------ //")

    def _prepare_snippets(self):
        """Iterates through files and calculates valid snippet start points."""
        logging.info("Calculating snippet indices...")
        num_files_processed = 0
        for file_index, file_path in enumerate(self.file_paths):
            try:
                # Get audio metadata (sampling rate, number of frames)
                # Use str(file_path) because torchaudio.info might expect string paths
                info = torchaudio.info(file_path)
                original_sr = info.sample_rate
                original_num_frames = info.num_frames

                if original_sr != self.target_sr:
                    logging.warning(
                        f"Skipping '{file_path.name}': Sample rate does not match target sample rate. ")
                    continue

                if original_num_frames < self.snippet_length:
                    logging.warning(
                        f"Skipping '{file_path.name}': Number of frames does not match snippet length.")
                    continue

                # Calculate start samples for non-overlapping snippets
                num_snippets_in_file = original_num_frames // self.snippet_length  # Integer division for non-overlapping
                for i in range(num_snippets_in_file):
                    start_sample = i * self.snippet_length
                    self.snippet_info.append((file_index, start_sample))

                num_files_processed += 1

            except Exception as e:
                # Log errors encountered while trying to get info or process a file
                logging.error(f"Could not process file {file_path}: {e}",
                              exc_info=False)  # Set exc_info=True for full traceback

        logging.info(f"Calculated snippets from {num_files_processed}/{len(self.file_paths)} files.")

    def __len__(self):
        return len(self.snippet_info)

    def __getitem__(self, idx):
        if idx >= len(self.snippet_info):
             # This should ideally not happen if used with standard samplers, but good practice
             raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")

        file_index, start_sample = self.snippet_info[idx]
        file_path = self.file_paths[file_index]

        try:
            snippet, sr = torchaudio.load(
                str(file_path),  # Use str() for safety with older torchaudio versions
                frame_offset=start_sample,
                num_frames=self.snippet_length
            )

            if snippet.shape[0] > 1:
                snippet = torch.mean(snippet, dim=0, keepdim=True)

            # Ensure snippet is 2D [1, num_samples] even if loaded as mono 1D
            if snippet.ndim == 1:
                snippet = snippet.unsqueeze(0)

            current_len = snippet.shape[1]
            if current_len < self.snippet_length:
                padding_needed = self.snippet_length - current_len
                # Pad with zeros on the right side
                snippet = torch.nn.functional.pad(snippet, (0, padding_needed))
                logging.debug(f"Padded snippet for index {idx} from {current_len} to {self.snippet_length}")
            elif current_len > self.snippet_length:
                # This shouldn't happen with the slicing above, but as a safeguard
                snippet = snippet[:, :self.snippet_length]
                logging.warning(f"Truncated snippet for index {idx} from {current_len} to {self.snippet_length}")

            # --- TODO: Add optional transform here ---
            if self.transform:
               snippet = self.transform(snippet)
            return snippet

        except Exception as e:
            logging.error(f"Error loading or processing snippet at index {idx} (file: {file_path}): {e}", exc_info=True)
            # Return a dummy tensor of zeros to avoid crashing the DataLoader.
            # The loss calculation should ideally handle this (e.g., ignore loss for zero inputs).
            return torch.zeros((1, self.snippet_length))


if __name__ == "__main__":
    dataset = AudioSnippetDataset("../testdata")
    x = dataset[0]
    print(x.shape)