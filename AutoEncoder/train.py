# training script
import argparse
import yaml
import logging

from torch import GradScaler

import wandb
from torch.utils.data import DataLoader, random_split
from Dataset import AudioSnippetDataset
from Model import SimpleConvAutoencoder
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(prog="Training Arguments",
                                     description="Please enter necessary training arguments",
                                     epilog="Use -h for help")
    parser.add_argument('--config',
                        type=str,
                        default='config.yaml',
                        help='Path to the YAML configuration file (default: config.yaml)')
    return parser.parse_args()

def yaml_parser(training_args):
    try:
        with open(training_args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        logger.info(f"Found configuration file: {training_args.config}")

        return yaml_config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {args.config}")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration file {args.config}: {e}")
        exit(1)

def training_pipeline(training_conf):
    with wandb.init(project=training_conf.get("wandb_project"), config=training_conf, mode="offline",):
        config = wandb.config

        # --- Determine and store effective device ---
        if config.device == "auto":
            effective_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            effective_device = config.device
        # Use allow_val_change=True if you need to update after init
        wandb.config.update({"effective_device": effective_device}, allow_val_change=True)

        model, train_loader, val_loader, criterion, optimizer, scheduler, scaler = init_training(config)
        logger.info(f"YEEES")

def init_training(config):
    # reproducibility
    set_seed(config["seed"])
    g = torch.Generator().manual_seed(config["seed"])

    full_dataset = AudioSnippetDataset(dataset_directory=config["dataset_dir"],
                                  snippet_length=config["snippet_length"],
                                  target_sr=config["target_sr"],
                                  recursive=config["recursive"],
                                  normalize_peak=config["normalize_audio"])
    logger.info(f"Successfully initialized dataset with {len(full_dataset)} audio snippets.")

    # Split dataset
    val_split = config["val_split"]
    if not 0 <= val_split < 1:
        raise ValueError("val_split must be between 0 and 1.")
    num_samples = len(full_dataset)
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size

    if train_size == 0 or val_size == 0:
         raise ValueError(f"Dataset size ({num_samples}) is too small for the validation split ({val_split}). Need at least one sample in both train and validation sets.")

    logger.info(f"Splitting dataset: {train_size} training samples, {val_size} validation samples.")

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        worker_init_fn=seed_worker if config['num_workers'] > 0 else None,
        pin_memory=True if config['effective_device'] == 'cuda' else False, # Optimize CUDA data transfer
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        worker_init_fn=seed_worker if config['num_workers'] > 0 else None,
        pin_memory=True if config['effective_device'] == 'cuda' else False,
        generator=g
    )

    # --- Model Initialization ---
    logging.info("Initializing model...")
    model = SimpleConvAutoencoder(snippet_length=config['snippet_length']).to(config['effective_device'])
    # logging.info(f"Model:\n{model}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of trainable parameters: {num_params:,}")

    # --- Loss Function ---
    criterion = nn.MSELoss()

    # Optimizer setup
    optimizer = optim.AdamW(
        [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in ['bias', 'norm'])],
             'weight_decay': float(config['weight_decay'])},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in ['bias', 'norm'])],
             'weight_decay': 0.0}
        ],
        lr=float(config['learning_rate']),  # Add float() here
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(step / config["warmup_steps"], 1.0) * max(
            0.0, float(config["training_steps"] - step) / float(
            max(1, config["training_steps"] - config["warmup_steps"]))
        )
    )

    scaler = GradScaler()

    return model, train_loader, val_loader, criterion, optimizer, scheduler, scaler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    """Sets the seed for a dataloader worker."""
    # Get the seed set by torch.manual_seed() and use it to seed numpy and random for this worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)

def main(training_args):
    yaml_conf = yaml_parser(training_args)
    training_pipeline(yaml_conf)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)