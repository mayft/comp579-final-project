import os
import math
import logging
import argparse
import mlflow
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

# from models.eswm import ESWM_T
# from models.epn import EPN
# from envs.hex_grid import generate_hex_grid_batch

def setup():
    # Change 'nccl' to 'gloo' for local testing
    dist.init_process_group(backend="nccl")

def cleanup():
    dist.destroy_process_group()

def setup_logger(global_rank):
    """Sets up a text logger that writes to a file (only on the master node)."""
    if global_rank == 0:
        os.makedirs('./outputs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler('./outputs/training_status.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    return None

def train(args):
    # Distributed Setup
    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Comment out CUDA lines and force device="cpu" for local testing
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    logger = setup_logger(global_rank)
    if global_rank == 0:
        logger.info(f"Starting distributed training job on {world_size} nodes.")
        mlflow.start_run()

    # model = ESWM_T(arguments).to(device)
    model = torch.nn.Linear(10, 10).to(device) 
    
    # Dynamic Learning Rate Scaling
    base_lr = 1e-4 
    
    # Square Root Scaling Rule 
    scaled_lr = base_lr * math.sqrt(world_size)
    
    if global_rank == 0:
        logger.info(f"Base LR: {base_lr} | Scaled LR for {world_size} GPUs: {scaled_lr:.6f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr)
    scaler = GradScaler() 

    # Checkpointing Setup
    checkpoint_dir = args.checkpoint_dir 
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    
    start_epoch = 0
    start_iteration = 0
    global_step = 0 

    if os.path.exists(checkpoint_path):
        if global_rank == 0:
            logger.info(f"Found checkpoint at {checkpoint_path}. Resuming training...")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration'] + 1
        global_step = checkpoint.get('global_step', 0)
        
    elif global_rank == 0:
        logger.info("No checkpoint found. Starting weights from scratch.")
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Wrap the model in DDP *after* loading any base model weights
    ddp_model = DDP(model, device_ids=[local_rank])

    # Training Loop
    epochs = 100
    iterations_per_epoch = 1000
    save_every_n_steps = 250 
    log_every_n_steps = 10 
    
    for epoch in range(start_epoch, epochs):
        ddp_model.train()
        
        current_start_iter = start_iteration if epoch == start_epoch else 0
        
        if global_rank == 0:
            logger.info(f"--- Starting Epoch {epoch} ---")

        for i in range(current_start_iter, iterations_per_epoch):
            
            # inputs, targets = generate_hex_grid_batch(batch_size=128)
            inputs = torch.randn(128, 10).to(device)
            targets = torch.randn(128, 10).to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = ddp_model(inputs)
                loss = torch.nn.functional.mse_loss(outputs, targets)

            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            # Logging
            if global_rank == 0 and i % log_every_n_steps == 0:
                logger.info(f"Epoch: {epoch:03d} | Iter: {i:04d}/{iterations_per_epoch} | Loss: {loss.item():.4f}")
                mlflow.log_metric("train_loss", loss.item(), step=global_step)
                mlflow.log_metric("epoch", epoch, step=global_step)

            if global_rank == 0 and (i % save_every_n_steps == 0 or i == iterations_per_epoch - 1):
                checkpoint = {
                    'epoch': epoch,
                    'iteration': i,
                    'global_step': global_step,
                    'model_state_dict': ddp_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': loss.item()
                }
                
                temp_path = checkpoint_path + ".tmp"
                
                with open(temp_path, "wb") as f:
                    torch.save(checkpoint, f)
                    f.flush()
                    os.fsync(f.fileno()) 
                
                os.replace(temp_path, checkpoint_path)
                logger.info(f"-> Bulletproof cloud checkpoint written at Epoch {epoch}, Iter {i}")

            global_step += 1

    if global_rank == 0:
        logger.info("Training complete! Model reached target epochs.")
        mlflow.end_run()

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to mounted cloud storage")
    args = parser.parse_args()
    
    train(args)