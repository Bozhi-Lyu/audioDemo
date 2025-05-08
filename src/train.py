import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import logging
import sys
import numpy as np
import random
import os

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, test_loader, config, device):
    set_seed(config["seed"])
    optimizer = optim.Adam(model.parameters(), 
                         lr=config["lr"], 
                         weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config["step_size"], 
        gamma=config["gamma"]
    )

    total_batches = len(train_loader) + len(test_loader)
    
    with tqdm(total=config["n_epoch"] * total_batches) as pbar:
        losses = []
        
        for epoch in range(1, config["n_epoch"] + 1):
            epoch_losses = train_epoch(
                model, train_loader, optimizer, 
                epoch, config, device, pbar
            )
            losses.extend(epoch_losses)
            
            test_model(model, test_loader, epoch, device, pbar)
            scheduler.step()
            
        return losses

def train_epoch(model, train_loader, optimizer, epoch, config, device, pbar):
    model.train()
    epoch_losses = []

    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update(1)
        epoch_losses.append(loss.item())
        
        # # Logging uses module-level logger
        # if batch_idx % config["log_interval"] == 0:
        #     logger.info(f"Train Epoch: {epoch} [...]")
            
    return epoch_losses


def test_model(model, test_loader, epoch, device, pbar):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:

            data = data.to(device)
            target = target.to(device)
            output = model(data)

            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)
            total += target.size(0)

            # update progress bar
            pbar.update(1)
    
    accuracy = 100. * correct / total if total > 0 else 0

    pbar.set_description(f"Epoch {epoch} [Test Accuracy: {accuracy:.2f}%]")

    logger.info(f"Test Epoch: {epoch} Accuracy: {accuracy:.2f}% ({correct}/{total})")

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)