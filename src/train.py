import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
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

    history = {
        "train_loss": [],
        "test_loss": [],
        "test_acc": []
    }

    log_interval = config["log_interval"]

    with tqdm(total=config["n_epoch"] * len(train_loader)) as pbar:
        
        for epoch in range(1, config["n_epoch"] + 1):
            epoch_losses = []
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                output = model(data)

                loss = F.nll_loss(output.squeeze(), target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                pbar.update(1)

                # log every 100 batches
                if (batch_idx + 1) % log_interval == 0:
                    avg_recent_loss = sum(epoch_losses[-log_interval:]) / log_interval
                    history["train_loss"].append(avg_recent_loss)
                    logger.info(f"Epoch {epoch}, Batch {batch_idx+1}: Train Loss = {avg_recent_loss:.4f}")

            # compute test metrics at the end of epoch
            accuracy, avg_test_loss = test(model, test_loader, device)
            history["test_loss"].append(avg_test_loss)
            history["test_acc"].append(accuracy)
            logger.info(
                f"Epoch {epoch} completed: "
                f"Test Loss = {avg_test_loss:.4f}, "
                f"Test Acc = {accuracy*100:.2f}%"
            )
            scheduler.step()

        return history

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


def test(model, test_loader, device = 'cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.cpu()
            output = model(data)

            loss = F.nll_loss(output.squeeze(), target, reduction='sum')
            total_loss += loss.item()
            num_batches += data.size(0)

            preds = output.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy().flatten())
    avg_loss = total_loss / num_batches    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, avg_loss

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)