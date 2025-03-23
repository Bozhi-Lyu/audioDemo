import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def train_model(model, train_loader, test_loader, config, device):
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
        
        # Logging uses module-level logger
        if batch_idx % config["log_interval"] == 0:
            logger.info(f"Train Epoch: {epoch} [...]")
            
    return epoch_losses


def test_model(model, test_loader, epoch, device, pbar):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(1)

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)