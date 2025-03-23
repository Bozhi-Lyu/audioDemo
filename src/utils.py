import logging
import sys
import torch

def setup_logging():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")