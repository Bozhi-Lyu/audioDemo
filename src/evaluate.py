import os
import time
import yaml
import torch
import argparse
from src.utils import *
from src.data_loader import get_data_loaders
from src.models.cnn_model import M5, QATM5
from src.train import number_of_correct, get_likely_index

def evaluate_model(model, test_loader, device, checkpoint_path, logger):
    """Evaluate model performance."""
    # Model Size
    model_size = os.path.getsize(checkpoint_path)
    logger.info(f"Model Size: {model_size / 1024:.2f} KB")

    # Inference Speed
    element = next(iter(test_loader))
    logger.info(f"Element Shape: {element[0].shape}")
    # dummy_input = next(iter(test_loader))[0][0].unsqueeze(0).to(device)
    # DEBUG
    # logger.info(f"Dummy Input Shape: {dummy_input.shape}")
    # logger.info(f"Test loader: {test_loader}")
    # logger.info(f"Test loader length: {len(test_loader)}")
    # logger.info(f"Test loader dataset: {len(test_loader.dataset)}")
    # logger.info(f"Test loader batch size: {test_loader.batch_size}")
    model.to(device)
    # print("Model Device now:", device) # cpu
    
    # # Warmup # Here's the bug! The model is not being warmed up
    # for _ in range(10):
    #     _ = model(dummy_input)
    
    # # Quantization timing
    # start = time.time()
    # for _ in range(100):
    #     _ = model(dummy_input)
    # inference_time = (time.time() - start)/100
    # logger.info(f"Inference Time (per sample): {inference_time * 1000:.2f} ms")
    
    # # Memory Usage (if using CUDA)
    # if torch.cuda.is_available():
    #     torch.cuda.reset_peak_memory_stats()
    #     _ = model(dummy_input)
    #     memory_usage = torch.cuda.max_memory_allocated()
    #     logger.info(f"GPU Memory Usage: {memory_usage / 1024**2:.2f} MB")

    # Accuracy Comparison
    accuracy = test(model, test_loader)
    logger.info(f"Accuracy: {accuracy:.2f}%")

@profile
def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate a trained model.')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    # parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = 'cpu'
    logger = setup_logging()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model_type = config["model_type"]
    model_params = config["model"]["base_cnn"]

    # Initialize model
    if model_type == "qat":
        print("QAT Model")
        model = QATM5(
            n_input=model_params["n_input"],
            n_output=model_params["n_output"],
            stride=model_params["stride"],
            n_channel=model_params["n_channel"],
            conv_kernel_sizes=model_params["conv_kernel_sizes"]
        )
        # Fuse and prepare for quantization
        model.eval()
        model.fuse_model()
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

        model.train()
        torch.ao.quantization.prepare_qat(model, inplace=True)

        # Convert to quantized model
        model.eval()
        model = torch.ao.quantization.convert(model, inplace=False)

        # Load checkpoint
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        model.to(device) # cpu

    elif model_type == "fp32":
        model = M5(
            n_input=model_params["n_input"],
            n_output=model_params["n_output"],
            stride=model_params["stride"],
            n_channel=model_params["n_channel"],
            conv_kernel_sizes=model_params["conv_kernel_sizes"]
        )
        model.load_state_dict(torch.load(args.checkpoint))
        model.to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


    model.eval()

    # Load test data
    train_loader, test_loader = get_data_loaders(config["data"])

    # Evaluate and log results
    logger.info(f"Evaluating model: {args.checkpoint}")
    evaluate_model(model, test_loader, device, args.checkpoint, logger)