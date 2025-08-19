import os
import time
import yaml
import torch
import argparse
from sklearn.metrics import accuracy_score
from src.utils import *
from src.data_loader import get_data_loaders
from src.models.cnn_model_LayerWiseQuant import M5Modular, PTQM5Modular, QATM5Modular

def evaluate_model(model, test_loader, device, checkpoint_path, logger, N=500):
    """Evaluate model performance: size, accuracy, and inference speed."""
    # Model Size
    model_size = os.path.getsize(checkpoint_path)
    logger.info(f"1. Model Size: {model_size / 1024:.2f} KB")

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
    logger.info(f"2. Accuracy: {accuracy * 100:.4f}%")

    # Inference Speed
    batch = next(iter(test_loader))[0].to(device)
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(batch)

    # Timing
    start = time.time()
    with torch.no_grad():
        for _ in range(N):
            _ = model(batch)
    end = time.time()
    avg_time_ms = (end - start) / N * 1000
    logger.info(f"3. Average inference time: {avg_time_ms:.2f} ms") 

def test(model, test_loader, device = 'cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cpu()
            output = model(data)
            preds = output.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy().flatten())
    accuracy = accuracy_score(all_labels, all_preds)
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
    if model_type == "cnn_qat":
        model = QATM5Modular(
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
        model.load_state_dict(torch.load(args.checkpoint))
        model.to(device) # cpu

    elif model_type == "cnn_ptq":
        model = PTQM5Modular(
            n_input=model_params["n_input"],
            n_output=model_params["n_output"],
            stride=model_params["stride"],
            n_channel=model_params["n_channel"],
            conv_kernel_sizes=model_params["conv_kernel_sizes"]
        ).to('cpu')
        model.eval()
        model.fuse_model()
        model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        torch.ao.quantization.prepare(model, inplace=True)
        model = torch.ao.quantization.convert(model, inplace=False)

        model.load_state_dict(torch.load(args.checkpoint))
        model.to('cpu')

    elif model_type == "cnn_fp32":
        model = M5Modular(
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
    train_loader, test_loader, _ = get_data_loaders(config["data"])

    # Evaluate and log results
    logger.info(f"Evaluating model: {args.checkpoint}")
    evaluate_model(model, train_loader, device, args.checkpoint, logger)
    evaluate_model(model, test_loader, device, args.checkpoint, logger)