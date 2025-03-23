import os
import time
import yaml
import torch
from torchsummary import summary
from src.utils import *
from src.data_loader import get_data_loaders
from src.models.cnn_model import M5, QATM5
from src.train import number_of_correct, get_likely_index

def compare_models(quant_model, fp32_model, test_loader, device):

    # Model Size Comparison
    quant_size = os.path.getsize("./models/qat_model.pth")
    fp32_size = os.path.getsize("./models/fp32_model.pth")
    logger.info(f"Quantized Model Size: {quant_size/1024:.2f} KB")
    logger.info(f"FP32 Model Size: {fp32_size/1024:.2f} KB")
    logger.info(f"Size Reduction: {1 - quant_size/fp32_size:.1%}")

    # Inference Speed Comparison
    dummy_input = next(iter(test_loader))[0][0].unsqueeze(0).to(device)
    
    # Warmup
    for _ in range(10):
        _ = quant_model(dummy_input)
        _ = fp32_model(dummy_input)
    
    # Quantized timing
    start = time.time()
    for _ in range(100):
        _ = quant_model(dummy_input)
    quant_time = (time.time() - start)/100
    
    # FP32 timing
    start = time.time()
    for _ in range(100):
        _ = fp32_model(dummy_input)
    fp32_time = (time.time() - start)/100
    
    print(f"\nInference Time (per sample):")
    print(f"Quantized: {quant_time*1000:.2f} ms")
    print(f"FP32: {fp32_time*1000:.2f} ms")
    print(f"Speedup: {fp32_time/quant_time:.1f}x")

    # Memory Usage (if using CUDA)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _ = quant_model(dummy_input)
        quant_mem = torch.cuda.max_memory_allocated()
        
        torch.cuda.reset_peak_memory_stats()
        _ = fp32_model(dummy_input)
        fp32_mem = torch.cuda.max_memory_allocated()
        
        print(f"\nGPU Memory Usage:")
        print(f"Quantized: {quant_mem/1024**2:.2f} MB")
        print(f"FP32: {fp32_mem/1024**2:.2f} MB")

    # Accuracy Comparison
    quant_acc = test(quant_model, test_loader)
    fp32_acc = test(fp32_model, test_loader)
    print(f"\nAccuracy Drop: {fp32_acc - quant_acc:.2f}%")

def test(model, test_loader):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)
    accuracy = 100. * correct / len(test_loader.dataset)
    logger.info(f"\nTest Epoch: {epoch}\tAccuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    device = get_device()
    logger = setup_logging()

    with open("./configs/cnn_qat.yaml") as f:
        config = yaml.safe_load(f)
    qat_model_config = config["model"]["base_cnn"]
    quant_model = QATM5(n_input=qat_model_config["n_input"],
                      n_output=qat_model_config["n_output"],
                      stride=qat_model_config["stride"],
                      n_channel=qat_model_config["n_channel"],
                      conv_kernel_sizes=qat_model_config["conv_kernel_sizes"]).to(device)
    quant_model.eval()
    quant_model.fuse_model()
    quant_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
    quant_model.train()
    torch.ao.quantization.prepare_qat(quant_model, inplace=True)

    quant_model.to("cpu")
    quant_model = torch.ao.quantization.convert(quant_model, inplace=False)
    quant_model.load_state_dict(torch.load("./models/qat_model.pth"))
    

    with open("./configs/cnn_fp32.yaml") as f:
        config = yaml.safe_load(f)
    fp32_model_config = config["model"]["base_cnn"]
    fp32_model = M5(n_input=fp32_model_config["n_input"],
                   n_output=fp32_model_config["n_output"],
                   stride=fp32_model_config["stride"],
                   n_channel=fp32_model_config["n_channel"],
                   conv_kernel_sizes=fp32_model_config["conv_kernel_sizes"])
    fp32_model.load_state_dict(torch.load("./models/fp32_model.pth"))
    fp32_model.to(device)

    logger.info("Ensure the quantized model is properly saved and loaded: ", torch.load("./models/qat_model.pth").keys())

    train_loader, test_loader = get_data_loaders(config["data"])
    compare_models(quant_model, fp32_model, test_loader, device)