import yaml
from src.utils import *
from src.data_loader import get_data_loaders
from src.models.cnn_model import M5, QATM5, PTQM5
from src.train import train_model
import argparse

def main(args):
    logger = setup_logging()
    device = get_device()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Get data
    train_loader, test_loader, validate_loader = get_data_loaders(config["data"])
    
    # Initialize model
    model_config = config["model"]["base_cnn"]

    # if config["model_type"] == "cnn_qat":
    #     model = QATM5(n_input=model_config["n_input"],
    #                   n_output=model_config["n_output"],
    #                   stride=model_config["stride"],
    #                   n_channel=model_config["n_channel"],
    #                   conv_kernel_sizes=model_config["conv_kernel_sizes"]).to(device)
        
    #     # DEBUG
    #     model.eval()
    #     model.fuse_model()
    #     model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

    #     # TODO: Add separate qat process.
    #     model.train()
    #     torch.ao.quantization.prepare_qat(model, inplace=True)

    if config["model_type"] == "cnn_qat":   # QAT from a fp checkpoint.

        assert "pretrained_path" in model_config, "Pretrained model must be provided for QAT from a fp checkpoint."
        fp32_checkpoint = torch.load(model_config["pretrained_path"])
        
        # print("Checkpoint dict keys:", fp32_checkpoint.keys())
        # odict_keys([
        # 'conv1.weight', 'conv1.bias', 
        # 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 
        # 'conv2.weight', 'conv2.bias', 
        # 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var', 'bn2.num_batches_tracked', 
        # 'conv3.weight', 'conv3.bias', 
        # 'bn3.weight', 'bn3.bias', 'bn3.running_mean', 'bn3.running_var', 'bn3.num_batches_tracked', 
        # 'conv4.weight', 'conv4.bias', 
        # 'bn4.weight', 'bn4.bias', 'bn4.running_mean', 'bn4.running_var', 'bn4.num_batches_tracked', 
        # 'fc1.weight', 'fc1.bias'])

        model = QATM5(n_input=model_config["n_input"],
                      n_output=model_config["n_output"],
                      stride=model_config["stride"],
                      n_channel=model_config["n_channel"],
                      conv_kernel_sizes=model_config["conv_kernel_sizes"]).to(device)
        
        missing_keys, unexpected_keys = model.load_state_dict(fp32_checkpoint, strict=False)
        # print("Missing:", missing_keys)
        # print("Unexpected:", unexpected_keys)

        with torch.no_grad():
            model.fc1.weight.copy_(fp32_checkpoint["fc1.weight"])
            model.fc1.bias.copy_(fp32_checkpoint["fc1.bias"])

        # model.eval() #?
        model.fuse_model()
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

        model.train()
        torch.ao.quantization.prepare_qat(model, inplace=True)
        train_model(model, train_loader, test_loader, config["train"], device)

    elif config["model_type"] == "cnn_fp32":
        model = M5(n_input=model_config["n_input"],
                   n_output=model_config["n_output"],
                   stride=model_config["stride"],
                   n_channel=model_config["n_channel"],
                   conv_kernel_sizes=model_config["conv_kernel_sizes"]).to(device)
        
        # fp32 model Train
        logger.info("Training fp32 model...")
        logger.info(f"Model parameters: {count_parameters(model)}")
        logger.info(f"Model type: {config['model_type']}")
        logger.info(f"Device: {device}")
        train_model(model, train_loader, test_loader, config["train"], device)
    
    elif config["model_type"] == "cnn_ptq":
        # Load FP32 model
        model_fp32 = PTQM5(n_input=model_config["n_input"],
                   n_output=model_config["n_output"],
                   stride=model_config["stride"],
                   n_channel=model_config["n_channel"],
                   conv_kernel_sizes=model_config["conv_kernel_sizes"]).to('cpu')
        model_fp32.eval()
        assert "pretrained_path" in model_config, "Pretrained model must be provided for PTQ."
        model_fp32.load_state_dict(torch.load(model_config["pretrained_path"]))
        model_fp32.fuse_model()
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        torch.ao.quantization.prepare(model_fp32, inplace=True)
        
        # Calibrate model - use validation set
        with torch.inference_mode():
            for data, _ in validate_loader:
                data = data.to("cpu")
                model_fp32(data)
        
        # # DEBUG
        # logger.info("Post Training Quantizing model...")
        # logger.info(f"fp32 model: {model_fp32}")

        # Convert to PTQ model
        model = torch.ao.quantization.convert(model_fp32, inplace=False)

        # # DEBUG
        # logger.info(f"PTQ model: {model}")

        
    elif config["model_type"] == "wav2vec2_fp16":
        pass
        # model = # TODO: Load model and use data loader from data_loader.py

    
    # Save models
    if config["model_type"] == "cnn_qat":
        model.to("cpu")
        logger.info("Quantizing model...")
        model.eval()
        torch.ao.quantization.convert(model, inplace=True)

    # # DEBUG: Check if model is quantized
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.ao.nn.quantized.Conv1d):
    #         print(f"Quantized Conv1d: {name}")
    #         print(f"Weight dtype: {module.weight().dtype}")  # Should be torch.qint8
    #     elif isinstance(module, torch.ao.nn.quantized.Linear):
    #         print(f"Quantized Linear: {name}")
    #         print(f"Weight dtype: {module.weight().dtype}")  # Should be torch.qint8
         
    torch.save(model.state_dict(), f"./models/{config['model_type']}_model.pth")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('--config', default="./configs/cnn_qat.yaml", type=str)
    args = parser.parse_args()

    main(args)
