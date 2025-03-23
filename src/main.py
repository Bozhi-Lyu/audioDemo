import yaml
from src.utils import *
from src.data_loader import get_data_loaders
from src.models.cnn_model import M5, QATM5
from src.train import train_model
import argparse

def main(args):
    logger = setup_logging()
    device = get_device()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Get data
    train_loader, test_loader = get_data_loaders(config["data"])
    
    # Initialize model
    model_config = config["model"]["base_cnn"]
    if config["model_type"] == "qat":
        model = QATM5(n_input=model_config["n_input"],
                      n_output=model_config["n_output"],
                      stride=model_config["stride"],
                      n_channel=model_config["n_channel"],
                      conv_kernel_sizes=model_config["conv_kernel_sizes"]).to(device)
        
        # DEBUG
        model.eval()
        model.fuse_model()
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

        # DEBUG
        # # Retrieve the QConfig
        qconfig = model.qconfig

        # # Instantiate the observers
        activation_observer = qconfig.activation()
        weight_observer = qconfig.weight()

        # # Print the quantization schemes
        # logger.info("Activation Observer qscheme:", activation_observer.qscheme) #  (torch.per_tensor_affine,)
        # logger.info("Weight Observer qscheme:", weight_observer.qscheme) #  (torch.per_channel_symmetric,)

        # # Check quantization ranges
        # logger.info("\nActivation Observer quant_min/max:", activation_observer.quant_min, activation_observer.quant_max) #  (0, 127)
        # logger.info("Weight Observer quant_min/max:", weight_observer.quant_min, weight_observer.quant_max) # (-128, 127)
        
        model.train()
        torch.ao.quantization.prepare_qat(model, inplace=True)

    else:
        model = M5(n_input=model_config["n_input"],
                   n_output=model_config["n_output"],
                   stride=model_config["stride"],
                   n_channel=model_config["n_channel"],
                   conv_kernel_sizes=model_config["conv_kernel_sizes"]).to(device)
    
    # Train
    logger.info("Training model...")
    logger.info(f"Model parameters: {count_parameters(model)}")
    logger.info(f"Model type: {config['model_type']}")
    logger.info(f"Device: {device}")
    train_model(model, train_loader, test_loader, config["train"], device)
    
    # Save models
    if config["model_type"] == "qat":
        model.to("cpu")
        logger.info("Quantizing model...")
        model.eval()
        torch.ao.quantization.convert(model, inplace=True)

    # DEBUG: Check if model is quantized
    for name, module in model.named_modules():
        if isinstance(module, torch.ao.nn.quantized.Conv1d):
            print(f"Quantized Conv1d: {name}")
            print(f"Weight dtype: {module.weight().dtype}")  # Should be torch.qint8
        elif isinstance(module, torch.ao.nn.quantized.Linear):
            print(f"Quantized Linear: {name}")
            print(f"Weight dtype: {module.weight().dtype}")  # Should be torch.qint8
         
    torch.save(model.state_dict(), f"./models/{config['model_type']}_model.pth")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('--config', default="./configs/cnn_qat.yaml", type=str)
    args = parser.parse_args()

    main(args)