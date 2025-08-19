import yaml
from src.utils import *
from src.data_loader import get_data_loaders
from src.models.cnn_model import M5, QATM5, PTQM5
from src.models.cnn_model_LayerWiseQuant import M5Modular, PTQM5Modular, PTQM5_LayerWiseQuant, QATM5Modular, QATM5_LayerWiseQuant
from src.train import set_seed, train_model
import argparse
import json

def main(args):
    logger = setup_logging()
    device = "cpu"
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Get data
    train_loader, test_loader, validate_loader = get_data_loaders(config["data"])
    logger.info(f"Dataloader prepared.")
    
    # Initialize model
    model_config = config["model"]["base_cnn"]


    if config["model_type"] == "cnn_qat":   # QAT from a fp checkpoint.

        assert "pretrained_path" in model_config, "Pretrained model must be provided for QAT from a fp checkpoint."
        fp32_checkpoint = torch.load(model_config["pretrained_path"])
        
        model = QATM5Modular(n_input=model_config["n_input"],
                      n_output=model_config["n_output"],
                      stride=model_config["stride"],
                      n_channel=model_config["n_channel"],
                      conv_kernel_sizes=model_config["conv_kernel_sizes"]).to(device)
        
        missing_keys, unexpected_keys = model.load_state_dict(fp32_checkpoint, strict=False)
        print("Missing:", missing_keys) if len(missing_keys) > 0 else None
        print("Unexpected:", unexpected_keys) if len(unexpected_keys) > 0 else None

        # model.eval() #?
        model.fuse_model()
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

        model.train()
        torch.ao.quantization.prepare_qat(model, inplace=True)
        history = train_model(model, train_loader, test_loader, config["train"], device)
        
        with open('history.json', 'w') as fp:
            json.dump(history, fp)
        # Quantize and Save model
        model.to("cpu")
        logger.info("Quantizing model...")
        model.eval()
        torch.ao.quantization.convert(model, inplace=True)
        torch.save(model.state_dict(), config["model"]["output"])


    elif config["model_type"] == "cnn_fp32":
        model = M5Modular(
            n_input=model_config["n_input"],
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

        # Save model
        torch.save(model.state_dict(), f"./models/{config['model_type']}_model.pth")

    
    elif config["model_type"] == "cnn_ptq_static":
        set_seed(config["model"]["base_cnn"]["seed"])
        # Load FP32 model
        model_fp32 = PTQM5Modular(n_input=model_config["n_input"],
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

        # Convert to PTQ model
        model = torch.ao.quantization.convert(model_fp32, inplace=False)

        # Save model
        torch.save(model.state_dict(), config["model"]["output"])

        
    elif config["model_type"] == "cnn_ptq_LayerWiseQuant":
        # Initialize layer-wise quantized models and load FP32 checkpoint
        for i in config["model"]["quantization"]:
            model_fp32 = PTQM5_LayerWiseQuant(
                quantized_block_idx = i,
                n_input=model_config["n_input"],
                n_output=model_config["n_output"],
                stride=model_config["stride"],
                n_channel=model_config["n_channel"],
                conv_kernel_sizes=model_config["conv_kernel_sizes"], 
                ).to('cpu')
            model_fp32.eval()
            assert "pretrained_path" in model_config, "Pretrained model must be provided for PTQ."
            model_fp32.load_state_dict(torch.load(model_config["pretrained_path"]))
            model_fp32.fuse_model()
            # Set qconfig only on the target block and stubs
            qconfig = torch.ao.quantization.get_default_qconfig('x86')
            model_fp32.set_qconfig_for_layerwise(qconfig)
            torch.ao.quantization.prepare(model_fp32, inplace=True)
        
            # Calibrate model - use validation set
            logger.info(f"Quantizing Layer {i} ...")
            with torch.inference_mode():
                for data, _ in validate_loader:
                    data = data.to("cpu")
                    model_fp32(data)

            # Convert to PTQ model
            model = torch.ao.quantization.convert(model_fp32, inplace=False)

            # Save model
            torch.save(model.state_dict(), f"./models/{config['model_type']}_q{i}_model.pth")


    elif config["model_type"] == "cnn_qat_LayerWiseQuant":
        # Initialize layer-wise quantized models and load ptq layerWise checkpoint
        for i in config["model"]["quantization"]:
            model = QATM5_LayerWiseQuant(
                quantized_block_idx = i,
                n_input=model_config["n_input"],
                n_output=model_config["n_output"],
                stride=model_config["stride"],
                n_channel=model_config["n_channel"],
                conv_kernel_sizes=model_config["conv_kernel_sizes"], 
                ).to('cpu')
            
            model.eval()
            model.load_state_dict(torch.load(model_config["pretrained_path"]))
            model.fuse_model()

            # Set qconfig only on the target block and stubs
            qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
            model.set_qconfig_for_layerwise(qconfig)
            
            # Quantize & Save the model
            model.train()
            logger.info(f"Quantizing Layer {i} ...")
            torch.ao.quantization.prepare_qat(model, inplace=True)
            print(device)
            train_model(model, train_loader, test_loader, config["train"], "cpu")
            
            model.to("cpu")
            model.eval()
            torch.ao.quantization.convert(model, inplace=True)
            torch.save(model.state_dict(), f"./models/{config['model_type']}_q{i}_model.pth")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('--config', default="./configs/cnn_qat.yaml", type=str)
    args = parser.parse_args()

    main(args)
