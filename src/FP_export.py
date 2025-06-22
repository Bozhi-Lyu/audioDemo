import yaml
from src.utils import *
from src.data_loader import get_data_loaders
from src.models.cnn_model_LayerWiseQuant import M5Modular
import argparse

import torch

def main(args):

    logger = setup_logging()
    device = "cpu"

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    train_loader, _, _ = get_data_loaders(config["data"])
    dummy_input = next(iter(train_loader))[0].to(device)
    
    model_config = config["model"]["base_cnn"]  

    if config["model_type"] == "cnn_fp32":
        model = M5Modular(
            n_input=model_config["n_input"],
            n_output=model_config["n_output"],
            stride=model_config["stride"],
            n_channel=model_config["n_channel"],
            conv_kernel_sizes=model_config["conv_kernel_sizes"]
        ).to("cpu")
        
        try:
            model.load_state_dict(torch.load(args.checkpoint))
            model.eval()
                    
            dynamic_axes_0 = { 
            'input' : {0: 'batchsize'}, 
            'output' : {0: 'batchsize'}
            }

            torch.onnx.export(
                model, 
                dummy_input, 
                args.output,
                opset_version=13,
                input_names=['input'], 
                output_names=['output'],
                dynamic_axes=dynamic_axes_0
            )
            logger.info("ONNX export successful.")
        except Exception as e:
            logger.error(f"Error exporting model to ONNX: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX format.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    main(args)