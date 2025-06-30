import yaml
from src.utils import *
from src.data_loader import get_data_loaders
from src.models.cnn_model_LayerWiseQuant import QATM5Modular
from src.FP_export import compare_model_outputs
from src.train import set_seed, train_model
import argparse

import torch

def qat_export_model(config, fp32_checkpoint, output_path, train_loader, test_loader):
    logger = setup_logging()
    input_tensor = next(iter(train_loader))[0].to('cpu')
    dynamic_axes_0 = {"input": {0: "batchsize"}, "output": {0: "batchsize"}}
    model_config = config["model"]["base_cnn"]
    

    model = QATM5Modular(
        n_input=model_config["n_input"],
        n_output=model_config["n_output"],
        stride=model_config["stride"],
        n_channel=model_config["n_channel"],
        conv_kernel_sizes=model_config["conv_kernel_sizes"]
    ).to("cpu")
    missing_keys, unexpected_keys = model.load_state_dict(fp32_checkpoint, strict=False)
    logger.error("Missing:", missing_keys) if len(missing_keys) > 0 else None
    logger.error("Unexpected:", unexpected_keys) if len(unexpected_keys) > 0 else None

    model.fuse_model()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

    model.train()
    torch.ao.quantization.prepare_qat(model, inplace=True)
    train_model(model, train_loader, test_loader, config["train"], "cpu")
    model.eval()
    torch.ao.quantization.convert(model, inplace=True)
    # convert() should removes all observers and fake quant ops, 
    # and inserts actual quantized weights and int8 computations.

    try:
        torch.onnx.export(
            model,
            input_tensor,
            output_path, 
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes_0,
        )
        logger.info(f"ONNX export successful: {output_path}")
    except Exception as e:
        logger.error(f"Error exporting model to ONNX: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX format.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", help="Path to the FP32 checkpoint", required=True, type=str)
    parser.add_argument("--output", help="Path to the output ONNX file", required=True, type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_loader, test_loader, _ = get_data_loaders(config["data"])
    fp32_checkpoint = torch.load(args.checkpoint)

    qat_export_model(config, fp32_checkpoint, args.output, train_loader, test_loader)