import yaml
from src.utils import *
from src.data_loader import get_data_loaders
from src.models.cnn_model_LayerWiseQuant import QATM5Modular
from src.FP_export import compare_model_outputs
from src.train import set_seed, train_model
import argparse

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX format.")
    parser.add_argument("--config", help="QAT configs", required=True, type=str)
    parser.add_argument("--checkpoint", help="Path to the FP32 checkpoint", required=True, type=str)
    parser.add_argument("--PretrainedQATCheckpoint", help="Path to the QAT checkpoint", required=False, type=str)
    parser.add_argument("--output", help="Path to the output ONNX file", required=True, type=str)
    args = parser.parse_args()

    logger = setup_logging()
    logger.info(f"Loaded config: {args.config}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_loader, test_loader, _ = get_data_loaders(config["data"])

    model_config = config["model"]["base_cnn"]
    model = QATM5Modular(
        n_input=model_config["n_input"],
        n_output=model_config["n_output"],
        stride=model_config["stride"],
        n_channel=model_config["n_channel"],
        conv_kernel_sizes=model_config["conv_kernel_sizes"]
    ).to("cpu")

    if args.PretrainedQATCheckpoint:

        # Fuse, prepare, and convert the model for QAT checkpoint
        model.eval()
        model.fuse_model()
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        model.train()
        torch.ao.quantization.prepare_qat(model, inplace=True)
        model.eval()
        model = torch.ao.quantization.convert(model, inplace=False)

        # Load the QAT checkpoint if provided
        pretrained_qat_checkpoint = torch.load(args.PretrainedQATCheckpoint)
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_qat_checkpoint, strict=False)
        if len(missing_keys) > 0: logger.error(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0: logger.error(f"Unexpected keys: {unexpected_keys}")

    else:
        # Load the FP32 checkpoint and QAT the model
        fp32_checkpoint = torch.load(args.checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(fp32_checkpoint, strict=False)
        if len(missing_keys) > 0: logger.error(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0: logger.error(f"Unexpected keys: {unexpected_keys}")

        model.eval()
        model.fuse_model()
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86') # Per-channel quantization?
        model.train()
        torch.ao.quantization.prepare_qat(model, inplace=True)
        train_model(model, train_loader, test_loader, config["train"], "cpu")
        model.eval()
        torch.ao.quantization.convert(model, inplace=True)
        # convert() should removes all observers and fake quant ops, 
        # and inserts actual quantized weights and int8 computations.
        torch.save(model.state_dict(), f"./models/{config['model_type']}_for_ONNX.pth")

    # Export the model to ONNX format
    input_tensor = next(iter(train_loader))[0].to('cpu')
    dynamic_axes_0 = {"input": {0: "batchsize"}, "output": {0: "batchsize"}}

    try:
        torch.onnx.export(
            model,
            input_tensor,
            args.output, 
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes_0,
        )
        logger.info(f"ONNX export successful: {args.output}")
    except Exception as e:
        logger.error(f"Error exporting model to ONNX: {e}")

    # Compare outputs between PyTorch and ONNX models
    compare_model_outputs(
        model, 
        args.output, 
        test_loader, 
        tolerance=(0.1, 0.1), 
        precision_check=False
    )