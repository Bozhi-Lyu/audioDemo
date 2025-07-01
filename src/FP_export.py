import yaml
from src.utils import *
from src.data_loader import get_data_loaders
from src.models.cnn_model_LayerWiseQuant import M5Modular
import argparse

import torch
import numpy as np
import onnxruntime as ort

def export_model(model_config, checkpoint, output, test_loader):
    logger = setup_logging()

    input_tensor = next(iter(test_loader))[0].to("cpu")
    dynamic_axes_0 = {"input": {0: "batchsize"}, "output": {0: "batchsize"}}

    if config["model_type"] == "cnn_fp32":
        model = M5Modular(
            n_input=model_config["n_input"],
            n_output=model_config["n_output"],
            stride=model_config["stride"],
            n_channel=model_config["n_channel"],
            conv_kernel_sizes=model_config["conv_kernel_sizes"]
        ).to("cpu")

        try:
            model.load_state_dict(torch.load(checkpoint))
            model.eval()

            torch.onnx.export(
                model,
                input_tensor,
                output,
                opset_version=13,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes_0,
            )
            logger.info("ONNX export successful.")
        except Exception as e:
            logger.error(f"Error exporting model to ONNX: {e}")

        compare_model_outputs(model, output, test_loader, tolerance=(1e-3, 1e-5), precision_check=True)

def compare_model_outputs(torch_model, onnx_model, data_loader, tolerance=(1e-3, 1e-5), precision_check=False):
    logger = setup_logging()
    logger.info("Precision Alignment: comparing outputs between PyTorch and ONNX...")
    torch_model.eval()
    ort_session = ort.InferenceSession(onnx_model)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    num_batches = 0
    all_close = True

    for x, _ in data_loader:
        with torch.no_grad():
            torch_output = torch_model(x).cpu().numpy()

        ort_output = ort_session.run([output_name], {input_name: x.cpu().numpy()})[0]

        # Check top-1 predictions match
        torch_preds = torch_output.argmax(axis=2).squeeze(1)
        onnx_preds = ort_output.argmax(axis=2).squeeze(1)

        num_correct = (torch_preds == onnx_preds).sum().item()
        num_total = len(torch_preds)
        match_ratio = num_correct / num_total
        logger.error(f"Batch {num_batches}: Match {num_correct} out of {num_total} ({match_ratio * 100:.2f}%)")
        # assert match_ratio > 0.99, f"Predictions diverge too much: {match_ratio * 100:.2f}%"

        if precision_check:
            try:
                np.testing.assert_allclose(
                    ort_output, torch_output, rtol=tolerance[0], atol=tolerance[1]
                )
            except AssertionError as e:
                logger.error(f"Batch {num_batches} failed precision check: {e}")
                all_close = False
        
        num_batches += 1

    if precision_check:
        if all_close:
            logger.info(f"All batches passed precision checks within tolerance {tolerance}.")
        else:
            logger.error("Some batches failed precision checks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX format.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_loader, test_loader, _ = get_data_loaders(config["data"])
    export_model(config["model"]["base_cnn"], args.checkpoint, args.output, test_loader)