import yaml
import torch
import copy
from src.utils import *
from src.data_loader import get_data_loaders
from src.models.cnn_model import M5, QATM5, PTQM5
from src.models.cnn_model_LayerWiseQuant import M5Modular, PTQM5Modular, PTQM5_LayerWiseQuant, QATM5Modular, QATM5_LayerWiseQuant
from src.train import set_seed, train_model
from src.evaluate import evaluate_model, test
import argparse
import json
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    PerChannelMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver
)
from torch.ao.quantization import QConfig, prepare, convert

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Qconfig tests.')
    parser.add_argument('--config', type=str, required=False, default="configs/cnn_ptq_LayerWiseQuant.yaml")
    args = parser.parse_args()


    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Get data
    train_loader, test_loader, validate_loader = get_data_loaders(config["data"])

    # Initialize model
    model_config = config["model"]["base_cnn"]

    # Load FP32 model
    base_model_fp32 = PTQM5Modular(
        n_input=model_config["n_input"],
        n_output=model_config["n_output"],
        stride=model_config["stride"],
        n_channel=model_config["n_channel"],
        conv_kernel_sizes=model_config["conv_kernel_sizes"]
    ).to('cpu')
    base_model_fp32.eval()
    base_model_fp32.load_state_dict(torch.load(model_config["pretrained_path"]))
    base_model_fp32.fuse_model()

    base_model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    torch.ao.quantization.prepare(base_model_fp32, inplace=True)

    # Calibrate model - use validation set
    with torch.inference_mode():
        for data, _ in validate_loader:
            data = data.to("cpu")
            base_model_fp32(data)

    # Convert to PTQ model
    model = torch.ao.quantization.convert(base_model_fp32, inplace=False)
    acc_train = test(model, train_loader)
    acc_test = test(model, test_loader)
    print(f"Default qconfig: Train Accuracy: {acc_train:.4f}, Test Accuracy: {acc_test:.4f}")

    def get_fresh_model():
        model = PTQM5Modular(
            n_input=model_config["n_input"],
            n_output=model_config["n_output"],
            stride=model_config["stride"],
            n_channel=model_config["n_channel"],
            conv_kernel_sizes=model_config["conv_kernel_sizes"]
        )
        model.eval()
        model.load_state_dict(torch.load(model_config["pretrained_path"]))
        model.fuse_model()
        return model

    # Define observer configurations
    activation_observers = {
        "MinMaxObserver": MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        # "MovingAverageMinMaxObserver": MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine), # not for PTQ, only for QAT.
        "HistogramObserver_affine": HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        "HistogramObserver_symmetric": HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric),
    }

    weight_observers = {
        # Standard choice
        "PerChannelMinMaxObserver_symmetric": PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        ), # MINMAX CS
        "PerChannelMinMaxObserver_affine": PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_affine
        ), # MINMAX CA

        # Fallback / simple methods
        "MinMaxObserver_per_tensor_affine": MinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_affine
        ), # MINMAX TA
        "MinMaxObserver_per_tensor_symmetric": MinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        ), # MINMAX TS

        # Alternative calibration methods
        "HistogramObserver_per_tensor_affine": HistogramObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_affine
        ), #HISTOGRAM TA
        "HistogramObserver_per_tensor_symmetric": HistogramObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        ),  #HISTOGRAM TS
        "HistogramObserver_per_channel_affine": HistogramObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_affine
        ), #HISTOGRAM CA
        "HistogramObserver_per_channel_symmetric": HistogramObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        ), #HISTOGRAM CS
    }

    # Test all qconfigs
    results = {}
    for act_name, act_observer in activation_observers.items():
        for wt_name, wt_observer in weight_observers.items():
            config_name = f"activation observer:{act_name}, weight observer:{wt_name}."
            print(f"\nTesting QConfig: {config_name}")

            try:
                qconfig = QConfig(activation=act_observer, weight=wt_observer)

                model_fp32 = get_fresh_model()
                model_fp32.qconfig = qconfig

                prepare(model_fp32, inplace=True)

                with torch.inference_mode():
                    for data, _ in validate_loader:
                        data = data.to("cpu")
                        model_fp32(data)

                model_int8 = convert(model_fp32, inplace=False)

                acc_train = test(model_int8, train_loader)
                acc_test = test(model_int8, test_loader)

                results[config_name] = {"train": acc_train, "test": acc_test}
                print(f"Train Accuracy: {acc_train:.4f}, Test Accuracy: {acc_test:.4f}")
            except Exception as e:
                print(f"Error with QConfig {config_name}: {e}")
                results[config_name] = {"error": str(e)}


    with open("ptq_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)