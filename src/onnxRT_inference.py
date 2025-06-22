import onnxruntime as ort
import numpy as np
from src.utils import *
from src.data_loader import get_data_loaders
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import time
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Runtime Inference Script.")
    parser.add_argument("--model", type=str, required=True, help="Path to the ONNX model.")
    args = parser.parse_args()

    logger = setup_logging()

    ort_session = ort.InferenceSession(args.model)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    all_preds = []
    all_labels = []

    data_config = {
        "raw_dir": "./data/raw",
        "processed_dir": "./data/processed",
        "sample_rate": 8000,
        "batch_size": 256,
        "version": "v0.1",
    }
    _, test_loader, _ = get_data_loaders(data_config)

    # Metric 1: Model Size
    model_size = os.path.getsize(args.model)
    logger.info(f"1. Model Size: {model_size / 1024:.2f} KB")

    # Metric 2: Accuracy calculation
    for x, y in tqdm(test_loader):
        x_numpy = x.cpu().numpy()

        onnx_inputs = {input_name: x_numpy}
        outputs = ort_session.run([output_name], onnx_inputs)

        pred = np.argmax(outputs[0], axis=-1)  # adjust if shape is [B, 1, num_classes]
        all_preds.extend(pred.flatten())
        all_labels.extend(y.cpu().numpy().flatten())

    accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f"2. Accuracy: {accuracy * 100:.2f}%")

    # Metric 3: Inference speed
    batch = next(iter(test_loader))[0].cpu().numpy()

    # Warmup
    for _ in range(10):
        ort_session.run([output_name], {input_name: batch})

    # Timing
    N = 500
    start = time.time()
    for _ in range(N):
        ort_session.run([output_name], {input_name: batch})
    end = time.time()
    logger.info(f"3. Average inference time: {(end - start) / N * 1000:.2f} ms")
