from src.utils import *
from src.data_loader import get_data_loaders
from src.models.cnn_model_LayerWiseQuant import M5Modular
import argparse

from onnxruntime.quantization import QuantFormat, quantize_static, CalibrationDataReader, QuantType

class DataReader(CalibrationDataReader):
    def __init__(self, dataloader):
        self.dataloader = iter(dataloader)
        self.iterator = None

    def get_next(self):
        if self.iterator is None:
            self.iterator = self._yield_batches()
        return next(self.iterator, None)

    def _yield_batches(self):
        for batch in self.dataloader:
            # Assume input is a tuple (input_tensor, label)
            if isinstance(batch, (list, tuple)):
                input_tensor = batch[0]
            else:
                input_tensor = batch
            yield {"input": input_tensor.numpy()}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Static quantization with ONNX Runtime.")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--per_channel", required=False, type=bool, default=True)
    args = parser.parse_args()

    logger = setup_logging()
    data_config = {
        "raw_dir": "./data/raw",
        "processed_dir": "./data/processed",
        "sample_rate": 8000,
        "batch_size": 256,
        "version": "v0.1"
    }
    _, _, validate_loader = get_data_loaders(data_config)

    quantize_static(
        model_input=args.input,
        model_output=args.output,
        calibration_data_reader=DataReader(validate_loader),
        quant_format=QuantFormat.QDQ,
        per_channel=args.per_channel,
    )
    logger.info("ONNX Runtime Static quantization successful.")
