import torch
from src.models.cnn_model import M5, QATM5
from src.utils import *
import yaml

device = get_device()
logger = setup_logging()

with open("./configs/cnn_qat.yaml") as f:
    config = yaml.safe_load(f)
qat_model_config = config["model"]["base_cnn"]
quant_model = QATM5(n_input=qat_model_config["n_input"],
                    n_output=qat_model_config["n_output"],
                    stride=qat_model_config["stride"],
                    n_channel=qat_model_config["n_channel"],
                    conv_kernel_sizes=qat_model_config["conv_kernel_sizes"])

PATH = './models/qat_model.pth'
state = {'model': quant_model.state_dict()}
# torch.save(state, PATH)
quant_model.load_state_dict(torch.load(PATH)['model'])
# print weights
for k, v in quant_model.named_parameters():
    print(k, v.shape, v.dtype)