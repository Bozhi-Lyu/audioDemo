# Audio Model Quantization

## Experiments setup

- Dataset: SpeechCommands ([Pytorch Link](https://pytorch.org/audio/main/generated/torchaudio.datasets.SPEECHCOMMANDS.html)).

- Model Architecture

    - CNN: M5 from [this paper](https://arxiv.org/abs/1610.00087).
    - Transformers: 
    - RNN-like # TODO

- Quantization Configuration

    - Default quantization config: 

    ``` Python
    import torch

    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
    qconfig = model.qconfig

    # Instantiate the observers
    activation_observer = qconfig.activation()
    weight_observer = qconfig.weight()

    # Print the quantization schemes
    print("Activation Observer qscheme:", activation_observer.qscheme) 
    # (torch.per_tensor_affine,)

    print("Weight Observer qscheme:", weight_observer.qscheme) 
    # (torch.per_channel_symmetric,)

    print("\nActivation Observer quant_min/max:", activation_observer.quant_min, activation_observer.quant_max) 
    # (0, 127)

    print("Weight Observer quant_min/max:", weight_observer.quant_min, weight_observer.quant_max) 
    # (-128, 127)
            
    ```


## Evaluation Metrics

- [x] Model Size
- [x] Accuracy Drop
- [x] Inference Time (Latency)
- ~~[ ] Memory Footprint~~

## Experiment Steps

- CNN models:
    - Model Training:

    ``` bash
    python src/main.py --config ./configs/cnn_fp32.yaml

    python src/main.py --config ./configs/cnn_ptq.yaml

    python src/main.py --config ./configs/cnn_qat.yaml
    ```

    - Evaluation (including inference time by profiling tools):

    ``` bash
    kernprof -l -v -o logs/profiling_logs/fp32_profiling.lprof src/evaluate.py --checkpoint ./models/cnn_fp32_model.pth --config ./configs/cnn_fp32.yaml

    kernprof -l -v -o logs/profiling_logs/ptq_profiling.lprof src/evaluate.py --checkpoint ./models/cnn_ptq_model.pth --config ./configs/cnn_ptq.yaml

    kernprof -l -v -o logs/profiling_logs/qat_profiling.lprof src/evaluate.py --checkpoint ./models/cnn_qat_model.pth --config ./configs/cnn_qat.yaml
    ```



- Transformers (Wav2Vec2)


## Experiment Progress

|  | CNN | Transformer | RNN?(NSNet 2) |
| ---- | ---- | ---- | ---- |
| Full Precision (fp32) | [x] | [ ] | [ ] |
| Quantization Aware Training (QAT) | [x] | [ ] | [ ] |
| Post Training Quantization (PTQ) | [x] | [ ] | [ ] |
| Activation-aware Weight Quantization (AWQ) | [ ] | [ ] | [ ] |


## Experiment Results

See [report](/notebook/visualization.ipynb).

## Some practices/tips for Quantization

- > Quantizing a model from a floating point checkpoint provides better accuracy. 
(https://arxiv.org/abs/1806.08342) 

    - Insert QAT after finishing FP model training? 

- 