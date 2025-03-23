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

    - 

## Evaluation Metrics

- [x] Model Size
- [ ] Accuracy Drop
- [ ] Inference Time (Latency)
- [ ] Memory Footprint

## Some best practices for Quantization

Got some best practices which could be applied/tested.

- > Quantizing a model from a floating point checkpoint provides better accuracy. 
(https://arxiv.org/abs/1806.08342) 

    - Insert QAT after finishing FP model training? 
        #TODO: Compare: QAT from scratch VS QAT from fp checkpoint VS PTQ from same checkpoint

- 

## Experiment Progress

|  | CNN | Transformer | RNN?(NSNet 2) |
| ---- | ---- | ---- | ---- |
| Full Precision (fp32) | [x] | [ ] | [ ] |
| Quantization Aware Training (QAT) | [x] | [ ] | [ ] |
| Post Training Quantization (PTQ) | [ ] | [ ] | [ ] |
| Activation-aware Weight Quantization (AWQ) | [ ] | [ ] | [ ] |

