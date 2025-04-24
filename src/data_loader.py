import os
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, root: str = "./data/raw"):
        super().__init__(root = root, download=True)
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

def get_data_loaders(config):

    root = config["raw_dir"]
    temp_set = SubsetSC("testing", root=root)
    _, original_sample_rate, label, _, _ = temp_set[0]
    new_sample_rate = config["sample_rate"] 
    labels = sorted(list(set(d[2] for d in temp_set)))

    def label_to_index(word):
    # Return the position of the word in labels
        return torch.tensor(labels.index(word))

    # Create resample transform
    transform = torchaudio.transforms.Resample(
        orig_freq=original_sample_rate,
        new_freq=new_sample_rate
    )

    def collate_fn(batch):
        tensors, targets = [], []
        for waveform, _, label, *_ in batch:
            # Apply resampling to each waveform
            resampled = transform(waveform)
            tensors.append(resampled)
            targets.append(label_to_index(label))
            
        tensors = pad_sequence(tensors)
        targets = torch.stack(targets)
        return tensors, targets
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    train_loader = DataLoader(
        SubsetSC("training", root=root),
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        SubsetSC("testing", root=root),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    validate_loader = DataLoader(
        SubsetSC("validation", root=root),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader, validate_loader

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)
