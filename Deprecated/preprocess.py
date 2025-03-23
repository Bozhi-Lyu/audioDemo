import os
import logging
import sys
import argparse
import yaml
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, root: str = "./data/raw"):
        super().__init__(root, download=True)

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

class PreprocessedDataset(Dataset):
    """Dataset for loading preprocessed data"""
    def __init__(self, file_path):
        self.data = torch.load(file_path)
        
    def __len__(self):
        return len(self.data["labels"])
    
    def __getitem__(self, idx):
        return self.data["tensors"][idx].squeeze(0), self.data["labels"][idx]

def preprocess_and_save(dataset, subset_name, config):
    """Process resample and save dataset"""
    save_path = os.path.join(
        config["data"]["processed_dir"],
        f"{subset_name}_{config['preprocessing']['version']}.pt"
    )
    
    if os.path.exists(save_path):
        return save_path

    # waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
    transform = torchaudio.transforms.Resample(
        orig_freq=dataset[0][1],
        new_freq=config["preprocessing"]["sample_rate"]
    )

    labels_set = sorted(list(set(datapoint[2] for datapoint in dataset)))
    batch_size = 1000
    current_batch = 0
    accumulated = {"tensors": [], "labels": []}
    len_dataset = len(dataset)

    for i in range(len_dataset):
        # Process sample
        waveform, _, label, *_ = dataset[i]
        processed = transform(waveform)
        
        accumulated["tensors"].append(processed.numpy().astype('float32'))
        accumulated["labels"].append(labels_set.index(label))
        
        # Clear memory
        del waveform, processed
        if (i + 1) % batch_size == 0:
            # Save incremental progress
            _save_incremental(accumulated, save_path, current_batch)
            logger.info(f"Processed {i+1}/{len_dataset} samples")
            current_batch += 1
            accumulated = {"tensors": [], "labels": []}
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save remaining samples
    if accumulated["tensors"]:
        _save_incremental(accumulated, save_path, current_batch)
    
    logger.info(f"Processing completed.")
    
    # Combine batches and save final file
    return _combine_batches(save_path, current_batch)

def _save_incremental(data, base_path, batch_num):
    batch_path = f"{base_path}.batch{batch_num}"
    torch.save({
        "tensors": data["tensors"],
        "labels": data["labels"]
    }, batch_path)

def _combine_batches(base_path, num_batches):
    final_data = {"tensors": [], "labels": []}
    
    for i in range(num_batches + 1):
        batch_path = f"{base_path}.batch{i}"
        batch = torch.load(batch_path, weights_only=False)
        final_data["tensors"].extend(batch["tensors"])
        final_data["labels"].extend(batch["labels"])
        os.remove(batch_path)
        
    torch.save(final_data, base_path)
    return base_path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument('--config', default="./configs/preprocess.yaml", type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_set = SubsetSC("training", root=config["data"]["raw_dir"])
    test_set = SubsetSC("testing", root=config["data"]["raw_dir"])

    os.makedirs(config["data"]["processed_dir"], exist_ok=True)

    logger.info("Processing dataset...")
    train_path = preprocess_and_save(train_set, "train", config)
    test_path = preprocess_and_save(test_set, "test", config)

    # Load preprocessed datasets
    train_preprocessed = PreprocessedDataset(train_path)
    test_preprocessed = PreprocessedDataset(test_path)

    logger.info("\nPreprocessing verification:")
    logger.info(f"Train samples: {len(train_preprocessed)}")
    logger.info(f"Test samples: {len(test_preprocessed)}")
    sample_tensor, sample_label = train_preprocessed[0]
    logger.info(f"Sample tensor shape: {sample_tensor.shape}")
    logger.info(f"Sample label: {sample_label}")

    # # DataLoader setup
    # def collate_fn(batch):
    #     tensors, targets = zip(*batch)
    #     tensors = [t.squeeze(0).t() for t in tensors]  # Convert to (seq_len, 1)
    #     padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    #     return padded.permute(0, 2, 1), torch.tensor(targets)

    # batch_size = 256
    # num_workers = 4 if device == "cuda" else 0

    # train_loader = torch.utils.data.DataLoader(
    #     PreprocessedDataset(train_path),
    #     batch_size=params["batch_size"],
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     num_workers=num_workers,
    #     pin_memory=True
    # )

    # test_loader = torch.utils.data.DataLoader(
    #     PreprocessedDataset(test_path),
    #     batch_size=params["batch_size"],
    #     shuffle=False,
    #     collate_fn=collate_fn,
    #     num_workers=num_workers,
    #     pin_memory=True
    # )

