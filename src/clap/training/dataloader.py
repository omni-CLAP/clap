"""Dataloader construction: a weighted sampler for multi-embodiment mixes, plain shuffling otherwise."""

from torch.utils.data import ConcatDataset, DataLoader

from clap.data.sampler import WeightedConcatSampler, default_collate_fn


def build_dataloader(dataset, training_config, data_config, mode):
    """Build a DataLoader for `dataset`.

    A `ConcatDataset` in train mode (one of the cross-embodiment mixes) gets
    `WeightedConcatSampler` so each sub-dataset is drawn in its configured
    proportion rather than uniformly by episode count; everything else
    (single-embodiment datasets, or any val split) shuffles/iterates normally.

    Args:
        training_config: `TrainingConfig` — batch_size/shuffle live here (they're
            training-loop concerns), not on `data_config`.
        data_config: `DataConfig` — only `num_workers` is read here.
    """
    if mode == "train" and isinstance(dataset, ConcatDataset):
        # sampling_weights is attached to the ConcatDataset by the build_oxe_*_dataset
        # functions (e.g. OXE_EE_SAMPLING_WEIGHTS), not a property of ConcatDataset itself.
        sampling_weights = getattr(dataset, "sampling_weights", None)
        sampler = WeightedConcatSampler(dataset, sampling_weights=sampling_weights, shuffle=True, seed=0)
        return DataLoader(
            dataset, batch_size=training_config.train_batch_size, sampler=sampler,
            num_workers=data_config.num_workers, collate_fn=default_collate_fn,
            drop_last=True,  # keeps batch size constant across ranks for accelerate/DDP
        )
    return DataLoader(
        dataset, batch_size=training_config.train_batch_size,
        shuffle=training_config.shuffle and mode == "train",
        num_workers=data_config.num_workers,
    )
