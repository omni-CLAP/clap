"""Weighted distributed sampler for a `ConcatDataset` mix of embodiments."""

import math

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def _get_world_rank():
    """(world_size, rank) from an initialized torch.distributed group, else single-process defaults."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


class WeightedConcatSampler(Sampler):
    """Samples from a `ConcatDataset`'s sub-datasets in fixed proportions, sharded across ranks.

    Args:
        sampling_weights: Relative weight per sub-dataset (e.g. `concat.sampling_weights`);
            defaults to uniform. Sub-datasets with 0 samples get 0 effective weight
            so a partial/missing dataset doesn't break training.
    """

    def __init__(self, dataset, sampling_weights=None, num_replicas=None, rank=None, shuffle=True, seed=0):
        world_size, world_rank = _get_world_rank()
        self.num_replicas = num_replicas if num_replicas is not None else world_size
        self.rank = rank if rank is not None else world_rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # ConcatDataset.cumulative_sizes is a running total across sub-datasets;
        # diff consecutive entries to recover each sub-dataset's own size.
        cumulative_sizes = getattr(dataset, "cumulative_sizes", None)
        assert cumulative_sizes is not None, "WeightedConcatSampler needs a ConcatDataset"
        self.dataset_sizes = []
        prev = 0
        for cs in cumulative_sizes:
            self.dataset_sizes.append(cs - prev)
            prev = cs

        if sampling_weights is None:
            sampling_weights = [1.0] * len(self.dataset_sizes)
        assert len(sampling_weights) == len(self.dataset_sizes)

        # Zero out weight for any empty sub-dataset, then renormalize to sum to 1.
        effective = [w if s > 0 else 0.0 for w, s in zip(sampling_weights, self.dataset_sizes)]
        total_w = sum(effective) or 1.0
        self.sampling_weights = [w / total_w for w in effective]

        # Global index offset of each sub-dataset within the concatenated dataset.
        self.offsets = [0]
        for s in self.dataset_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)

        # Round total_size up to a multiple of num_replicas so every rank gets
        # exactly num_samples indices (the padding step below covers the remainder).
        total_size = sum(self.dataset_sizes)
        self.num_samples = max(1, math.ceil(total_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Draw each sub-dataset's share of indices independently, then merge.
        indices = []
        for size, weight, offset in zip(self.dataset_sizes, self.sampling_weights, self.offsets):
            if size == 0 or weight == 0:
                continue
            n = max(1, int(weight * self.total_size))
            sub = torch.randint(0, size, (n,), generator=g) + offset
            indices.append(sub)
        indices = torch.cat(indices) if indices else torch.zeros(self.total_size, dtype=torch.long)

        if self.shuffle and len(indices) > 0:
            perm = torch.randperm(len(indices), generator=g)
            indices = indices[perm]

        if len(indices) < self.total_size:
            # Rounding (int(weight * total_size)) can undershoot; pad by resampling.
            pad = torch.randint(0, max(1, len(indices)), (self.total_size - len(indices),), generator=g)
            indices = torch.cat([indices, indices[pad]])
        indices = indices[:self.total_size]
        return iter(indices[self.rank::self.num_replicas].tolist())  # this rank's shard

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch  # reseeds shuffling per epoch, called by the training loop


def default_collate_fn(batch):
    """Collate tensors normally; keep strings/per-step caption lists as Python lists."""
    out = {}
    for key in batch[0].keys():
        vals = [s[key] for s in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)  # (B, ...) batch of per-sample tensors
        elif isinstance(vals[0], str):
            out[key] = vals  # leave as a plain list of strings
        elif isinstance(vals[0], (int, float)):
            out[key] = torch.tensor(vals)
        else:
            out[key] = vals  # e.g. per-step caption lists for language conditioning
    return out
