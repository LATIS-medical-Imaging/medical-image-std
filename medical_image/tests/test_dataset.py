import pytest


class TestDataset:
    def test_one(self):
        from torch.utils.data import DataLoader

        # Assuming you already defined `DicomDataset` as shown above
        dataset = DicomDataset(
            base_path="data/images",
            file_format="dcm",
            label_type="mask",
            label_data="data/masks",
            transform=transform
        )

        # Create DataLoader for batch-wise loading
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4,  # Parallel loading (depends on your CPU)
            pin_memory=True  # Optimized transfer to GPU
        )

        # Training loop (example)
        for batch in dataloader:
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            # ... forward pass, loss, etc.

