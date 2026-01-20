import torch
import random
from ..config import MEMORY_SIZE

class ReplayBuffer:
    """Stores (image, mask) pairs from previous tasks"""

    def __init__(self, max_size: int = MEMORY_SIZE):
        self.max_size = max_size
        self.buffer = []
        self.task_counts = {}

    def add_samples(self, images, masks, filenames, task_id):
        batch_size = images.shape[0]

        for i in range(batch_size):
            if len(self.buffer) < self.max_size:
                self.buffer.append({
                    'image': images[i].cpu(),
                    'mask': masks[i].cpu(),
                    'filename': filenames[i],
                    'task_id': task_id
                })
                self.task_counts[task_id] = self.task_counts.get(task_id, 0) + 1
            else:
                # Buffer full, use reservoir sampling
                idx = random.randint(0, len(self.buffer) - 1)
                old_task = self.buffer[idx]['task_id']
                self.task_counts[old_task] -= 1

                self.buffer[idx] = {
                    'image': images[i].cpu(),
                    'mask': masks[i].cpu(),
                    'filename': filenames[i],
                    'task_id': task_id
                }
                self.task_counts[task_id] = self.task_counts.get(task_id, 0) + 1

    def sample(self, batch_size: int, device='cuda'):
        if len(self.buffer) == 0:
            return None, None

        sample_size = min(batch_size, len(self.buffer))
        samples = random.sample(self.buffer, sample_size)

        images = torch.stack([s['image'] for s in samples]).to(device)
        masks = torch.stack([s['mask'] for s in samples]).to(device)

        return images, masks

    def __len__(self):
        return len(self.buffer)

    def get_stats(self):
        return {
            'total': len(self.buffer),
            'max_size': self.max_size,
            'task_distribution': self.task_counts
        }
