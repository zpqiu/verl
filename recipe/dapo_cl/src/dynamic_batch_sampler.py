# Some code are modified from https://github.com/pytorch/pytorch/blob/main/torch/utils/data/sampler.py

from typing import Any, Dict, Iterator, Union, Callable

import torch.utils.data.sampler
from torch.utils.data.dataloader import _InfiniteConstantSampler

from torchdata.stateful_dataloader import Stateful


class _BatchSamplerIterator(Iterator[list[int]], Stateful):
    _SAMPLES_YIELDED = "samples_yielded"
    _SAMPLER_STATE = "sampler_state"
    _SAMPLER_ITER_STATE = "sampler_iter_state"

    def __init__(self, sampler, batch_size: Union[int, Callable[[], int]], drop_last: bool):
        self.sampler = sampler
        self.sampler_iter = iter(self.sampler)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.samples_yielded = 0

    def __next__(self) -> list[int]:
        batch = []
        # 动态获取当前 batch 的大小
        current_batch_size = self.batch_size() if callable(self.batch_size) else self.batch_size
        # print(f"current_batch_size: {current_batch_size}")
        try:
            for _ in range(current_batch_size):
                batch.append(next(self.sampler_iter))
                self.samples_yielded += 1
            return batch
        except StopIteration:
            if self.drop_last or len(batch) == 0:
                raise StopIteration
            else:
                return batch

    def state_dict(self) -> Dict[str, Any]:
        sd: Dict[str, Any] = {self._SAMPLES_YIELDED: self.samples_yielded}
        if isinstance(self.sampler, Stateful):
            sd[self._SAMPLER_STATE] = self.sampler.state_dict()
        if isinstance(self.sampler_iter, Stateful):
            sd[self._SAMPLER_ITER_STATE] = self.sampler_iter.state_dict()
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.samples_yielded = state_dict[self._SAMPLES_YIELDED]
        if self._SAMPLER_STATE in state_dict:
            assert isinstance(self.sampler, Stateful)
            self.sampler.load_state_dict(state_dict[self._SAMPLER_STATE])
        self.sampler_iter = iter(self.sampler)
        if self._SAMPLER_ITER_STATE in state_dict:
            assert isinstance(self.sampler_iter, Stateful)
            self.sampler_iter.load_state_dict(state_dict[self._SAMPLER_ITER_STATE])

        if not (isinstance(self.sampler, Stateful) or isinstance(self.sampler_iter, Stateful)) and not isinstance(
            self.sampler, _InfiniteConstantSampler
        ):
            # We skip x samples if underlying sampler is not stateful
            for _ in range(self.samples_yielded):
                next(self.sampler_iter)

    def update_state_dict(self) -> None:
        if isinstance(self.sampler_iter, Stateful) and hasattr(self.sampler_iter, "update_state_dict"):
            self.sampler_iter.update_state_dict()


class BatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, sampler, batch_size: Union[int, Callable[[], int]], drop_last):
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        return _BatchSamplerIterator(
            sampler=self.sampler,
            batch_size=self.batch_size,  # 可传入固定值或者 callable
            drop_last=self.drop_last,
        )


if __name__ == "__main__":

    GLOBAL_BATCH_SIZE = 2

    def dynamic_batch_size():
        global GLOBAL_BATCH_SIZE
        return GLOBAL_BATCH_SIZE


    sample_dataset = range(10000)

    from torch.utils.data import RandomSampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    train_dataloader_generator = torch.Generator()
    train_dataloader_generator.manual_seed(1)
    sampler = RandomSampler(data_source=sample_dataset, generator=train_dataloader_generator)
    batch_sampler = BatchSampler(sampler, dynamic_batch_size, drop_last=False)

    dataloader = StatefulDataLoader(
        dataset=sample_dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
    )

    for i, batch in enumerate(dataloader):
        print(f"batch {i}: {batch}")
        
        if i >= 4:
            GLOBAL_BATCH_SIZE = i + 1

        if i > 6:
            state = dataloader.state_dict()
            break
    print(state)
