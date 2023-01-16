from numpy import random
from pyarrow import parquet as pq

import numpy as np
import pyarrow as pa

import sys

DATES = []

# generate some iso dates that may lack a month field
for y in range(1900, 2020):
    DATES.append(f"{y}")
    for m in range(1, 13):
        DATES.append(f"{y}-{m:02d}")
        for d in range(1, 6):
            DATES.append(f"{y}-{m:02d}-{d:02d}")


def rbatch(SIZE):
    SIZE = int(SIZE)

    frames = []
    for c in ["Banana", "Strawberry", "Apple", "Mulberry"]:
        mid_x = np.random.normal()
        scale = random.random()
        x = random.normal(loc=mid_x, scale=(scale + 0.5) / 3, size=SIZE // 4)
        mid_y = np.random.normal()
        scale = random.random()
        y = random.normal(loc=mid_y, scale=(scale + 0.5) / 3, size=SIZE // 4)
        date = random.choice(DATES, size=len(x), replace=True)
        frame = pa.table(
            {
                "x": x,
                "y": y,
                "class": [c] * len(x),
                "quantity": random.random(SIZE // 4),
                "date": date,
            }
        )
        frames.append(frame)
    return pa.concat_tables(frames)


def demo_parquet(path, size, batchsize=2e5) -> None:
    writer = None
    written = 0
    while written < size:
        if size - written < batchsize:
            batchsize = size - written
        batch = rbatch(batchsize)
        if writer is None:
            writer = pq.ParquetWriter(path, batch.schema)
        writer.write_table(batch)
        written = written + batchsize


def main(path="tmp.csv", SIZE=None) -> None:
    if SIZE is None:
        try:
            SIZE = int(sys.argv[1])
        except ValueError:
            SIZE = 100_000

    frames = rbatch(SIZE).to_pandas()
    frames = frames.sample(frac=1)

    # Add an unseen level at the very end.
    frames.iloc[-1, -1] = "2040-01-01"

    frames.to_csv(path, index=False)
