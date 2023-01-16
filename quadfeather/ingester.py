from pyarrow import parquet as pq, feather
from pathlib import Path
from typing import List, Optional

import pyarrow as pa


class Ingester:
    def __init__(
        self,
        files: list[Path],
        batch_size: int = 1024 * 1024 * 1024,
        columns=None,
        destructive=False,
    ):
        # Allow iteration over a st
        # queue_size: maximum bytes in an insert chunk.
        self.files = files
        self.queue = []
        self.columns = columns
        self.batch_size = batch_size
        self.destructive = destructive

    def __iter__(self):
        queue_length = 0
        for batch in self.batches():
            self.queue.append(batch)
            queue_length = queue_length + batch.nbytes
            if queue_length > self.batch_size:
                tb = pa.Table.from_batches(batches=self.queue)
                yield tb.combine_chunks()
                self.queue = []
                queue_length = 0
        if len(self.queue) > 0:
            tb = pa.Table.from_batches(batches=self.queue)
            yield tb.combine_chunks()


class ArrowIngester(Ingester):
    """
    Ingesting IPC serialization format, not the feather format. It is slightly
    different from the feather format, because it lacks a schema at the
    bottom.)
    """

    def batches(self):
        for file in self.files:
            source = pa.OSFile(str(file), "rb")
            with pa.ipc.open_file(
                source,
                options=pa.ipc.IpcReadOptions(included_fields=self.columns),
            ) as fin:
                for i in range(fin.num_record_batches):
                    yield fin.get_batch(i)
            if self.destructive:
                file.unlink()


"""
class CSVIngester(Ingester):
"""


class FeatherIngester(Ingester):
    """
    Ingesting the feather format.
    """

    def batches(self):
        for file in self.files:
            fin = feather.read_table(file, columns=self.columns)
            for batch in fin.to_batches():
                yield batch
            if self.destructive:
                fin.unlink()


class ParquetIngester(Ingester):
    """
    Ingesting the parquet format.
    """

    def batches(self):
        for f in self.files:
            f = pq.ParquetFile(f)
            for batch in f.iter_batches(columns=self.columns):
                yield batch
            if self.destructive:
                f.unlink()


def get_ingester(
    files: List[Path], destructive=False, columns: Optional[List[str]] = None
) -> Ingester:
    """
    Returns the correct ingester (i.e. ArrowIngester, FeatherIngester or
    ParquetIngester), depending on the suffix of the files provided.
    """

    suffixes = list(set([f.suffix for f in files]))

    assert len(suffixes) == 1, "All files must be of the same type"

    suffix = suffixes[0]
    if suffix == ".parquet":
        return ParquetIngester(files, destructive=destructive, columns=columns)
    elif suffix == ".feather":
        return FeatherIngester(files, destructive=destructive, columns=columns)
    elif suffix == ".arrow":
        return ArrowIngester(files, destructive=destructive, columns=columns)
    else:
        raise Exception(f"Unsupported file type: {suffix}")
