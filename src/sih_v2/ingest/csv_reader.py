"""Chunked, memory-bounded readers for flow telemetry files"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

from .schema import column_mapping, normalize_frame

log = logging.getLogger(__name__)

PCAP_SUFFIXES = {".pcap", ".pcapng", ".cap"}


def _csv_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0, encoding_errors="replace").columns)


def iter_flow_chunks(path: str | Path, chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    """Yield canonical flow frames chunk by chunk.

    * only mapped columns are read (CIC files have 80 columns; we need ~24)
    * malformed lines, repeated header rows, infinity/NaN values and bad timestamps
      are skipped rather than crashing the pipeline
    """
    path = Path(path)
    if path.suffix.lower() in PCAP_SUFFIXES:
        from .pcap_reader import read_pcap_flows

        yield read_pcap_flows(path)
        return
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        mapping = column_mapping(pf.schema_arrow.names)
        for batch in pf.iter_batches(batch_size=chunksize, columns=list(mapping)):
            yield normalize_frame(batch.to_pandas(), mapping)
        return

    header = _csv_header(path)
    mapping = column_mapping(header)
    if not any(c == "ts" for c, _ in mapping.values()):
        raise ValueError(f"{path.name}: no timestamp column recognised in header {header[:15]}")
    reader = pd.read_csv(
        path, usecols=list(mapping), chunksize=chunksize, dtype=str,
        on_bad_lines="skip", encoding_errors="replace", low_memory=True,
        compression="infer",
    )
    total = 0
    for i, chunk in enumerate(reader):
        try:
            df = normalize_frame(chunk, mapping)
        except Exception as exc:  # corrupt chunk: log and continue
            log.warning("%s chunk %d skipped: %s", path.name, i, exc)
            continue
        total += len(df)
        yield df
    log.info("%s: %d valid flows", path.name, total)


def read_flows(path: str | Path, chunksize: int = 200_000) -> pd.DataFrame:
    """Read a whole (small) file into one canonical frame."""
    parts = list(iter_flow_chunks(path, chunksize))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def list_inputs(paths) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for ext in ("*.csv", "*.csv.gz", "*.parquet", "*.binetflow", "*.pcap", "*.pcapng"):
                files.extend(sorted(p.rglob(ext)))
        elif p.exists():
            files.append(p)
    return files
