"""Fault-tolerant, memory-bounded telemetry ingestion (CSV / Parquet / PCAP / PCAPNG)."""
from .schema import CANON_COLUMNS, normalize_frame
from .csv_reader import iter_flow_chunks, read_flows
from .pcap_reader import read_pcap_flows

__all__ = ["CANON_COLUMNS", "normalize_frame", "iter_flow_chunks", "read_flows", "read_pcap_flows"]
