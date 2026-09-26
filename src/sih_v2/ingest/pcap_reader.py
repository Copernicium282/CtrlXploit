"""Dependency-free PCAP / PCAPNG parser + bidirectional flow assembler.

Extracts the packet-level attributes that CICFlowMeter throws away and that
matter for early warning: per-flow TTL mean/std (OS / hop-count fingerprint
drift), initial TCP window size, TCP flag counts, retransmissions and
inter-arrival statistics. Streaming: flows are flushed on idle timeout so memory
is bounded by the number of *concurrently active* flows, not the capture size.
"""
from __future__ import annotations

import math
import socket
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterator

import pandas as pd

from .schema import normalize_frame

LINK_ETHERNET, LINK_RAW, LINK_RAW_ALT, LINK_SLL, LINK_IPV4 = 1, 101, 12, 113, 228
SMALL_PAYLOAD = 64      # bytes; scan probes and bare handshakes sit at or below this


@dataclass
class Packet:
    ts: float
    src: str
    dst: str
    sport: int
    dport: int
    proto: int
    ttl: int
    length: int
    flags: int = 0       # TCP flag byte
    window: int = 0
    seq: int = 0
    payload: int = 0
    ip_flags: int = 0    # IPv4 flags field: 0x1 = MF (more fragments), 0x2 = DF (don't fragment)
    frag_off: int = 0    # IPv4 fragment offset in 8-byte units (0 = first fragment)


def _iter_pcap(fh: BinaryIO) -> Iterator[tuple[float, int, bytes]]:
    magic = fh.read(4)
    if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        endian = "<"
    elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
        endian = ">"
    elif magic == b"\x0a\x0d\x0d\x0a":
        yield from _iter_pcapng(fh, magic)
        return
    else:
        raise ValueError("Not a pcap/pcapng file")
    nano = magic in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
    hdr = fh.read(20)
    linktype = struct.unpack(endian + "HHiIII", hdr)[5] & 0xFFFF
    div = 1e9 if nano else 1e6
    while True:
        rec = fh.read(16)
        if len(rec) < 16:
            return
        sec, frac, incl, _orig = struct.unpack(endian + "IIII", rec)
        data = fh.read(incl)
        if len(data) < incl:
            return  # truncated capture - stop gracefully
        yield sec + frac / div, linktype, data


def _iter_pcapng(fh: BinaryIO, first: bytes) -> Iterator[tuple[float, int, bytes]]:
    endian = "<"
    links: list[int] = []
    tsres: list[float] = []
    block_type = first
    while True:
        if block_type is None:
            block_type = fh.read(4)
        if len(block_type) < 4:
            return
        raw_len = fh.read(4)
        if len(raw_len) < 4:
            return
        if block_type == b"\x0a\x0d\x0d\x0a":
            bom = fh.read(4)
            endian = "<" if bom == b"\x4d\x3c\x2b\x1a" else ">"
            blen = struct.unpack(endian + "I", raw_len)[0]
            fh.read(blen - 12)
            links, tsres = [], []
            block_type = None
            continue
        btype = struct.unpack(endian + "I", block_type)[0]
        blen = struct.unpack(endian + "I", raw_len)[0]
        body = fh.read(blen - 8)
        block_type = None
        if btype == 1:  # Interface Description Block
            links.append(struct.unpack(endian + "H", body[:2])[0])
            res = 1e-6
            opts = body[8:-4]
            i = 0
            while i + 4 <= len(opts):
                code, olen = struct.unpack(endian + "HH", opts[i:i + 4])
                if code == 0:
                    break
                if code == 9 and olen >= 1:
                    v = opts[i + 4]
                    res = 2.0 ** -(v & 0x7F) if v & 0x80 else 10.0 ** -v
                i += 4 + olen + ((4 - olen % 4) % 4)
            tsres.append(res)
        elif btype == 6:  # Enhanced Packet Block
            iface, hi, lo, cap, _ = struct.unpack(endian + "IIIII", body[:20])
            ts = ((hi << 32) | lo) * (tsres[iface] if iface < len(tsres) else 1e-6)
            yield ts, links[iface] if iface < len(links) else LINK_ETHERNET, body[20:20 + cap]
        elif btype == 3:  # Simple Packet Block (no timestamp)
            yield 0.0, links[0] if links else LINK_ETHERNET, body[4:]


def _decode(ts: float, linktype: int, data: bytes) -> Packet | None:
    off = 0
    if linktype == LINK_ETHERNET:
        if len(data) < 14:
            return None
        etype = struct.unpack("!H", data[12:14])[0]
        off = 14
        while etype in (0x8100, 0x88A8) and len(data) >= off + 4:  # VLAN tags
            etype = struct.unpack("!H", data[off + 2:off + 4])[0]
            off += 4
        if etype != 0x0800:
            return None
    elif linktype == LINK_SLL:
        if len(data) < 16 or struct.unpack("!H", data[14:16])[0] != 0x0800:
            return None
        off = 16
    elif linktype not in (LINK_RAW, LINK_RAW_ALT, LINK_IPV4):
        return None
    ip = data[off:]
    if len(ip) < 20 or ip[0] >> 4 != 4:
        return None
    ihl = (ip[0] & 0x0F) * 4
    total_len = struct.unpack("!H", ip[2:4])[0]
    frag_field = struct.unpack("!H", ip[6:8])[0]
    ttl, proto = ip[8], ip[9]
    src, dst = socket.inet_ntoa(ip[12:16]), socket.inet_ntoa(ip[16:20])
    l4 = ip[ihl:]
    pkt = Packet(ts, src, dst, 0, 0, proto, ttl, total_len,
                 ip_flags=frag_field >> 13, frag_off=frag_field & 0x1FFF)
    if proto == 6 and len(l4) >= 20:
        pkt.sport, pkt.dport, pkt.seq = struct.unpack("!HHI", l4[:8])
        doff = (l4[12] >> 4) * 4
        pkt.flags = l4[13]
        pkt.window = struct.unpack("!H", l4[14:16])[0]
        pkt.payload = max(total_len - ihl - doff, 0)
    elif proto == 17 and len(l4) >= 8:
        pkt.sport, pkt.dport = struct.unpack("!HH", l4[:4])
        pkt.payload = max(total_len - ihl - 8, 0)
    return pkt


@dataclass
class _Flow:
    src: str
    dst: str
    sport: int
    dport: int
    proto: int
    first: float
    last: float
    fwd_pkts: int = 0
    bwd_pkts: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    flags: dict = field(default_factory=lambda: dict(syn=0, ack=0, fin=0, rst=0, psh=0, urg=0))
    ttl_sum: float = 0.0
    ttl_sq: float = 0.0
    init_win: int = -1
    iat_sum: float = 0.0
    iat_sq: float = 0.0
    iat_max: float = 0.0
    n_iat: int = 0
    retrans: int = 0
    frag_pkts: int = 0        # packets with MF set or a non-zero fragment offset
    df_pkts: int = 0          # packets with DF set
    payload_sum: float = 0.0
    payload_sq: float = 0.0
    payload_small: int = 0    # packets whose payload is a scan-sized <= SMALL_PAYLOAD bytes
    seen_seq: set = field(default_factory=set)

    def add(self, p: Packet, forward: bool) -> None:
        if self.fwd_pkts + self.bwd_pkts:
            d = max(p.ts - self.last, 0.0)
            self.iat_sum += d
            self.iat_sq += d * d
            self.iat_max = max(self.iat_max, d)
            self.n_iat += 1
        self.last = max(self.last, p.ts)
        if forward:
            self.fwd_pkts += 1
            self.fwd_bytes += p.length
            if self.init_win < 0 and p.proto == 6:
                self.init_win = p.window
        else:
            self.bwd_pkts += 1
            self.bwd_bytes += p.length
        self.ttl_sum += p.ttl
        self.ttl_sq += p.ttl * p.ttl
        if p.ip_flags & 0x1 or p.frag_off:      # MF set, or a continuation fragment
            self.frag_pkts += 1
        if p.ip_flags & 0x2:
            self.df_pkts += 1
        self.payload_sum += p.payload
        self.payload_sq += p.payload * p.payload
        if p.payload <= SMALL_PAYLOAD:
            self.payload_small += 1
        if p.proto == 6:
            for bit, name in ((0x02, "syn"), (0x10, "ack"), (0x01, "fin"), (0x04, "rst"), (0x08, "psh"), (0x20, "urg")):
                if p.flags & bit:
                    self.flags[name] += 1
            if p.payload > 0:
                key = (forward, p.seq)
                if key in self.seen_seq:
                    self.retrans += 1
                else:
                    self.seen_seq.add(key)

    def row(self) -> dict:
        n = self.fwd_pkts + self.bwd_pkts
        ttl_m = self.ttl_sum / n
        pl_m = self.payload_sum / n
        iat_m = self.iat_sum / self.n_iat if self.n_iat else 0.0
        return {
            "Timestamp": self.first, "Src IP": self.src, "Dst IP": self.dst,
            "Src Port": self.sport, "Dst Port": self.dport, "Protocol": self.proto,
            "Duration": self.last - self.first,
            "Tot Fwd Pkts": self.fwd_pkts, "Tot Bwd Pkts": self.bwd_pkts,
            "TotLen Fwd Pkts": self.fwd_bytes, "TotLen Bwd Pkts": self.bwd_bytes,
            **{f"{k} Flag Cnt": v for k, v in self.flags.items()},
            "TTL Mean": ttl_m, "TTL Std": math.sqrt(max(self.ttl_sq / n - ttl_m ** 2, 0.0)),
            "Init Fwd Win Byts": max(self.init_win, 0),
            "iat_mean": iat_m,
            "iat_std": math.sqrt(max(self.iat_sq / self.n_iat - iat_m ** 2, 0.0)) if self.n_iat else 0.0,
            "iat_max": self.iat_max,
            "Frag Pkts": self.frag_pkts, "DF Pkts": self.df_pkts,
            "Payload Mean": pl_m, "Payload Std": math.sqrt(max(self.payload_sq / n - pl_m ** 2, 0.0)),
            "Payload Small Pkts": self.payload_small,
            "Retrans": self.retrans, "Label": "Benign",
        }


def read_pcap_flows(path: str | Path, idle_timeout: float = 120.0, active_timeout: float = 1800.0) -> pd.DataFrame:
    """Parse a capture into canonical flows (labels unknown -> 'Benign')"""
    flows: dict[tuple, _Flow] = {}
    rows: list[dict] = []
    last_sweep = None
    with open(path, "rb") as fh:
        for ts, link, data in _iter_pcap(fh):
            try:
                p = _decode(ts, link, data)
            except (struct.error, OSError, IndexError):
                continue  # malformed packet
            if p is None:
                continue
            fkey = (p.src, p.dst, p.sport, p.dport, p.proto)
            rkey = (p.dst, p.src, p.dport, p.sport, p.proto)
            if fkey in flows:
                key, fwd = fkey, True
            elif rkey in flows:
                key, fwd = rkey, False
            else:
                key, fwd = fkey, True
            f = flows.get(key)
            if f is not None and (p.ts - f.last > idle_timeout or p.ts - f.first > active_timeout):
                rows.append(flows.pop(key).row())
                f = None
            if f is None:
                f = flows[key] = _Flow(p.src, p.dst, p.sport, p.dport, p.proto, p.ts, p.ts)
                fwd = True
            f.add(p, fwd)
            if last_sweep is None or p.ts - last_sweep > idle_timeout:
                last_sweep = p.ts
                for k in [k for k, v in flows.items() if p.ts - v.last > idle_timeout]:
                    rows.append(flows.pop(k).row())
    rows.extend(f.row() for f in flows.values())
    if not rows:
        return normalize_frame(pd.DataFrame({"Timestamp": pd.Series(dtype=float)}))
    return normalize_frame(pd.DataFrame(rows))


# --------------------------------------------------------------------------- writer (tests / samples)
def write_pcap(path: str | Path, packets: list[Packet]) -> None:
    """Write Ethernet/IPv4 packets to a little-endian pcap file"""
    with open(path, "wb") as fh:
        fh.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, LINK_ETHERNET))
        for p in packets:
            if p.proto == 6:
                l4 = struct.pack("!HHIIBBHHH", p.sport, p.dport, p.seq, 0, 5 << 4, p.flags, p.window, 0, 0)
            else:
                l4 = struct.pack("!HHHH", p.sport, p.dport, 8 + p.payload, 0)
            l4 += b"\x00" * p.payload
            total = 20 + len(l4)
            frag = ((p.ip_flags & 0x7) << 13) | (p.frag_off & 0x1FFF)
            ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, total, 0, frag, p.ttl, p.proto, 0,
                             socket.inet_aton(p.src), socket.inet_aton(p.dst))
            frame = b"\x00" * 12 + b"\x08\x00" + ip + l4
            sec = int(p.ts)
            fh.write(struct.pack("<IIII", sec, int(round((p.ts - sec) * 1e6)), len(frame), len(frame)))
            fh.write(frame)
