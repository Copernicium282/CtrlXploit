"""Representative telemetry generator (CSE-CIC-IDS2018 CSV layout + packet extensions).

No public dataset ships labelled, time-ordered multi-stage campaigns with
enough repetitions to train/evaluate forecasting, so we synthesise episodes
whose per-flow statistics follow the published CIC-IDS2018 / CTU-13 profiles:

  benign background (diurnal load, DNS, SMB, backups, IT scanners)
  -> Reconnaissance (low-and-slow ramp, OS-probe TTL spread, narrowing port set)
  -> Initial Access (SSH/FTP/RDP/Web brute force)
  -> Command & Control (low-jitter beaconing)  -> Lateral Movement (SMB/WinRM/RDP)
  -> Exfiltration (large outbound transfers)  -> optional Impact (DDoS)

Hard negatives are deliberately included (benign IT vulnerability scanners,
backups that look like exfiltration, DNS bursts, stealthy campaigns without
recon) so that the forecasting problem is not trivially separable.
The output is parsed by exactly the same ingestion path as real CIC CSVs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

W = 60.0  # seconds per generator tick (matches default window)

CIC_COLUMNS = [
    "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp", "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt",
    "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", "Init Fwd Win Byts",
    "TTL Mean", "TTL Std", "Retrans", "Label",
]

_CLIENTS = [f"10.0.0.{i}" for i in range(2, 121)]
_SERVERS = [f"10.0.1.{i}" for i in range(1, 21)]
_EXTERNAL = [f"{a}.{b}.{c}.{d}" for a, b, c, d in
             np.random.default_rng(7).integers([23, 0, 0, 1], [220, 255, 255, 254], size=(400, 4))]
_BENIGN_PORTS = np.array([443, 80, 53, 22, 445, 3389, 8080, 123, 25, 993])
_BENIGN_P = np.array([0.50, 0.12, 0.20, 0.02, 0.04, 0.01, 0.03, 0.03, 0.02, 0.03])
_TOP_PORTS = np.array([21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 445, 993, 995,
                       1723, 3306, 3389, 5900, 8080, 8443, 8000, 5985, 161, 389, 636, 1433, 1521])


def _flows(rng, n, t0, src, dst, dport, proto, dur, fpk, bpk, fby, bby, ttl, ttl_std,
           win, label, syn=None, rst=None, ts=None):
    n = int(n)
    if n <= 0:
        return None
    arr = lambda v, dt=None: np.broadcast_to(np.asarray(v, dtype=dt), (n,)).copy()
    fpk, bpk = arr(fpk, float), arr(bpk, float)
    tcp = arr(proto) == 6
    tot = fpk + bpk
    dur = arr(dur, float)
    iat = dur / np.maximum(tot - 1, 1)
    return {
        "Src IP": arr(src, object), "Src Port": rng.integers(1024, 65535, n),
        "Dst IP": arr(dst, object), "Dst Port": arr(dport), "Protocol": arr(proto),
        "ts": np.sort(t0 + rng.uniform(0, W, n)) if ts is None else arr(ts, float),
        "Flow Duration": (dur * 1e6).astype(np.int64),
        "Tot Fwd Pkts": fpk.astype(int), "Tot Bwd Pkts": bpk.astype(int),
        "TotLen Fwd Pkts": arr(fby, float).astype(int), "TotLen Bwd Pkts": arr(bby, float).astype(int),
        "Flow IAT Mean": iat * 1e6, "Flow IAT Std": iat * 1e6 * rng.uniform(0.2, 1.4, n),
        "Flow IAT Max": iat * 1e6 * rng.uniform(1.5, 4.0, n),      # always above the mean, like a real capture
        "FIN Flag Cnt": np.where(tcp, rng.integers(0, 3, n), 0),
        "SYN Flag Cnt": np.where(tcp, 1 if syn is None else arr(syn), 0),
        "RST Flag Cnt": np.where(tcp, (rng.random(n) < 0.03).astype(int) if rst is None else arr(rst), 0),
        "PSH Flag Cnt": np.where(tcp, (tot / 3).astype(int), 0),
        "ACK Flag Cnt": np.where(tcp, np.maximum(tot - 1, 0), 0).astype(int),
        "URG Flag Cnt": np.zeros(n, int),
        "Init Fwd Win Byts": np.where(tcp, arr(win), 0),
        "TTL Mean": arr(ttl, float), "TTL Std": arr(ttl_std, float),
        "Retrans": rng.poisson(0.05, n), "Label": arr(label, object),
    }


# ----------------------------------------------------------------------------- profiles
def benign(rng, n, t0):
    port = rng.choice(_BENIGN_PORTS, n, p=_BENIGN_P)
    udp = np.isin(port, [53, 123])
    internal = (port == 445) | (port == 53) | (rng.random(n) < 0.15)
    dst = np.where(port == 53, "10.0.1.53",
                   np.where(internal, rng.choice(_SERVERS, n), rng.choice(_EXTERNAL, n)))
    fpk = 1 + rng.poisson(np.where(udp, 0.2, rng.lognormal(1.8, 0.8, n)))
    bpk = rng.poisson(np.where(udp, 1.0, fpk * 1.2))
    ttl = np.where(internal, rng.choice([64, 128], n), rng.normal(58, 3, n))
    return _flows(rng, n, t0, rng.choice(_CLIENTS, n), dst, port, np.where(udp, 17, 6),
                  np.where(udp, rng.uniform(0.001, 0.05, n), np.clip(rng.lognormal(0, 1.5, n), 0.001, 300)),
                  fpk, bpk, fpk * rng.lognormal(5.5, 0.8, n), bpk * rng.lognormal(6.3, 1.0, n),
                  ttl, np.where(internal, rng.uniform(0, 1, n), rng.uniform(0, 8, n)),
                  rng.choice([64240, 65535, 29200, 8192], n), "Benign")


def dns_burst(rng, n, t0):
    return _flows(rng, n, t0, rng.choice(_CLIENTS[:10], n), "10.0.1.53", 53, 17,
                  rng.uniform(0.001, 0.05, n), 1, 1, rng.integers(60, 120, n), rng.integers(80, 400, n),
                  64, rng.uniform(0, 1, n), 0, "Benign")


def backup(rng, n, t0):
    return _flows(rng, n, t0, rng.choice(_CLIENTS, n), "10.0.1.10", rng.choice([445, 22], n), 6,
                  rng.uniform(20, 59, n), rng.integers(2000, 20000, n), rng.integers(500, 5000, n),
                  rng.integers(5_000_000, 60_000_000, n), rng.integers(30_000, 300_000, n),
                  rng.choice([64, 128], n), rng.uniform(0, 1, n), 64240, "Benign")


def it_scanner(rng, n, t0):
    """Benign-intent internal vulnerability scanner: broad, constant-TTL, no follow-up."""
    return _flows(rng, n, t0, "10.0.1.250", rng.choice(_CLIENTS + _SERVERS, n), rng.choice(_TOP_PORTS, n), 6,
                  rng.uniform(0.0005, 0.02, n), rng.integers(1, 3, n), rng.integers(0, 2, n),
                  rng.integers(40, 120, n), rng.integers(0, 60, n), 64, 0.0, 29200, "PortScan",
                  rst=(rng.random(n) < 0.7).astype(int))


def recon(rng, n, t0, attacker, narrowing):
    focus = rng.random(n) < narrowing
    port = np.where(focus, rng.choice([22, 445, 3389, 80, 21, 139], n), rng.integers(1, 65535, n))
    return _flows(rng, n, t0, attacker, rng.choice(_CLIENTS + _SERVERS, n), port, 6,
                  rng.uniform(0.0001, 0.01, n), rng.integers(1, 3, n), rng.integers(0, 2, n),
                  rng.integers(40, 100, n), rng.integers(0, 60, n),
                  rng.choice([37, 45, 52, 115, 250], n), 0.0, rng.choice([1024, 2048, 3072, 4096], n),
                  "PortScan", rst=(rng.random(n) < 0.8).astype(int))


def brute(rng, n, t0, attacker, victim, port):
    label = {22: "SSH-Bruteforce", 21: "FTP-BruteForce", 80: "Brute Force -Web", 3389: "RDP-Bruteforce"}[port]
    fpk = rng.integers(8, 22, n)
    return _flows(rng, n, t0, attacker, victim, port, 6, rng.uniform(1, 5, n), fpk, fpk - rng.integers(1, 4, n),
                  fpk * rng.integers(60, 140, n), fpk * rng.integers(50, 200, n),
                  rng.normal(52, 1.5, n), rng.uniform(3, 7, n), rng.choice([29200, 64240], n), label,
                  rst=(rng.random(n) < 0.3).astype(int))


def lateral(rng, n, t0, pivot):
    fpk = rng.integers(5, 400, n)
    return _flows(rng, n, t0, pivot, rng.choice(_CLIENTS + _SERVERS, n),
                  rng.choice([445, 135, 5985, 22, 3389], n, p=[0.45, 0.15, 0.15, 0.1, 0.15]), 6,
                  rng.uniform(0.5, 20, n), fpk, rng.integers(3, 200, n), fpk * rng.integers(200, 1400, n),
                  rng.integers(500, 50_000, n), 128, rng.uniform(0, 1, n), 64240, "Infilteration")


def c2(rng, t0, pivot, server, period, phase):
    ts = t0 + np.arange(phase, W, period) + rng.normal(0, 0.15, int(np.ceil((W - phase) / period)))
    n = len(ts)
    return _flows(rng, n, t0, pivot, server, 8443 if period < 8 else 443, 6, rng.uniform(0.1, 0.4, n),
                  rng.integers(4, 7, n), rng.integers(3, 6, n), rng.integers(300, 700, n),
                  rng.integers(200, 1600, n), rng.normal(49, 1, n), rng.uniform(4, 6, n), 64240, "Bot",
                  ts=np.clip(ts, t0, t0 + W - 1e-3))


def exfil(rng, n, t0, pivot, server):
    return _flows(rng, n, t0, pivot, server, rng.choice([443, 21, 53], n), 6, rng.uniform(20, 59, n),
                  rng.integers(5000, 40000, n), rng.integers(1000, 8000, n),
                  rng.integers(10_000_000, 90_000_000, n), rng.integers(50_000, 500_000, n),
                  rng.normal(49, 1, n), rng.uniform(4, 6, n), 64240, "Exfiltration")


def ddos(rng, n, t0, target):
    return _flows(rng, n, t0, rng.choice(_EXTERNAL, n), target, 80, 6, rng.uniform(0.001, 0.5, n),
                  rng.integers(1, 4, n), rng.integers(0, 2, n), rng.integers(60, 400, n),
                  rng.integers(0, 200, n), rng.normal(110, 12, n), rng.uniform(0, 3, n),
                  rng.choice([512, 1024, 8192], n), "DDOS attack-HOIC", rst=(rng.random(n) < 0.4).astype(int))


# ----------------------------------------------------------------------------- episodes
def episode(rng, cfg: dict, start_ts: float) -> pd.DataFrame:
    nw = int(cfg["windows_per_episode"])
    sched: dict[int, list] = {w: [] for w in range(nw)}

    def add(w, fn, *a):
        if 0 <= w < nw:
            sched[w].append((fn, a))

    campaign_end = -1
    if rng.random() < cfg["campaign_prob"]:
        attacker, c2srv = rng.choice(_EXTERNAL, 2, replace=False)
        victim, pivot = rng.choice(_SERVERS), rng.choice(_CLIENTS)
        s = int(rng.integers(25, 80))
        if rng.random() < cfg["stealth_prob"]:
            ia = s
            for w in range(ia - int(rng.integers(2, 5)), ia):
                add(w, lambda r, t0: dns_burst(r, int(r.integers(30, 60)), t0))
                if rng.random() < 0.5:
                    add(w, lambda r, t0, a=attacker: recon(r, int(r.integers(1, 3)), t0, a, 1.0))
        else:
            rl = int(rng.integers(8, 21))
            for i in range(rl):
                frac = (i + 1) / rl
                n = int(2 + 40 * frac ** 1.5 * rng.uniform(0.7, 1.3))
                add(s + i, lambda r, t0, n=n, f=frac, a=attacker: recon(r, n, t0, a, f))
            ia = s + rl + int(rng.integers(0, 4))
        port = int(rng.choice([22, 21, 80, 3389]))
        il = int(rng.integers(4, 11))
        for w in range(ia, ia + il):
            add(w, lambda r, t0, a=attacker, v=victim, p=port: brute(r, int(r.integers(15, 61)), t0, a, v, p))
        c2s, c2l = ia + il, int(rng.integers(15, 36))
        period, phase = float(rng.uniform(5, 12)), float(rng.uniform(0, 3))
        for w in range(c2s, c2s + c2l):
            add(w, lambda r, t0, pv=pivot, sv=c2srv, pe=period, ph=phase: c2(r, t0, pv, sv, pe, ph))
        ls = c2s + int(rng.integers(1, 5))
        ll = int(rng.integers(3, 11))
        for w in range(ls, ls + ll):
            add(w, lambda r, t0, pv=pivot: lateral(r, int(r.integers(10, 41)), t0, pv))
        es = ls + ll + int(rng.integers(1, 6))
        el = int(rng.integers(2, 7))
        for w in range(es, es + el):
            add(w, lambda r, t0, pv=pivot, sv=c2srv: exfil(r, int(r.integers(1, 5)), t0, pv, sv))
        campaign_end = max(c2s + c2l, es + el)
        if rng.random() < 0.3:
            ds = es + el + int(rng.integers(1, 6))
            for w in range(ds, ds + int(rng.integers(3, 9))):
                add(w, lambda r, t0, v=victim: ddos(r, int(r.integers(200, 501)), t0, v))
                campaign_end = max(campaign_end, w)
        busy = range(s - 25, campaign_end + 5)
    else:
        busy = range(0)

    free = [w for w in range(5, nw - 12) if w not in busy]
    if rng.random() < cfg["decoy_recon_prob"] and free:
        d0 = int(rng.choice(free))
        for w in range(d0, d0 + int(rng.integers(3, 11))):
            add(w, lambda r, t0: it_scanner(r, int(r.integers(20, 81)), t0))
    for _ in range(int(rng.integers(1, 4))):
        if free:
            add(int(rng.choice(free)), lambda r, t0: backup(r, int(r.integers(1, 4)), t0))
    for _ in range(int(rng.integers(1, 4))):
        if free:
            w0 = int(rng.choice(free))
            for w in range(w0, w0 + int(rng.integers(1, 4))):
                add(w, lambda r, t0: dns_burst(r, int(r.integers(30, 60)), t0))

    parts = []
    base = cfg["benign_flows_per_window"]
    for w in range(nw):
        t0 = start_ts + w * W
        load = base * (1 + 0.35 * np.sin(2 * np.pi * w / nw)) * rng.uniform(0.85, 1.15)
        parts.append(benign(rng, rng.poisson(load), t0))
        for fn, a in sched[w]:
            parts.append(fn(rng, t0, *a))
    parts = [p for p in parts if p is not None]
    df = pd.concat([pd.DataFrame(p) for p in parts], ignore_index=True).sort_values("ts", kind="stable")
    return df


def to_cic_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out["Timestamp"] = pd.to_datetime(out["ts"], unit="s").dt.strftime("%d/%m/%Y %H:%M:%S")
    out[CIC_COLUMNS].to_csv(path, index=False, float_format="%.3f")


def to_ctu_csv(df: pd.DataFrame, path: Path) -> None:
    """CTU-13 binetflow layout (Argus): flags encoded in State, botnet labels."""
    flags = np.where(df["SYN Flag Cnt"] > 0, "S", "") + np.where(df["FIN Flag Cnt"] > 0, "F", "") \
        + np.where(df["PSH Flag Cnt"] > 0, "P", "") + np.where(df["ACK Flag Cnt"] > 0, "A", "") \
        + np.where(df["RST Flag Cnt"] > 0, "R", "")
    lab = np.where(df["Label"] == "Benign", "flow=Background-Established-cmpgw-CVUT",
                   "flow=From-Botnet-V42-" + df["Label"].astype(str))
    out = pd.DataFrame({
        "StartTime": pd.to_datetime(df["ts"], unit="s").dt.strftime("%Y/%m/%d %H:%M:%S.%f"),
        "Dur": df["Flow Duration"] / 1e6, "Proto": np.where(df["Protocol"] == 17, "udp", "tcp"),
        "SrcAddr": df["Src IP"], "Sport": df["Src Port"], "Dir": "->", "DstAddr": df["Dst IP"],
        "Dport": df["Dst Port"], "State": np.where(df["Protocol"] == 17, "CON", flags),
        "sTos": 0, "dTos": 0, "TotPkts": df["Tot Fwd Pkts"] + df["Tot Bwd Pkts"],
        "TotBytes": df["TotLen Fwd Pkts"] + df["TotLen Bwd Pkts"], "SrcBytes": df["TotLen Fwd Pkts"],
        "Label": lab,
    })
    out.to_csv(path, index=False)


def generate(out_dir: str | Path, cfg: dict, seed: int = 42, episodes: int | None = None,
             episodes_per_file: int = 12, prefix: str = "synthetic_cic2018") -> list[Path]:
    """Write episodes to CIC-IDS2018-style daily CSVs; episodes are 1 day apart."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(episodes or cfg["episodes"])
    base = pd.Timestamp("2018-02-14 08:00:00").timestamp()
    files, buf = [], []
    for e in range(n):
        buf.append(episode(rng, cfg, base + e * 86400.0))
        if len(buf) == episodes_per_file or e == n - 1:
            path = out_dir / f"{prefix}_part{len(files):02d}.csv"
            to_cic_csv(pd.concat(buf, ignore_index=True), path)
            files.append(path)
            buf = []
    return files


def make_sample_pcap(path: str | Path, seed: int = 3, minutes: int = 40) -> Path:
    """Small packet capture: benign web/DNS, then scan -> SSH brute force -> beaconing."""
    from .pcap_reader import Packet, write_pcap

    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2018-02-20 10:00:00").timestamp()
    pk: list[Packet] = []

    def tcp_session(ts, src, dst, sport, dport, ttl, n, win=64240, payload=200, rtt=0.02):
        seq = int(rng.integers(0, 2**31))
        pk.append(Packet(ts, src, dst, sport, dport, 6, ttl, 40, 0x02, win, seq))
        pk.append(Packet(ts + rtt, dst, src, dport, sport, 6, 60, 40, 0x12, 65535, 0))
        for i in range(n):
            t = ts + rtt * (i + 2)
            pk.append(Packet(t, src, dst, sport, dport, 6, ttl, 40 + payload, 0x18, win, seq + i * payload, payload))
            pk.append(Packet(t + rtt / 2, dst, src, dport, sport, 6, 60, 40, 0x10, 65535, 0))
        pk.append(Packet(ts + rtt * (n + 3), src, dst, sport, dport, 6, ttl, 40, 0x11, win, seq))

    for m in range(minutes):
        base = t0 + m * 60
        for _ in range(rng.poisson(12)):
            tcp_session(base + rng.uniform(0, 58), rng.choice(_CLIENTS[:30]), rng.choice(_EXTERNAL[:50]),
                        int(rng.integers(1024, 65535)), 443, 64, int(rng.integers(2, 8)))
        for _ in range(rng.poisson(5)):
            ts = base + rng.uniform(0, 59)
            c = rng.choice(_CLIENTS[:30])
            sp = int(rng.integers(1024, 65535))
            pk.append(Packet(ts, c, "10.0.1.53", sp, 53, 17, 64, 60, payload=32))
            pk.append(Packet(ts + 0.004, "10.0.1.53", c, 53, sp, 17, 64, 120, payload=92))
        if 10 <= m < 20:  # low-and-slow scan ramp with OS-probe TTL spread
            for _ in range(2 + 4 * (m - 10)):
                ts = base + rng.uniform(0, 59)
                ttl = int(rng.choice([37, 45, 52, 115, 250]))
                dp = int(rng.choice([22, 445, 3389]) if rng.random() < (m - 10) / 10 else rng.integers(1, 65535))
                dst = rng.choice(_CLIENTS + _SERVERS)
                pk.append(Packet(ts, "203.0.113.66", dst, 40000, dp, 6, ttl, 44, 0x02, 1024, 0))
                pk.append(Packet(ts + 0.001, dst, "203.0.113.66", dp, 40000, 6, 64, 40, 0x14, 0, 0))
        if 21 <= m < 28:  # SSH brute force with retransmissions
            for _ in range(20):
                ts = base + rng.uniform(0, 55)
                sp = int(rng.integers(1024, 65535))
                tcp_session(ts, "203.0.113.66", "10.0.1.5", sp, 22, 52, 6, win=29200, payload=90)
                pk.append(Packet(ts + 0.2, "203.0.113.66", "10.0.1.5", sp, 22, 6, 52, 130, 0x18, 29200,
                                 pk[-3].seq, 90))
        if m >= 28:  # Command & Control (C2) beaconing every ~7 s
            for k in range(8):
                tcp_session(base + k * 7.2 + rng.normal(0, 0.1), "10.0.0.17", "198.51.100.23",
                            int(rng.integers(1024, 65535)), 8443, 64, 2, payload=300)
    pk.sort(key=lambda p: p.ts)
    write_pcap(path, pk)
    return Path(path)
