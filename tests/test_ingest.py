import numpy as np
import pandas as pd

from sih_v2.constants import STAGE_TO_ID, map_label
from sih_v2.ingest import read_flows
from sih_v2.ingest.pcap_reader import Packet, read_pcap_flows, write_pcap
from sih_v2.ingest.schema import MISSING_OK, normalize_frame


def test_label_mapping_covers_public_datasets():
    assert map_label("Benign") == 0
    assert map_label("BENIGN") == 0
    assert map_label("flow=Background-UDP-Established") == 0
    assert map_label("flow=To-Background-UDP-CVUT-DNS-Server") == 0
    assert map_label("flow=From-Background-CVUT-Proxy") == 0
    assert map_label("flow=From-Normal-V42-Grill") == 0
    assert map_label("flow=From-Botnet-V42-TCP-Attempt-SPAM") == STAGE_TO_ID["Impact"]
    assert map_label("flow=From-Botnet-V42-UDP-DNS") == STAGE_TO_ID["Reconnaissance"]
    assert map_label("PortScan") == STAGE_TO_ID["Reconnaissance"]
    assert map_label("SSH-Bruteforce") == STAGE_TO_ID["InitialAccess"]
    assert map_label("FTP-Patator") == STAGE_TO_ID["InitialAccess"]
    assert map_label("Infilteration") == STAGE_TO_ID["LateralMovement"]
    assert map_label("Bot") == STAGE_TO_ID["CommandAndControl"]
    assert map_label("flow=From-Botnet-V42-TCP-CC16-HTTP-Not-Encrypted") == STAGE_TO_ID["CommandAndControl"]
    assert map_label("DDOS attack-HOIC") == STAGE_TO_ID["Impact"]
    assert map_label("DoS attacks-Hulk") == STAGE_TO_ID["Impact"]


def test_cic_units_and_bad_rows(tmp_path):
    p = tmp_path / "cic.csv"
    p.write_text(
        "Dst Port,Protocol,Timestamp,Flow Duration,Tot Fwd Pkts,Tot Bwd Pkts,TotLen Fwd Pkts,TotLen Bwd Pkts,"
        "Flow IAT Mean,SYN Flag Cnt,Init Fwd Win Byts,Label\n"
        "22,6,14/02/2018 08:31:01,2000000,10,8,1200,900,5000,1,29200,SSH-Bruteforce\n"
        "Dst Port,Protocol,Timestamp,Flow Duration,Tot Fwd Pkts,Tot Bwd Pkts,TotLen Fwd Pkts,TotLen Bwd Pkts,"
        "Flow IAT Mean,SYN Flag Cnt,Init Fwd Win Byts,Label\n"            # repeated header (real CIC quirk)
        "80,6,14/02/2018 08:31:05,inf,3,2,100,Infinity,NaN,0,-1,Benign\n"  # infinity / NaN / negative
        "443,6,not-a-date,1,1,1,1,1,1,1,1,Benign\n"                        # unparseable timestamp
    )
    df = read_flows(p)
    assert len(df) == 2
    assert np.isclose(df.loc[0, "duration"], 2.0)        # microseconds -> seconds
    assert np.isclose(df.loc[0, "iat_mean"], 0.005)
    assert df.loc[0, "stage"] == STAGE_TO_ID["InitialAccess"]
    assert np.isfinite(df.drop(columns=list(MISSING_OK)).select_dtypes("number").to_numpy()).all()
    assert df["ttl_mean"].isna().all()                   # CICFlowMeter has no TTL -> missing, not 0
    assert df["frag_pkts"].isna().all()                  # ... nor IP fragment flags
    assert df["payload_mean"].isna().all()               # ... nor per-packet payload sizes
    assert df.loc[1, "init_win"] == 0


def test_ctu13_state_flags_and_bytes():
    raw = pd.DataFrame({
        "StartTime": ["2011/08/10 09:46:53.047277"], "Dur": [1.5], "Proto": ["tcp"], "SrcAddr": ["147.32.84.165"],
        "Sport": ["1025"], "Dir": ["->"], "DstAddr": ["60.190.222.139"], "Dport": ["0x0050"], "State": ["FSPA_FSPA"],
        "TotPkts": [10], "TotBytes": [1000], "SrcBytes": [300], "Label": ["flow=From-Botnet-V42-TCP-Established"]})
    df = normalize_frame(raw)
    r = df.iloc[0]
    assert r["syn"] == 2 and r["fin"] == 2 and r["psh"] == 2 and r["ack"] == 2 and r["rst"] == 0
    assert r["fwd_bytes"] == 300 and r["bwd_bytes"] == 700
    assert r["dst_port"] == 0            # hex port coerced safely
    assert r["stage"] == STAGE_TO_ID["CommandAndControl"]
    assert abs(r["ts"] - pd.Timestamp("2011-08-10 09:46:53.047277").timestamp()) < 1e-3


def test_pcap_roundtrip_extracts_packet_features(tmp_path):
    pk = [
        Packet(100.0, "10.0.0.5", "8.8.4.4", 5555, 443, 6, 64, 40, 0x02, 64240, 1),
        Packet(100.01, "8.8.4.4", "10.0.0.5", 443, 5555, 6, 50, 40, 0x12, 65535, 0),
        Packet(100.02, "10.0.0.5", "8.8.4.4", 5555, 443, 6, 64, 140, 0x18, 64240, 2, 100),
        Packet(100.30, "10.0.0.5", "8.8.4.4", 5555, 443, 6, 64, 140, 0x18, 64240, 2, 100),  # retransmission
        Packet(100.40, "10.0.0.5", "8.8.4.4", 5555, 443, 6, 64, 40, 0x11, 64240, 3),
        Packet(101.0, "10.0.0.9", "10.0.1.53", 999, 53, 17, 128, 60, payload=32),
    ]
    f = tmp_path / "t.pcap"
    write_pcap(f, pk)
    df = read_pcap_flows(f).sort_values("ts").reset_index(drop=True)
    assert len(df) == 2
    tcp = df.iloc[0]
    assert tcp["fwd_pkts"] == 4 and tcp["bwd_pkts"] == 1
    assert tcp["syn"] == 2 and tcp["fin"] == 1 and tcp["psh"] == 2
    assert tcp["retrans"] == 1
    assert tcp["init_win"] == 64240
    assert abs(tcp["ttl_mean"] - (64 * 4 + 50) / 5) < 1e-4 and tcp["ttl_std"] > 0
    assert df.iloc[1]["proto"] == 17


def test_pcap_extracts_fragments_payload_sizes_and_iat_max(tmp_path):
    pk = [
        Packet(200.0, "10.0.2.5", "10.0.2.9", 40000, 22, 6, 64, 60, 0x02, 29200, 1, 0, 0x2),
        Packet(200.5, "10.0.2.5", "10.0.2.9", 40000, 22, 6, 64, 1500, 0x18, 29200, 2, 1460, 0x2),
        Packet(203.0, "10.0.2.5", "10.0.2.9", 40000, 22, 6, 64, 1500, 0x18, 29200, 3, 1460, 0x1),      # MF set
        Packet(203.2, "10.0.2.5", "10.0.2.9", 40000, 22, 6, 64, 700, 0x10, 29200, 4, 600, 0x0, 185),  # continuation
        Packet(204.0, "10.0.2.5", "10.0.2.9", 40000, 22, 6, 64, 40, 0x10, 29200, 5, 40),
    ]
    f = tmp_path / "frag.pcap"
    write_pcap(f, pk)
    r = read_pcap_flows(f).iloc[0]
    assert r["frag_pkts"] == 2 and r["df_pkts"] == 2           # MF + continuation, two don't-fragment
    assert r["payload_small_pkts"] == 2                         # bare ACK + 40-byte packet
    assert abs(r["payload_mean"] - (0 + 1460 + 1460 + 600 + 40) / 5) < 1e-3 and r["payload_std"] > 0
    assert abs(r["iat_max"] - 2.5) < 1e-6                       # 200.5 -> 203.0 is the longest gap


def test_sample_files_parse(cfg):
    from sih_v2.config import resolve
    d = resolve(cfg["paths"]["sample_dir"])
    for name in ("sample_cic2018.csv", "sample_ctu13.binetflow", "sample_capture.pcap"):
        if (d / name).exists():
            assert len(read_flows(d / name)) > 100
