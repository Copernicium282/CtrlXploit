"""Kill-chain stages and their MITRE ATT&CK mapping"""
from __future__ import annotations

STAGES = [
    "Benign",
    "Reconnaissance",
    "InitialAccess",
    "LateralMovement",
    "CommandAndControl",
    "Exfiltration",
    "Impact",
]
STAGE_TO_ID = {s: i for i, s in enumerate(STAGES)}
N_STAGES = len(STAGES)

# Stages that constitute active exploitation / compromise. Reconnaissance is a
# precursor: forecasting it is the early-warning signal, not the target.
EXPLOIT_STAGES = [2, 3, 4, 5, 6]

MITRE = {
    "Benign": {"tactic": "-", "id": "-", "techniques": []},
    "Reconnaissance": {
        "tactic": "Reconnaissance", "id": "TA0043",
        "techniques": ["T1595 Active Scanning", "T1046 Network Service Discovery"],
    },
    "InitialAccess": {
        "tactic": "Initial Access / Credential Access", "id": "TA0001",
        "techniques": ["T1110 Brute Force", "T1190 Exploit Public-Facing Application"],
    },
    "LateralMovement": {
        "tactic": "Lateral Movement", "id": "TA0008",
        "techniques": ["T1021 Remote Services (SMB/RDP/SSH)", "T1570 Lateral Tool Transfer"],
    },
    "CommandAndControl": {
        "tactic": "Command and Control", "id": "TA0011",
        "techniques": ["T1071 Application Layer Protocol", "T1573 Encrypted Channel (beaconing)"],
    },
    "Exfiltration": {
        "tactic": "Exfiltration", "id": "TA0010",
        "techniques": ["T1041 Exfiltration Over C2 Channel", "T1048 Exfiltration Over Alternative Protocol"],
    },
    "Impact": {
        "tactic": "Impact", "id": "TA0040",
        "techniques": ["T1498 Network Denial of Service", "T1499 Endpoint DoS"],
    },
}

STAGE_COLORS = {
    "Benign": "#2e7d32",
    "Reconnaissance": "#f9a825",
    "InitialAccess": "#ef6c00",
    "LateralMovement": "#d84315",
    "CommandAndControl": "#8e24aa",
    "Exfiltration": "#c62828",
    "Impact": "#4a148c",
}


def map_label(raw: str) -> int:
    """Map a raw dataset label (CSE-CIC-IDS2017/2018, CTU-13, UNSW, custom) to a stage id."""
    s = str(raw).strip().lower()
    if s in STAGE_TO_ID_LOWER:
        return STAGE_TO_ID_LOWER[s]
    if s in ("", "nan", "benign") or (("background" in s or "normal" in s) and "botnet" not in s):
        return 0  # CTU-13: Background / To-Background-CVUT-DNS-Server / From-Normal-... are all benign
    if "botnet" in s:
        return _ctu_botnet_stage(s)
    if s == "bot" or "c&c" in s or "c2" in s or "beacon" in s:
        return STAGE_TO_ID["CommandAndControl"]
    if "scan" in s or "recon" in s or "probe" in s or "portsweep" in s or "analysis" in s:
        return STAGE_TO_ID["Reconnaissance"]
    if "infilt" in s or "lateral" in s or "smb" in s:
        return STAGE_TO_ID["LateralMovement"]
    if "exfil" in s or "backdoor" in s:
        return STAGE_TO_ID["Exfiltration"]
    if "dos" in s or "ddos" in s or "loic" in s or "hoic" in s or "slowloris" in s or "hulk" in s or "heartbleed" in s:
        return STAGE_TO_ID["Impact"]
    if ("brute" in s or "patator" in s or "sql" in s or "xss" in s or "web attack" in s
            or "exploit" in s or "shellcode" in s or "fuzzer" in s):
        return STAGE_TO_ID["InitialAccess"]
    if "worm" in s:
        return STAGE_TO_ID["LateralMovement"]
    return STAGE_TO_ID["InitialAccess"]  # unknown malicious label -> treat as compromise attempt


STAGE_TO_ID_LOWER = {s.lower(): i for i, s in enumerate(STAGES)}


def _ctu_botnet_stage(s: str) -> int:
    """CTU-13 annotates every malicious flow with its behaviour, e.g.
    'flow=From-Botnet-V42-TCP-CC16-HTTP-Not-Encrypted'. Mapping onto ATT&CK
    (We follow ATT&CK T1496 and treat spam / click-fraud as resource hijacking = Impact):

      SPAM / SMTP / click-fraud ads / web browsing by the bot -> Impact (T1496 Resource Hijacking)
      DNS lookups, connection Attempts, ICMP, scans              -> Reconnaissance (T1590 / T1595 / T1046)
      CC / custom-encrypted / IRC / P2P / proxy / established,
      inbound To-Botnet, binary download (T1105)                 -> Command and Control
    Lateral movement is added by a behavioural rule (internal SMB/RPC/RDP/SSH)."""
    if any(k in s for k in ("spam", "smtp", "-ad-", "http-ad", "click", "google", "web-established",
                            "microsoft", "yieldmanager")):
        return STAGE_TO_ID["Impact"]
    if any(k in s for k in ("dns", "attempt", "icmp", "scan")):
        return STAGE_TO_ID["Reconnaissance"]
    return STAGE_TO_ID["CommandAndControl"]


LATERAL_PORTS = (22, 23, 135, 139, 445, 3389, 5900, 5985)
