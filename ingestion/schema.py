from dataclasses import dataclass
from datetime import datetime

@dataclass
class TrafficEvent:
    timestamp: datetime
    src_ip: str
    method: str
    uri_path: str
    status_code: int
    payload_size: int
    response_time_ms: float
    user_agent: str
