import time
import random
import requests
import numpy as np

API_INGEST = "http://localhost:8000/ingest"
API_SCORE = "http://localhost:8000/score"
API_UPDATE = "http://localhost:8000/update"

ips = ["192.168.1.10", "10.0.0.5", "127.0.0.1"]

def gen_vec():
    """Generate 11 numeric ML features matching the extractor output"""
    return [
        random.uniform(5, 120),     # req_rate
        random.uniform(1, 20),     # unique_uri_count
        random.uniform(20, 200),   # payload_size_mean
        random.uniform(0.1, 4.0),  # payload_entropy
        random.uniform(0, 0.3),    # error_rate_4xx
        random.uniform(0, 0.2),    # error_rate_5xx
        random.uniform(0, 1.0),    # burstiness
        random.uniform(0.001, 0.1),# endpoint_rarity
        random.uniform(0.1, 2.0),  # interarrival_mean
        random.uniform(0.1, 2.5),  # interarrival_std
        random.uniform(10, 500),   # avg_response_time
    ]

print("🚀 Starting streaming client simulator...\n")

count = 0
while True:
    vec = gen_vec()
    ip = random.choice(ips)

    # Send to ingest
    r = requests.post(API_INGEST, json={"features": vec, "src_ip": ip})
    print("[INGEST]", r.json())

    # After baseline is ready, score requests
    if count > 200:  # inference should already be enabled by now
        r2 = requests.post(API_SCORE, json={"features": vec, "src_ip": ip})
        print("[SCORE]", r2.json())

    # Trigger adaptive update every 200 samples
    if count > 0 and count % 200 == 0:
        r3 = requests.post(API_UPDATE, json={})
        print("[UPDATE]", r3.json())

    count += 1
    # time.sleep(0.2)  # 5 requests per second approx
