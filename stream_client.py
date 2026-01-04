import random
import asyncio
import httpx

API_INGEST = "http://localhost:8000/ingest"
API_SCORE = "http://localhost:8000/score"
API_UPDATE = "http://localhost:8000/update"

ips = ["192.168.1.10", "10.0.0.5", "127.0.0.1"]

def gen_vec():
    return [
        random.uniform(5, 120),
        random.uniform(1, 20),
        random.uniform(20, 200),
        random.uniform(0.1, 4.0),
        random.uniform(0, 0.3),
        random.uniform(0, 0.2),
        random.uniform(0, 1.0),
        random.uniform(0.001, 0.1),
        random.uniform(0.1, 2.0),
        random.uniform(0.1, 2.5),
        random.uniform(10, 500),
    ]

async def stream():
    print("🚀 Streaming client started (FAST MODE)...")

    async with httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=200)) as client:
        count = 0
        while True:
            vec = gen_vec()
            ip = random.choice(ips)
            is_fraud = random.random() < 0.10

            # Fire ingest request
            await client.post(API_INGEST, json={"features": vec, "src_ip": ip, "is_fraud": is_fraud})

            # Score after 200 samples
            if count > 200:
                await client.post(API_SCORE, json={"features": vec, "src_ip": ip})

            # Adaptive update every 200 samples
            if count > 0 and count % 200 == 0:
                await client.post(API_UPDATE, json={})

            count += 1

            # Remove this delay or keep very tiny for stress testing
            if count % 50 == 0:
                print(f"[TPS CHECK] sent {count} events")

            await asyncio.sleep(0.001)  # 1000 TPS target approx

asyncio.run(stream())
