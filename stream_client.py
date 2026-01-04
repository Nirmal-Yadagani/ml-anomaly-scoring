import random, asyncio, httpx, numpy as np, pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

API_INGEST = "http://localhost:8000/ingest"
TABLE = "requests"
LOCAL_DSN = 'postgresql://postgres:postgres@localhost:5432/waf_test'

def gen_vec():
    return [random.uniform(5,120),random.uniform(1,20),random.uniform(20,200),
            random.uniform(0.1,4),random.uniform(0,0.3),random.uniform(0,0.2),
            random.uniform(0,1),random.uniform(0.001,0.1),random.uniform(0.1,2),
            random.uniform(0.1,2.5),random.uniform(10,500)]

def insert(batch):
    with psycopg2.connect(LOCAL_DSN) as conn, conn.cursor() as cur:
        execute_batch(cur, f"INSERT INTO {TABLE} (timestamp,src_ip,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,is_fraud) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", batch)
        conn.commit()

async def stream():
    print("🚀 Streaming fast...\n")
    batch=[]
    async with httpx.AsyncClient(timeout=None,limits=httpx.Limits(max_connections=200)) as c:
        i=0
        while True:
            v=gen_vec();ip=random.choice(["192.168.1.10","10.0.0.5","127.0.0.1"])
            f=random.random()<0.1;ts=pd.Timestamp.utcnow()
            batch.append([ts,ip,*v,f])
            i+=1
            if len(batch)>=500:
                insert(batch);batch.clear();print("[DB] +500")
            await c.post(API_INGEST,json={"features":v,"src_ip":ip,"is_fraud":f})
            if i%1000==0: print(f"[TPS] {i}")
            await asyncio.sleep(0.001)

asyncio.run(stream())
