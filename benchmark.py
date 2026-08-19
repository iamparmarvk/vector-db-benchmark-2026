import time
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

NUM_VECTORS = 10000
DIMENSIONS = 1536
NUM_QUERIES = 200

print(f"Generating {NUM_VECTORS} synthetic {DIMENSIONS}-dim vectors...")
vectors = np.random.rand(NUM_VECTORS, DIMENSIONS).astype(np.float32)
tenants = [f"org_{i % 10}" for i in range(NUM_VECTORS)]

# ==========================
# 1. QDRANT BENCHMARK
# ==========================
print("\n--- Testing Qdrant ---")
qdrant = QdrantClient("http://localhost:6333")
qdrant.recreate_collection(
    collection_name="benchmark_kb",
    vectors_config=VectorParams(size=DIMENSIONS, distance=Distance.COSINE, on_disk=True)
)

t0 = time.time()
points = [
    PointStruct(id=idx, vector=vectors[idx].tolist(), payload={"tenant_id": tenants[idx]})
    for idx in range(NUM_VECTORS)
]
qdrant.upsert(collection_name="benchmark_kb", points=points, wait=True)
print(f"Qdrant Ingest Time: {time.time() - t0:.2f}s")

qdrant_latencies = []
for _ in range(NUM_QUERIES):
    q_vec = np.random.rand(DIMENSIONS).tolist()
    start = time.perf_counter()
    _ = qdrant.search(
        collection_name="benchmark_kb",
        query_vector=q_vec,
        query_filter=Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value="org_1"))]),
        limit=5
    )
    qdrant_latencies.append((time.perf_counter() - start) * 1000)

print(f"Qdrant P50 Latency: {np.percentile(qdrant_latencies, 50):.2f} ms")
print(f"Qdrant P99 Latency: {np.percentile(qdrant_latencies, 99):.2f} ms")

# ==========================
# 2. PGVECTOR BENCHMARK
# ==========================
print("\n--- Testing pgvector ---")
conn = psycopg2.connect("dbname=vector_db user=postgres password=postgres host=localhost port=5432")
cur = conn.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute("DROP TABLE IF EXISTS benchmark_embeddings;")
cur.execute(f"CREATE TABLE benchmark_embeddings (id INT PRIMARY KEY, tenant_id VARCHAR(20), embedding vector({DIMENSIONS}));")
conn.commit()

t0 = time.time()
insert_data = [(i, tenants[i], vectors[i].tolist()) for i in range(NUM_VECTORS)]
execute_values(cur, "INSERT INTO benchmark_embeddings (id, tenant_id, embedding) VALUES %s", insert_data)
conn.commit()

cur.execute("CREATE INDEX ON benchmark_embeddings USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);")
conn.commit()
print(f"pgvector Ingest + HNSW Index Time: {time.time() - t0:.2f}s")

pg_latencies = []
for _ in range(NUM_QUERIES):
    q_vec = str(np.random.rand(DIMENSIONS).tolist())
    start = time.perf_counter()
    cur.execute(
        "SELECT id FROM benchmark_embeddings WHERE tenant_id = 'org_1' ORDER BY embedding <=> %s LIMIT 5;",
        (q_vec,)
    )
    _ = cur.fetchall()
    pg_latencies.append((time.perf_counter() - start) * 1000)

print(f"pgvector P50 Latency: {np.percentile(pg_latencies, 50):.2f} ms")
print(f"pgvector P99 Latency: {np.percentile(pg_latencies, 99):.2f} ms")

cur.close()
conn.close()
