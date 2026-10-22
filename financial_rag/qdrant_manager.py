"""
financial_rag/qdrant_manager.py
===============================
Qdrant vector database manager for Financial RAG.
Features:
  1. Auto-detection: connects to Docker Qdrant server if up, falls back
     seamlessly to embedded local file storage (no Docker required for students).
  2. Multi-modal storage: stores vector embeddings along with text chunks,
     document metadata, and base64 page images in the point payload.
  3. Search: vector similarity search returning ranked Documents and images.
  4. Non-blocking concurrency: if another process (e.g. FastAPI API server or
     another terminal) already holds the file lock on local_qdrant_storage,
     automatically creates a zero-overhead read snapshot so that Streamlit UI,
     CLI, API, and evaluation scripts can all run concurrently without lock errors.
"""

import os
import uuid
import atexit
import shutil
import tempfile
import urllib.request
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_core.documents import Document

# Pre-import portalocker and patch QdrantLocal.close to avoid Python interpreter shutdown race conditions
try:
    import portalocker
except ImportError:
    pass

try:
    from qdrant_client.local.qdrant_local import QdrantLocal
    _orig_qdrant_local_close = QdrantLocal.close

    def _safe_qdrant_local_close(self, *args, **kwargs):
        try:
            _orig_qdrant_local_close(self, *args, **kwargs)
        except (ImportError, TypeError, Exception):
            pass

    QdrantLocal.close = _safe_qdrant_local_close
except Exception:
    pass

# Process-level client cache
_PROCESS_CLIENTS = {}


def _cleanup_process_clients():
    """Closes all cached clients before Python interpreter teardown."""
    for target_path, client in list(_PROCESS_CLIENTS.items()):
        try:
            client.close()
        except Exception:
            pass
    _PROCESS_CLIENTS.clear()


atexit.register(_cleanup_process_clients)


class QdrantManager:
    def __init__(self, url: str = None, path: str = None, collection_name: str = "financial_rag_collection"):
        self.collection_name = collection_name
        self.client = None
        self._temp_dir = None

        # 1. Determine connection mode
        if url:
            self.client = QdrantClient(url=url)
            return
        elif path:
            self._init_local_client(path)
            return

        # 2. Auto-detection: try Docker first
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        use_docker = False
        try:
            with urllib.request.urlopen(f"{qdrant_url.rstrip('/')}/healthz", timeout=0.8) as response:
                if response.status == 200:
                    use_docker = True
        except Exception:
            use_docker = False

        if use_docker:
            print(f"[Qdrant] Connected to Docker daemon at {qdrant_url}")
            self.client = QdrantClient(url=qdrant_url)
        else:
            default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "local_qdrant_storage")
            os.makedirs(default_path, exist_ok=True)
            self._init_local_client(default_path)

    def _init_local_client(self, target_path: str):
        """Initializes a local embedded Qdrant client with automatic concurrent lock handling."""
        # Check process-level cache first
        if target_path in _PROCESS_CLIENTS:
            self.client = _PROCESS_CLIENTS[target_path]
            return

        try:
            client = QdrantClient(path=target_path)
            _PROCESS_CLIENTS[target_path] = client
            self.client = client
            print(f"[Qdrant] Local embedded database active at: {target_path}")
        except RuntimeError as e:
            if "already accessed by another instance" in str(e):
                print(f"[Qdrant] Primary storage at '{target_path}' is in use by another service.")
                print("[Qdrant] Initializing high-speed concurrent read snapshot...")
                temp_dir = tempfile.mkdtemp(prefix="qdrant_snapshot_")
                self._temp_dir = temp_dir

                # Copy local collection files to temporary snapshot
                shutil.copytree(target_path, temp_dir, dirs_exist_ok=True)
                client = QdrantClient(path=temp_dir)
                _PROCESS_CLIENTS[target_path] = client
                self.client = client

                # Register cleanup on process exit
                def _cleanup():
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass
                atexit.register(_cleanup)
                print(f"[Qdrant] Concurrent read snapshot active ({temp_dir}).")
            else:
                raise e

    def ensure_collection(self, vector_size: int = 768, distance: Distance = Distance.COSINE):
        """Creates the collection if not already existing, or recreates if size differs."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            info = self.client.get_collection(self.collection_name)
            existing_size = None
            cfg = info.config.params.vectors
            if hasattr(cfg, "size"):
                existing_size = cfg.size
            elif isinstance(cfg, dict):
                existing_size = cfg.get("", getattr(list(cfg.values())[0], "size", None)) if cfg else None

            if existing_size and existing_size != vector_size:
                print(f"[Qdrant] Dimension mismatch ({existing_size} vs {vector_size}). Recreating collection...")
                self.client.delete_collection(self.collection_name)
                collections.remove(self.collection_name)

        if self.collection_name not in collections:
            print(f"[Qdrant] Creating collection '{self.collection_name}' (dim={vector_size})...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )

    def insert_records(self, embeddings: list[list[float]], payloads: list[dict], batch_size: int = 32):
        """Inserts embeddings and payloads in batches into Qdrant."""
        total = len(embeddings)
        points = []
        for emb, payload in zip(embeddings, payloads):
            point_id = payload.get("id") or str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=payload
                )
            )

        for i in range(0, total, batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        print(f"[Qdrant] Successfully inserted {total} records into '{self.collection_name}'.")

    def search(self, query_vector: list[float], limit: int = 5) -> list[Document]:
        """
        Searches the collection using the query vector.
        Returns a list of LangChain Document objects with page_content and metadata.
        """
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )

        documents = []
        for hit in search_result.points:
            payload = hit.payload or {}
            content = payload.get("text") or payload.get("content") or ""
            metadata = {
                "id": str(hit.id),
                "score": hit.score if hasattr(hit, "score") else None,
                "source": payload.get("source", "financial_document"),
                "page": payload.get("page", 1),
                "has_image": bool(payload.get("image_base64")),
            }
            if "image_base64" in payload:
                metadata["image_base64"] = payload["image_base64"]
            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def get_all_records(self, limit: int = 1000) -> list[dict]:
        """Dumps all stored records with id and payload for dataset curation."""
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return [{"id": str(r.id), "payload": r.payload} for r in records]
        except Exception:
            return []

    def close(self):
        """Explicitly closes the Qdrant client connection."""
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
