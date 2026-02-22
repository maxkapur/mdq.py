from fastembed import TextEmbedding
from platformdirs import user_cache_path
from rich.console import Console

cache_dir = user_cache_path("mdq", ensure_exists=True)

console = Console(stderr=True)
conn_path = cache_dir / "cache.db"
embed_model = TextEmbedding(
    "BAAI/bge-small-en-v1.5", cache_dir=cache_dir / "text_embedding"
)
query_prefix = "query: "
