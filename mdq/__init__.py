from fastembed import TextEmbedding
from platformdirs import user_cache_path
from rich.console import Console

console = Console(stderr=True)
conn_path = user_cache_path("mdq", ensure_exists=True) / "cache.db"

embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
query_prefix = "query: "
