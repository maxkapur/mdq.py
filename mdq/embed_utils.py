import numpy as np
from fastembed import TextEmbedding
from numpy.typing import NDArray


class LazyEmbedder:
    def __init__(self, text_embedding_kwargs: dict[str, any]) -> None:
        self.text_embedding_kwargs = text_embedding_kwargs

    @property
    def model(self) -> TextEmbedding:
        if not hasattr(self, "_model"):
            self._model = TextEmbedding(**self.text_embedding_kwargs)
        return self._model

    def embed_one(self, document) -> NDArray[np.float32]:
        """Embed a single document."""

        # TODO: Chunking

        (res,) = self.model.embed([document])
        return res.astype(np.float32)
