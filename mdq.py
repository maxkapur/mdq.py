import numpy as np
from fastembed import TextEmbedding


def main():
    model = TextEmbedding()
    docs = ["test1", "test2"]
    embeddings = np.array(list(model.embed(docs)))
    print(embeddings.shape)


if __name__ == "__main__":
    main()
