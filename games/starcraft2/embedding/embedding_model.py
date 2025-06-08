from typing import List
import os
import logging
from sentence_transformers import SentenceTransformer
from games.starcraft2.sc2_config import MODEL_BASE_PATH

logger = logging.getLogger("EMBEDDING")
logger.setLevel(logging.DEBUG)


class SingletonEmbeddingModel(type):
    """
    Base class for implementing the Singleton pattern
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonEmbeddingModel, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class BaseEmbeddingModel(metaclass=SingletonEmbeddingModel):
    """
    Base class for all embedding models, must implement the embed_text method
    """
    # A required attribute: model_name
    model_name: str = None

    def embed_text(self, input_string: str) -> List[float]:
        pass


class LocalEmbedding(BaseEmbeddingModel):
    """
    Component for embedding text, loads local Huggingface weights according to config and performs inference
    Recommended usage:
        1. uer/sbert-base-chinese-nli
        2. sentence-transformers/all-mpnet-base-v2
        3. sentence-transformers/all-MiniLM-L6-v2
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", vector_width: int = 768):
        #####################################
        # Convert model_name to a local Huggingface folder,
        # e.g., uer/sbert-base-chinese-nli  ==> uer_sbert-base-chinese-nli
        #####################################
        self.model_path_hf = MODEL_BASE_PATH / "embedding" / model_name.replace("/", "_")
        print("model_path_hf", self.model_path_hf)
        self.model_name = model_name
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Load the model locally
        if not os.path.exists(self.model_path_hf):
            logger.info(f"Weights for model {model_name} do not exist, downloading... Target path: {self.model_path_hf}")
            model = SentenceTransformer(model_name)
            model.save(str(self.model_path_hf))
            self.model = model
        else:
            logger.info(f"Weights for model {model_name} exist, loading local weights... Path: {self.model_path_hf}")
            self.model = SentenceTransformer(self.model_path_hf)

        # Get and check vector width
        vector_width_from_weights: int = self.model.get_sentence_embedding_dimension()  # e.g., 768
        assert vector_width == vector_width_from_weights, f"Vector width for model {model_name} is {vector_width_from_weights}, does not match user-specified {vector_width}"
        self.vector_width = vector_width

        logger.info(f"Weights for model {model_name} loaded, vector width is {vector_width_from_weights}")

    def embed_text(self, input_string: str) -> List[float]:
        """
        Embed a given string using the local embedding model.

        Args:
            input_string (str): The input text to be embedded.

        Returns:
            List[float]: The embedding vector for the input string.
        """
        try:
            vector = self.model.encode(input_string).tolist()
        except Exception as e:
            import traceback
            logger.error(f"Error occurred while embedding text: {e}")
            vector = [0.0] * self.vector_width
        return vector


if __name__ == "__main__":
    # os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    # os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    # local embedding
    embedding = LocalEmbedding()
    print(embedding.embed_text("你好"))
    # huggingface embedding

    """ Below are some examples of available models
    "model_id": "uer/sbert-base-chinese-nli",
    "dim": 768,

    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "dim": 384,

    Code example:
    "uer/sbert-base-chinese-nli" ==> LocalEmbedding(model_name="uer/sbert-base-chinese-nli", vector_width=768)
    embedding = LocalEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", vector_width=384)
    # The component will check if there are corresponding weights in the material/embedding folder, if not, it will automatically download (requires internet connection if not available)

    embedding = HuggingFaceEmbedding(model_name=NPC_MEMORY_CONFIG["hf_model_id"], vector_width=NPC_MEMORY_CONFIG["hf_dim"])
    # Online API requests are very unstable and may time out, so it is not recommended to use
    """
