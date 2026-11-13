"""Direct image and text embeddings for the small retrieval demonstration.

Uses Qwen's native Sentence Transformers integration:
https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B
Install requirements-embeddings.txt before calling load_embedding_model().
"""
from PIL import Image


def load_embedding_model():
    """Load Qwen once; use a CUDA GPU when available, otherwise a slower CPU."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return SentenceTransformer(
        "Qwen/Qwen3-VL-Embedding-2B",
        device=device,
        model_kwargs={"torch_dtype": dtype, "attn_implementation": "sdpa"},
        processor_kwargs={"min_pixels": 4096, "max_pixels": 512 * 512},
    )


def embed_images(model, image_paths):
    """Read image pixels and return one unit-length vector per image, in order."""
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    if not images:
        raise ValueError("Supply at least one image.")
    return model.encode(
        images, batch_size=1, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    )


def embed_query(model, query):
    """Return a unit-length text vector in the same space as the image vectors."""
    if not query.strip():
        raise ValueError("Supply a nonempty query.")
    return model.encode(
        [query], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
