"""CLIP semantic embedding enums."""

from enum import Enum


class CLIPModelType(Enum):
    """Supported CLIP model variants."""

    OPENAI_CLIP_VIT_B_32 = "openai/clip-vit-base-patch32"
    OPENAI_CLIP_VIT_B_16 = "openai/clip-vit-base-patch16"
    OPENAI_CLIP_VIT_L_14 = "openai/clip-vit-large-patch14"
    OPENAI_CLIP_VIT_L_14_336 = "openai/clip-vit-large-patch14-336"


class SimilarityMetric(Enum):
    """Similarity calculation methods for embeddings."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


class ClusteringMethod(Enum):
    """Clustering algorithms for semantic grouping."""

    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
