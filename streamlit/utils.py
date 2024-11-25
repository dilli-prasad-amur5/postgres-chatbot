import numpy as np
from typing import List

def normalize_vector(vector: List[float]) -> List[float]:
    """
    Normalize a vector to unit length.
    Args:
        vector (List[float]): Input vector to normalize.
    Returns:
        List[float]: Normalized vector.
    """
    magnitude = np.linalg.norm(vector)
    return [float(x / magnitude) for x in vector] if magnitude != 0 else vector

def sanitize_text(text: str) -> str:
    """
    Sanitize text by removing NULL characters.
    Args:
        text (str): Input text to sanitize.
    Returns:
        str: Sanitized text.
    """
    return text.replace("\x00", "")
