"""Attack simulator module for ModelSentinel."""

from .xlsb_attack import EICAR_PAYLOAD, embed_payload, embedding_rate, extract_payload

__all__ = ["embed_payload", "embedding_rate", "extract_payload", "EICAR_PAYLOAD"]
