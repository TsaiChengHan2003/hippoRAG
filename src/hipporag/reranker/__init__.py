from .HFReranker import HFReranker
from ..utils.config_utils import BaseConfig

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def _get_rerank_model_class(config: BaseConfig):
    if "ms-marco-MiniLM-L6-v2" in config.rerank_model_name:
        return HFReranker()
    # elif "nli-deberta-v3-base" in config.rerank_model_name: #但DeBERTa不太能
    #     return DeBERTa

    assert False, f"Unknown rerank model name: {config.rerank_model_name}"