"""Environment template for BM25s KeywordSearch plugin."""

from typing import Dict, Any

def bm25s_keyword_env_template() -> Dict[str, Any]:
    """Return environment template for BM25s KeywordSearch configuration.
    
    Returns:
        Dictionary containing environment variable template
    """
    return {
        "REFINIRE_RAG_BM25S_INDEX_PATH": {
            "description": "Path to BM25s index file",
            "type": "str",
            "default": "data/bm25s_index.pkl",
            "required": False
        },
        "REFINIRE_RAG_BM25S_K1": {
            "description": "BM25s k1 parameter (term frequency saturation point)",
            "type": "float",
            "default": 1.2,
            "required": False
        },
        "REFINIRE_RAG_BM25S_B": {
            "description": "BM25s b parameter (field length normalization)",
            "type": "float",
            "default": 0.75,
            "required": False
        },
        "REFINIRE_RAG_BM25S_EPSILON": {
            "description": "BM25s epsilon parameter (IDF cutoff parameter)",
            "type": "float",
            "default": 0.25,
            "required": False
        },
        "REFINIRE_RAG_BM25S_METHOD": {
            "description": "BM25s scoring method (bm25, bm25+, bm25l)",
            "type": "str",
            "default": "bm25",
            "required": False
        },
        "REFINIRE_RAG_BM25S_STEMMER": {
            "description": "Stemmer to use (janome for Japanese, None for no stemming)",
            "type": "str",
            "default": "janome",
            "required": False
        },
        "REFINIRE_RAG_BM25S_STOPWORDS": {
            "description": "Stopwords to filter (ja for Japanese stopwords, None for no filtering)",
            "type": "str",
            "default": "ja",
            "required": False
        }
    }