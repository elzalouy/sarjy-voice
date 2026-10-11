"""The fixed Arabic sentences and the helpers that load them."""

from sarjy_agent.eval.corpus import (
    AUDIO_POLICY,
    COMPARISON_RULE,
    AmbiguityKind,
    CorpusCategory,
    CorpusError,
    EvaluationCorpus,
    EvaluationItem,
    NumberKind,
    load_corpus,
    parse_corpus,
    stt_references,
    tts_inputs,
)

__all__ = [
    "AUDIO_POLICY",
    "COMPARISON_RULE",
    "AmbiguityKind",
    "CorpusCategory",
    "CorpusError",
    "EvaluationCorpus",
    "EvaluationItem",
    "NumberKind",
    "load_corpus",
    "parse_corpus",
    "stt_references",
    "tts_inputs",
]
