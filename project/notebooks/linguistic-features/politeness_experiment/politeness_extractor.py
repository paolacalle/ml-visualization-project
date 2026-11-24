"""Score long texts with a frozen single-label transformer classifier."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import hashlib
import logging
from time import perf_counter

import numpy as np
import pandas as pd


POLITE_GUARD = "Intel/polite-guard"
POLITE_GUARD_REVISION = "2294eaff6b7ba7c238148b6bcea548d9b63ab4df"
POLITENESS_WEIGHTS = {"impolite": 0, "neutral": 1, "somewhat polite": 2, "polite": 3}
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Segmentation:
    content_limit: int
    random_seed: int | None = None

    def __post_init__(self):
        if self.content_limit < 1:
            raise ValueError("The content-token limit must be positive.")

    @property
    def name(self):
        suffix = "fixed" if self.random_seed is None else f"random_{self.random_seed}"
        return f"{self.content_limit}_{suffix}"

    def spans(self, n_tokens: int, argument_id: str) -> list[tuple[int, int]]:
        """Return contiguous token ranges covering the entire argument."""
        limit = self.content_limit
        if self.random_seed is None:
            return [(start, min(start + limit, n_tokens)) for start in range(0, n_tokens, limit)]

        key = f"{self.random_seed}:{argument_id}".encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
        chunks_left = (n_tokens + limit - 1) // limit
        start = 0
        spans = []
        while chunks_left:
            remaining = n_tokens - start
            # Keep the fixed chunk count and leave enough tokens for every later chunk.
            low = max(1, remaining - limit * (chunks_left - 1))
            high = min(limit, remaining - (chunks_left - 1))
            end = start + int(rng.integers(low, high + 1))
            spans.append((start, end))
            start = end
            chunks_left -= 1
        return spans


class BatchedTransformerClassifier:
    """Tokenize once and reuse predictions for identical token IDs within a run."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        *,
        batch_size: int = 16,
        device: str = "cpu",
        checkpoint: Path | None = None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        source = str(checkpoint) if checkpoint is not None else model_name
        settings = {"local_files_only": True} if checkpoint is not None else {"revision": revision}
        self.tokenizer = AutoTokenizer.from_pretrained(source, **settings)
        self.model = AutoModelForSequenceClassification.from_pretrained(source, **settings).eval().to(device)
        if self.model.config.problem_type in {"multi_label_classification", "regression"}:
            raise ValueError("Expected a single-label classifier with categorical probabilities.")
        self.labels = tuple(self.model.config.id2label[i].strip().lower() for i in range(self.model.config.num_labels))
        self.batch_size = batch_size
        self.device = device
        self.torch = torch
        positions = self.model.config.max_position_embeddings
        embeddings = getattr(self.model.base_model, "embeddings", None)
        position_padding = getattr(getattr(embeddings, "position_embeddings", None), "padding_idx", None)
        # RoBERTa starts assigning real positions after its position padding index.
        if position_padding is not None:
            positions -= position_padding + 1
        self.content_limit = min(self.tokenizer.model_max_length, positions)
        self.content_limit -= self.tokenizer.num_special_tokens_to_add(pair=False)
        # Read the single-sequence template once so chunks can use the original token IDs.
        template = self.tokenizer("An example.", return_special_tokens_mask=True)
        content = [index for index, special in enumerate(template["special_tokens_mask"]) if not special]
        self.content_start, self.content_end = content[0], content[-1] + 1
        if content != list(range(self.content_start, self.content_end)):
            raise ValueError("Expected a contiguous single-sequence token template.")
        self.input_template = {key: template[key] for key in ("input_ids", "token_type_ids") if key in template}
        types = self.input_template.get("token_type_ids")
        if types is not None and len(set(types[self.content_start:self.content_end])) != 1:
            raise ValueError("Expected one token type for the template's content.")
        self.cache: dict[tuple[int, ...], np.ndarray] = {}

    def tokenize(self, texts: Mapping[str, str]) -> dict[str, tuple[int, ...]]:
        encoded = self.tokenizer(list(texts.values()), add_special_tokens=False, truncation=False, verbose=False)
        return dict(zip(texts, map(tuple, encoded["input_ids"]), strict=True))

    def _predict(self, chunks: Sequence[tuple[int, ...]]) -> dict[tuple[int, ...], float]:
        unseen = sorted(dict.fromkeys(chunk for chunk in chunks if chunk not in self.cache), key=len)
        times = {}
        with self.torch.inference_mode():
            for start in range(0, len(unseen), self.batch_size):
                batch = unseen[start:start + self.batch_size]
                begin = perf_counter()
                encoded = []
                for ids in batch:
                    inputs = {}
                    for field, template in self.input_template.items():
                        content = list(ids) if field == "input_ids" else [template[self.content_start]] * len(ids)
                        inputs[field] = template[:self.content_start] + content + template[self.content_end:]
                    encoded.append(inputs)
                prepared = self.tokenizer.pad(
                    encoded, padding=True, return_attention_mask=True, return_tensors="pt",
                ).to(self.device)
                probabilities = self.model(**prepared).logits.float().softmax(dim=-1).cpu().numpy()
                per_chunk = (perf_counter() - begin) / len(batch)
                for ids, probability in zip(batch, probabilities, strict=True):
                    self.cache[ids] = probability
                    times[ids] = per_chunk
                processed = start + len(batch)
                if processed == len(unseen) or (start // self.batch_size + 1) % 100 == 0:
                    LOGGER.info("Predicted %s/%s unique uncached chunks", f"{processed:,}", f"{len(unseen):,}")
        return times

    def score(
        self,
        tokenized: Mapping[str, tuple[int, ...]],
        segmentation: Segmentation,
        *,
        probability_prefix: str = "prob",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if segmentation.content_limit > self.content_limit:
            raise ValueError(f"This checkpoint supports at most {self.content_limit} content tokens per chunk.")
        spans = {key: segmentation.spans(len(ids), key) for key, ids in tokenized.items()}
        return self.score_layouts(tokenized, spans, segmentation.name, probability_prefix=probability_prefix)

    def score_layouts(
        self,
        tokenized: Mapping[str, tuple[int, ...]],
        spans: Mapping[str, Sequence[tuple[int, int]]],
        name: str,
        *,
        probability_prefix: str = "prob",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Score explicit layouts while preserving the original content-token IDs."""
        if not tokenized or tokenized.keys() != spans.keys():
            raise ValueError("Token sequences and layouts must cover the same nonempty set of IDs.")
        for key, ids in tokenized.items():
            end = 0
            for start, stop in spans[key]:
                if start != end or not 0 < stop - start <= self.content_limit:
                    raise ValueError(f"Invalid chunk boundaries for {key}.")
                end = stop
            if not ids or end != len(ids):
                raise ValueError(f"Incomplete content-token coverage for {key}.")
        all_chunks = [tokenized[key][start:end] for key, ranges in spans.items() for start, end in ranges]
        times = self._predict(all_chunks)
        probability_fields = [f"{probability_prefix}_{label.replace(' ', '_')}" for label in self.labels]
        arguments, chunks = [], []
        for key, ranges in spans.items():
            ids = tokenized[key]
            probabilities, counts, inference_time = [], [], 0.0
            for index, (start, end) in enumerate(ranges):
                content = ids[start:end]
                probability = self.cache[content]
                elapsed = times.pop(content, None)
                probabilities.append(probability)
                counts.append(end - start)
                inference_time += elapsed or 0.0
                chunks.append({
                    "argument_id": key, "segmentation": name, "chunk_index": index,
                    "token_start": start, "token_end": end, "n_tokens": end - start,
                    "inference_time": elapsed or 0.0, "cache_hit": elapsed is None,
                    **dict(zip(probability_fields, probability, strict=True)),
                })
            aggregate = np.average(probabilities, weights=counts, axis=0)
            arguments.append({
                "argument_id": key, "segmentation": name, "n_chunks": len(ranges),
                "n_tokens": len(ids), "inference_time": inference_time,
                "predicted_class": self.labels[int(aggregate.argmax())],
                **dict(zip(probability_fields, aggregate, strict=True)),
            })
        return pd.DataFrame(arguments), pd.DataFrame(chunks)


def politeness_features(arguments: pd.DataFrame, chunks: pd.DataFrame) -> pd.DataFrame:
    """Add an ordinal politeness proxy and the peak chunk-level impoliteness."""
    weights = pd.Series({f"politeness_prob_{label.replace(' ', '_')}": weight
                         for label, weight in POLITENESS_WEIGHTS.items()})
    peak = chunks.groupby(["argument_id", "segmentation"], sort=False)["politeness_prob_impolite"].max()
    return arguments.assign(politeness_score=arguments[weights.index].dot(weights)).join(
        peak.rename("max_impoliteness"), on=["argument_id", "segmentation"],
    )
