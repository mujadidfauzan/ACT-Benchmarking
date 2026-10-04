"""Frozen MiniLM sentence encoding and cached embedding lookup."""

import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import nn


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class FrozenMiniLMEncoder(nn.Module):
    """Encode short instructions with a frozen, normalized MiniLM embedding."""

    def __init__(self, model_name=DEFAULT_MODEL_NAME, revision=None, device="cpu"):
        super().__init__()
        self.model_name = model_name
        self.revision = revision
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=device,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @property
    def embedding_dim(self):
        return int(self.model.get_embedding_dimension())

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    def tokenize(self, instructions):
        return self.model.preprocess(list(instructions))

    @torch.inference_mode()
    def encode(self, instructions, batch_size=32):
        if isinstance(instructions, str):
            instructions = [instructions]
        instructions = list(instructions)
        if not instructions:
            raise ValueError("instructions cannot be empty")
        embeddings = self.model.encode(
            instructions,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return embeddings.to(dtype=torch.float32)


class FrozenMiniLMTokenEncoder(FrozenMiniLMEncoder):
    """Return frozen MiniLM token embeddings before sentence pooling."""

    @torch.inference_mode()
    def encode_tokens(self, instructions, batch_size=32):
        if isinstance(instructions, str):
            instructions = [instructions]
        instructions = list(instructions)
        if not instructions:
            raise ValueError("instructions cannot be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        sequences = []
        input_ids = []
        tokens = []
        for start in range(0, len(instructions), batch_size):
            batch = instructions[start : start + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                batch_size=len(batch),
                show_progress_bar=False,
                output_value="token_embeddings",
                convert_to_numpy=False,
                convert_to_tensor=False,
            )
            tokenized = self.model.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.model.max_seq_length,
                return_tensors="pt",
            )
            batch_mask = tokenized["attention_mask"].bool()
            batch_ids = tokenized["input_ids"]
            for index in range(len(batch)):
                length = int(batch_mask[index].sum().item())
                embedding = batch_embeddings[index].to(torch.float32).cpu()
                if len(embedding) != length:
                    raise ValueError(
                        "MiniLM token embeddings and tokenizer output do not align"
                    )
                sequences.append(embedding.clone())
                input_ids.append(batch_ids[index, :length].clone())
                tokens.append(
                    self.model.tokenizer.convert_ids_to_tokens(
                        batch_ids[index, :length].tolist()
                    )
                )
        return {
            "token_embeddings": sequences,
            "input_ids": input_ids,
            "tokens": tokens,
        }


class CachedLanguageEmbeddings:
    """Read-only mapping from exact instruction text to frozen embeddings."""

    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as archive:
            required = {"instructions", "embeddings", "metadata_json"}
            missing = sorted(required - set(archive.files))
            if missing:
                raise KeyError(f"{self.path} is missing keys: {missing}")
            instructions = archive["instructions"].astype(str).tolist()
            embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
            metadata_raw = archive["metadata_json"].item()

        if embeddings.ndim != 2 or len(embeddings) != len(instructions):
            raise ValueError("Language cache instructions and embeddings do not align")
        if len(set(instructions)) != len(instructions):
            raise ValueError("Language cache contains duplicate instructions")
        if not np.isfinite(embeddings).all():
            raise ValueError("Language cache contains NaN or Inf")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("Language cache embeddings are not unit-normalized")

        self.instructions = instructions
        self.embeddings = torch.from_numpy(embeddings.copy())
        self.metadata = json.loads(str(metadata_raw))
        self._indices = {
            instruction: index for index, instruction in enumerate(instructions)
        }

    @property
    def embedding_dim(self):
        return int(self.embeddings.shape[1])

    def __len__(self):
        return len(self.instructions)

    def __contains__(self, instruction):
        return instruction in self._indices

    def encode(self, instructions, device=None):
        single = isinstance(instructions, str)
        values = [instructions] if single else list(instructions)
        missing = [value for value in values if value not in self._indices]
        if missing:
            preview = ", ".join(repr(value) for value in missing[:3])
            raise KeyError(f"Instructions are missing from language cache: {preview}")
        indices = torch.tensor(
            [self._indices[value] for value in values], dtype=torch.long
        )
        result = self.embeddings.index_select(0, indices)
        if device is not None:
            result = result.to(device)
        return result[0] if single else result


class CachedTokenEmbeddings:
    """Read-only lookup for padded token embeddings and attention masks."""

    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as archive:
            required = {
                "instructions",
                "token_embeddings",
                "attention_masks",
                "input_ids",
                "tokens_json",
                "metadata_json",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise KeyError(f"{self.path} is missing keys: {missing}")
            instructions = archive["instructions"].astype(str).tolist()
            embeddings = np.asarray(archive["token_embeddings"], dtype=np.float32)
            masks = np.asarray(archive["attention_masks"], dtype=np.bool_)
            input_ids = np.asarray(archive["input_ids"], dtype=np.int64)
            tokens = json.loads(str(archive["tokens_json"].item()))
            metadata = json.loads(str(archive["metadata_json"].item()))

        if embeddings.ndim != 3:
            raise ValueError("Token embeddings must have shape [N, L, D]")
        if masks.shape != embeddings.shape[:2] or input_ids.shape != masks.shape:
            raise ValueError("Token cache arrays do not align")
        if len(instructions) != len(embeddings) or len(tokens) != len(instructions):
            raise ValueError("Token cache instruction count does not align")
        if len(set(instructions)) != len(instructions):
            raise ValueError("Token cache contains duplicate instructions")
        if not np.isfinite(embeddings).all():
            raise ValueError("Token cache contains NaN or Inf")
        if np.any(masks.sum(axis=1) == 0):
            raise ValueError("Token cache contains an empty sequence")

        self.instructions = instructions
        self.token_embeddings = torch.from_numpy(embeddings.copy())
        self.attention_masks = torch.from_numpy(masks.copy())
        self.input_ids = torch.from_numpy(input_ids.copy())
        self.tokens = tokens
        self.metadata = metadata
        self._indices = {
            instruction: index for index, instruction in enumerate(instructions)
        }

    def __len__(self):
        return len(self.instructions)

    @property
    def embedding_dim(self):
        return int(self.token_embeddings.shape[-1])

    @property
    def max_length(self):
        return int(self.token_embeddings.shape[1])

    def __contains__(self, instruction):
        return instruction in self._indices

    def lookup(self, instructions, device=None):
        single = isinstance(instructions, str)
        values = [instructions] if single else list(instructions)
        missing = [value for value in values if value not in self._indices]
        if missing:
            preview = ", ".join(repr(value) for value in missing[:3])
            raise KeyError(f"Instructions are missing from token cache: {preview}")
        indices = torch.tensor(
            [self._indices[value] for value in values], dtype=torch.long
        )
        result = {
            "token_embeddings": self.token_embeddings.index_select(0, indices),
            "attention_mask": self.attention_masks.index_select(0, indices),
            "input_ids": self.input_ids.index_select(0, indices),
        }
        if device is not None:
            result = {key: value.to(device) for key, value in result.items()}
        if single:
            result = {key: value[0] for key, value in result.items()}
        return result


class TrainableAttentionPooling(nn.Module):
    """Learn a masked weighted sum over frozen language token embeddings."""

    def __init__(self, input_dim=384, output_dim=128, dropout=0.1):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention_score = nn.Linear(output_dim, 1, bias=False)

    def forward(self, token_embeddings, attention_mask):
        if token_embeddings.ndim != 3:
            raise ValueError("token_embeddings must have shape [B, L, D]")
        if token_embeddings.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected token dimension {self.input_dim}, "
                f"got {token_embeddings.shape[-1]}"
            )
        if attention_mask.shape != token_embeddings.shape[:2]:
            raise ValueError("attention_mask must have shape [B, L]")
        mask = attention_mask.bool()
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError("Every instruction must contain at least one token")

        projected = self.projection(token_embeddings)
        scores = self.attention_score(projected).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(~mask, 0.0)
        feature = torch.sum(projected * weights.unsqueeze(-1), dim=1)
        return {
            "feature": feature,
            "attention_weights": weights,
        }
