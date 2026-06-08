"""
Ground-truth schema for image extraction evaluation.

Each image has one JSON file describing:
  - the document type (banking, news, ...)
  - the list of atomic entities to test under Binding Accuracy (Threat 1)
  - the list of content blocks to test under Content Fidelity (Threat 2)

Schema:
{
  "doc_type": "banking",
  "image_path": "data/ui_dataset/images/pil/banking_0000.png",
  "entities": [
    {
      "label": "account_holder",
      "type": "name",
      "value": "Ella Thompson",
      "question": "What is the account holder name?"
    },
    ...
  ],
  "content_blocks": [
    {"label": "transaction_history", "text": "2024-10-01 Groceries -123.45 ..."}
  ]
}

`question` is optional on each entity — entities without a question are
evaluated only under Token Presence (search in transcription), not Binding
Accuracy (targeted query). In practice all entities should have questions.

`content_blocks` may be an empty list for pure-field domains (identity).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


ENTITY_TYPES: frozenset[str] = frozenset(
    {"name", "digit_seq", "amount", "date", "date_range", "short_phrase"}
)


@dataclass
class Entity:
    """One atomic protected fact."""
    label: str
    type: str
    value: str
    question: str = ""   # targeted query used for Binding Accuracy (Threat 1)

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError(f"Entity.label must be a non-empty string, got {self.label!r}")
        if self.type not in ENTITY_TYPES:
            raise ValueError(
                f"Entity.type for label={self.label!r} must be one of "
                f"{sorted(ENTITY_TYPES)}, got {self.type!r}"
            )
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError(
                f"Entity.value for label={self.label!r} must be a non-empty "
                f"string, got {self.value!r}"
            )


@dataclass
class ContentBlock:
    """One block of text whose verbatim reproduction we measure."""
    label: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError(f"ContentBlock.label must be a non-empty string, got {self.label!r}")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(
                f"ContentBlock.text for label={self.label!r} must be a non-empty "
                f"string, got {self.text!r}"
            )


@dataclass
class GroundTruth:
    """Ground truth for a single image."""
    doc_type: str
    image_path: str
    entities: list[Entity] = field(default_factory=list)
    content_blocks: list[ContentBlock] = field(default_factory=list)

    REQUIRED_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"doc_type", "image_path", "entities", "content_blocks"}
    )

    def __post_init__(self) -> None:
        if not isinstance(self.doc_type, str) or not self.doc_type.strip():
            raise ValueError(f"doc_type must be a non-empty string, got {self.doc_type!r}")
        if not isinstance(self.image_path, str) or not self.image_path.strip():
            raise ValueError(f"image_path must be a non-empty string, got {self.image_path!r}")
        labels = [e.label for e in self.entities]
        if len(labels) != len(set(labels)):
            duplicates = [l for l in labels if labels.count(l) > 1]
            raise ValueError(f"Duplicate entity labels in GT: {sorted(set(duplicates))}")
        block_labels = [b.label for b in self.content_blocks]
        if len(block_labels) != len(set(block_labels)):
            duplicates = [l for l in block_labels if block_labels.count(l) > 1]
            raise ValueError(f"Duplicate content_block labels in GT: {sorted(set(duplicates))}")

    @classmethod
    def from_dict(cls, data: dict) -> "GroundTruth":
        missing = cls.REQUIRED_KEYS - set(data.keys())
        if missing:
            raise ValueError(f"Missing required keys in ground-truth dict: {sorted(missing)}")
        entities = []
        for e in data["entities"]:
            entities.append(Entity(
                label=e["label"],
                type=e["type"],
                value=e["value"],
                question=e.get("question", ""),
            ))
        content_blocks = [ContentBlock(**b) for b in data["content_blocks"]]
        return cls(
            doc_type=data["doc_type"],
            image_path=data["image_path"],
            entities=entities,
            content_blocks=content_blocks,
        )

    @classmethod
    def from_json(cls, path: "Path | str") -> "GroundTruth":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Ground-truth file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        try:
            return cls.from_dict(data)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid ground-truth file {path}: {e}") from e


def validate_gt(path: "Path | str") -> list[str]:
    """Load a GT file, return list of problems or [] if clean."""
    problems: list[str] = []
    try:
        gt = GroundTruth.from_json(path)
        missing_q = [e.label for e in gt.entities if not e.question]
        if missing_q:
            problems.append(
                f"WARN: {len(missing_q)} entities have no question "
                f"(will be Token Presence only): {missing_q}"
            )
    except (FileNotFoundError, ValueError) as e:
        problems.append(str(e))
    return problems