"""Prompt dataset: loads prompts (+ optional reference images) from JSONL or plain text."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset


@dataclass
class PromptItem:
    prompt: str
    reference_image_path: Optional[str] = None


class PromptDataset(Dataset):
    """Minimal dataset that yields (prompt, optional_reference_path) pairs.

    Supports two formats:
    - JSONL: each line is ``{"prompt": "...", "reference": "optional/path.png"}``
    - Plain text (.txt): one prompt per line, no reference images
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        if path.suffix == ".jsonl":
            self.items = self._load_jsonl(path)
        elif path.suffix == ".json":
            self.items = self._load_json(path)
        else:
            self.items = self._load_txt(path)

    @staticmethod
    def _load_json(path: Path) -> list[PromptItem]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
        items = []
        for obj in data:
            if isinstance(obj, str):
                items.append(PromptItem(prompt=obj))
            else:
                items.append(PromptItem(prompt=obj["prompt"], reference_image_path=obj.get("reference")))
        return items

    @staticmethod
    def _load_jsonl(path: Path) -> list[PromptItem]:
        items = []
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {lineno} of {path}: {e}") from e
                items.append(PromptItem(prompt=obj["prompt"], reference_image_path=obj.get("reference")))
        return items

    @staticmethod
    def _load_txt(path: Path) -> list[PromptItem]:
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                prompt = line.strip()
                if prompt:
                    items.append(PromptItem(prompt=prompt))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> PromptItem:
        return self.items[idx]
