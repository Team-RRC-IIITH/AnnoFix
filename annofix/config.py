"""labels.yaml: the material list and object classes with suggested part names."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class LabelsError(ValueError):
    pass


@dataclass(frozen=True)
class Labels:
    materials: tuple[str, ...]
    objects: dict[str, tuple[str, ...]]  # class -> suggested part names

    def parts_for(self, cls: str | None) -> tuple[str, ...]:
        return self.objects.get(cls, ()) if cls else ()


def _names(value, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise LabelsError(f"{where}: expected a list of non-empty strings, got {value!r}")
    names = tuple(v.strip() for v in value)
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise LabelsError(f"{where}: duplicate names {dupes}")
    return names


def load_labels(path: str | Path) -> Labels:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise LabelsError(f"{path}: top level must be a mapping with 'materials' and 'objects'")
    materials = _names(data.get("materials"), "materials")
    if not materials:
        raise LabelsError("materials: need at least one material")
    raw_objects = data.get("objects") or {}
    if not isinstance(raw_objects, dict):
        raise LabelsError("objects: expected a mapping of class -> list of part names")
    objects = {str(cls): _names(parts or [], f"objects.{cls}") for cls, parts in raw_objects.items()}
    return Labels(materials=materials, objects=objects)
