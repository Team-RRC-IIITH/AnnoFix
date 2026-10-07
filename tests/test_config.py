from pathlib import Path

import pytest

from annofix.config import LabelsError, load_labels

REPO = Path(__file__).resolve().parents[1]


def write(tmp_path, text):
    p = tmp_path / "labels.yaml"
    p.write_text(text)
    return p


def test_example_labels_load():
    labels = load_labels(REPO / "labels.yaml")
    assert labels.materials[0] == "plastic"
    assert labels.parts_for("mug") == ("body", "handle")
    assert labels.parts_for(None) == () and labels.parts_for("unknown") == ()


def test_object_without_part_suggestions(tmp_path):
    labels = load_labels(write(tmp_path, "materials: [metal]\nobjects:\n  widget:\n"))
    assert labels.parts_for("widget") == ()


@pytest.mark.parametrize("text, message", [
    ("objects: {}\n", "materials"),
    ("materials: []\n", "at least one"),
    ("materials: [metal, metal]\n", "duplicate"),
    ("materials: [metal, 3]\n", "non-empty strings"),
    ("materials: [metal]\nobjects: [mug]\n", "mapping"),
    ("- just a list\n", "top level"),
])
def test_invalid_files_are_rejected(tmp_path, text, message):
    with pytest.raises(LabelsError, match=message):
        load_labels(write(tmp_path, text))


def test_unknown_classes_and_materials_are_reported():
    from annofix.annotation import AnnObject, Annotation, Part

    labels = load_labels(REPO / "labels.yaml")
    ann = Annotation.empty(2, 2)
    ann.objects = [AnnObject(1, "spaceship", [Part(2, "wing", "unobtainium"), Part(3, "free text", "metal")]),
                   AnnObject(4, "mug", [Part(5, None, None)])]
    assert labels.unknown_names(ann) == ["object 1: class 'spaceship' is not in labels.yaml",
                                         "part 2: material 'unobtainium' is not in labels.yaml"]


def test_broken_yaml_is_a_labels_error(tmp_path):
    with pytest.raises(LabelsError, match="YAML"):
        load_labels(write(tmp_path, "materials: [metal, glass\n"))
