import json
import os
import tempfile

import pytest
import numpy as np

from medical_image.data.in_memory_image import InMemoryImage
from medical_image.data.image import image_from_json
from medical_image.utils.annotation import Annotation, GeometryType


class TestImageAnnotations:
    def test_annotations_default_none(self):
        img = InMemoryImage(width=100, height=100)
        assert img.annotations is None

    def test_add_annotation(self):
        img = InMemoryImage(width=100, height=100)
        ann = Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "mass")
        img.add_annotation(ann)
        assert img.annotations is not None
        assert len(img.annotations) == 1
        assert img.annotations[0] is ann

    def test_add_multiple_annotations(self):
        img = InMemoryImage(width=100, height=100)
        ann1 = Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "mass")
        ann2 = Annotation(GeometryType.ELLIPSE, [50.0, 50.0, 10.0, 10.0], "calc")
        img.add_annotation(ann1)
        img.add_annotation(ann2)
        assert len(img.annotations) == 2

    def test_remove_annotation(self):
        img = InMemoryImage(width=100, height=100)
        ann = Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "mass")
        img.add_annotation(ann)
        removed = img.remove_annotation(0)
        assert removed is ann
        assert len(img.annotations) == 0

    def test_remove_annotation_out_of_range(self):
        img = InMemoryImage(width=100, height=100)
        with pytest.raises(IndexError):
            img.remove_annotation(0)

    def test_clone_copies_annotations(self):
        img = InMemoryImage(array=np.zeros((100, 100)))
        ann = Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "mass")
        img.add_annotation(ann)
        cloned = img.clone()
        assert cloned.annotations is not None
        assert len(cloned.annotations) == 1
        # Deep copy: annotations are independent objects
        assert cloned.annotations[0] is not ann
        assert cloned.annotations[0].label == ann.label
        assert cloned.annotations[0].coordinates == ann.coordinates
        # Different list
        assert cloned.annotations is not img.annotations
        # Mutating clone's annotation does NOT affect original
        cloned.annotations[0].metadata["test"] = True
        assert "test" not in ann.metadata

    def test_clone_none_annotations(self):
        img = InMemoryImage(array=np.zeros((100, 100)))
        cloned = img.clone()
        assert cloned.annotations is None

    def test_source_image_copies_annotations(self):
        img = InMemoryImage(width=100, height=100)
        ann = Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "mass")
        img.add_annotation(ann)
        img2 = InMemoryImage(source_image=img)
        # Annotations are deep-copied (independent from source)
        assert img2.annotations is not img.annotations
        assert len(img2.annotations) == 1
        assert img2.annotations[0].label == ann.label
        # Adding to copy does not affect original
        img2.add_annotation(Annotation(GeometryType.RECTANGLE, [0, 0, 5, 5], "test"))
        assert len(img.annotations) == 1


class TestImageJsonSerialization:
    def test_to_json_no_annotations(self):
        img = InMemoryImage(width=200, height=300)
        json_str = img.to_json()
        data = json.loads(json_str)
        assert data["width"] == 200
        assert data["height"] == 300
        assert data["annotations"] == []
        assert data["image_type"] == "InMemoryImage"

    def test_to_json_with_annotations(self):
        img = InMemoryImage(width=200, height=300)
        img.add_annotation(Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "mass"))
        img.add_annotation(
            Annotation(GeometryType.POLYGON, [(0, 0), (5, 0), (5, 5)], "calc")
        )
        json_str = img.to_json()
        data = json.loads(json_str)
        assert len(data["annotations"]) == 2
        assert data["annotations"][0]["shape"] == "RECTANGLE"
        assert data["annotations"][1]["shape"] == "POLYGON"

    def test_to_json_writes_file(self):
        img = InMemoryImage(width=100, height=100)
        img.add_annotation(Annotation(GeometryType.RECTANGLE, [1, 2, 3, 4], "test"))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            img.to_json(file_path=path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert len(data["annotations"]) == 1
        finally:
            os.unlink(path)

    def test_from_json_string(self):
        img = InMemoryImage(width=200, height=300)
        img.add_annotation(Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "mass"))
        json_str = img.to_json()

        restored = InMemoryImage.from_json(json_str)
        assert restored.width == 200
        assert restored.height == 300
        assert restored.annotations is not None
        assert len(restored.annotations) == 1
        assert restored.annotations[0].label == "mass"
        assert restored.annotations[0].shape == GeometryType.RECTANGLE

    def test_from_json_file(self):
        img = InMemoryImage(width=100, height=100)
        img.add_annotation(
            Annotation(GeometryType.ELLIPSE, [50.0, 50.0, 10.0, 20.0], "calc")
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            img.to_json(file_path=path)
            restored = InMemoryImage.from_json(path)
            assert restored.annotations is not None
            assert len(restored.annotations) == 1
            assert restored.annotations[0].shape == GeometryType.ELLIPSE
        finally:
            os.unlink(path)

    def test_round_trip_all_geometry_types(self):
        img = InMemoryImage(width=500, height=500)
        img.add_annotation(Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "rect"))
        img.add_annotation(
            Annotation(GeometryType.ELLIPSE, [50.0, 50.0, 10.0, 20.0], "ell")
        )
        img.add_annotation(
            Annotation(
                GeometryType.POLYGON,
                [(0, 0), (10, 0), (10, 10), (0, 10)],
                "poly",
            )
        )
        json_str = img.to_json()
        restored = InMemoryImage.from_json(json_str)
        assert len(restored.annotations) == 3
        assert restored.annotations[0].shape == GeometryType.RECTANGLE
        assert restored.annotations[1].shape == GeometryType.ELLIPSE
        assert restored.annotations[2].shape == GeometryType.POLYGON


class TestImageFromJsonFactory:
    def test_factory_dispatches_to_correct_class(self):
        img = InMemoryImage(width=100, height=100)
        img.add_annotation(Annotation(GeometryType.RECTANGLE, [1, 2, 3, 4], "test"))
        json_str = img.to_json()
        restored = image_from_json(json_str)
        assert type(restored).__name__ == "InMemoryImage"
        assert restored.annotations is not None
        assert len(restored.annotations) == 1
