import json
import os
import tempfile
from typing import Dict, Any, List

import pytest
import torch

from medical_image.datasets.base_dataset import BaseDataset
from medical_image.utils.annotation import Annotation, GeometryType


class MockDataset(BaseDataset):
    """A minimal dataset for testing COCO export/import."""

    def __init__(self, images, annotations_map=None):
        self._images = images
        self._annotations_map = annotations_map or {}
        # Skip parent __init__ to avoid scanning a real directory
        self._samples = list(range(len(images)))
        self.transform = None
        self.target_transform = None
        self.target_size = None

    def _build_sample_list(self):
        pass

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        img_info = self._images[idx]
        return {
            "image": torch.zeros(1, img_info["height"], img_info["width"]),
            "metadata": {
                "file_name": img_info["file_name"],
            },
        }

    def _get_annotations(self, idx: int) -> List[Annotation]:
        return self._annotations_map.get(idx, [])


class TestCocoExport:
    def _make_dataset(self):
        images = [
            {"file_name": "img_001.dcm", "width": 200, "height": 300},
            {"file_name": "img_002.dcm", "width": 400, "height": 500},
        ]
        annotations_map = {
            0: [
                Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "mass"),
                Annotation(
                    GeometryType.POLYGON,
                    [(5, 5), (15, 5), (15, 15), (5, 15)],
                    "calcification",
                ),
            ],
            1: [
                Annotation(GeometryType.ELLIPSE, [100.0, 200.0, 50.0, 30.0], "mass"),
            ],
        }
        return MockDataset(images, annotations_map)

    def test_coco_structure(self):
        ds = self._make_dataset()
        coco = ds.to_coco_json()

        assert "info" in coco
        assert "images" in coco
        assert "annotations" in coco
        assert "categories" in coco

    def test_coco_images(self):
        ds = self._make_dataset()
        coco = ds.to_coco_json()

        assert len(coco["images"]) == 2
        assert coco["images"][0]["file_name"] == "img_001.dcm"
        assert coco["images"][0]["width"] == 200
        assert coco["images"][0]["height"] == 300
        assert coco["images"][1]["file_name"] == "img_002.dcm"

    def test_coco_annotations_count(self):
        ds = self._make_dataset()
        coco = ds.to_coco_json()

        assert len(coco["annotations"]) == 3

    def test_coco_bbox_format(self):
        """COCO bbox is [x, y, width, height], not [x_min, y_min, x_max, y_max]."""
        ds = self._make_dataset()
        coco = ds.to_coco_json()

        rect_ann = coco["annotations"][0]
        # Original: [10, 20, 30, 40] => COCO: [10, 20, 20, 20]
        assert rect_ann["bbox"] == [10, 20, 20, 20]

    def test_coco_center_field(self):
        ds = self._make_dataset()
        coco = ds.to_coco_json()

        rect_ann = coco["annotations"][0]
        assert rect_ann["center"] == [20.0, 30.0]

    def test_coco_categories(self):
        ds = self._make_dataset()
        coco = ds.to_coco_json()

        cat_names = {c["name"] for c in coco["categories"]}
        assert "mass" in cat_names
        assert "calcification" in cat_names

    def test_coco_segmentation_polygon(self):
        ds = self._make_dataset()
        coco = ds.to_coco_json()

        poly_ann = coco["annotations"][1]  # The polygon annotation
        assert len(poly_ann["segmentation"]) == 1
        flat = poly_ann["segmentation"][0]
        assert flat == [5, 5, 15, 5, 15, 15, 5, 15]

    def test_coco_segmentation_rectangle(self):
        ds = self._make_dataset()
        coco = ds.to_coco_json()

        rect_ann = coco["annotations"][0]
        seg = rect_ann["segmentation"][0]
        # Rectangle corners: [10,20, 30,20, 30,40, 10,40]
        assert seg == [10, 20, 30, 20, 30, 40, 10, 40]

    def test_coco_segmentation_ellipse(self):
        ds = self._make_dataset()
        coco = ds.to_coco_json()

        ell_ann = coco["annotations"][2]  # The ellipse annotation
        seg = ell_ann["segmentation"][0]
        # 36 points * 2 coords = 72
        assert len(seg) == 72

    def test_coco_annotation_ids_unique(self):
        ds = self._make_dataset()
        coco = ds.to_coco_json()

        ids = [a["id"] for a in coco["annotations"]]
        assert len(ids) == len(set(ids))

    def test_coco_image_ids_match(self):
        ds = self._make_dataset()
        coco = ds.to_coco_json()

        img_ids = {img["id"] for img in coco["images"]}
        ann_img_ids = {a["image_id"] for a in coco["annotations"]}
        assert ann_img_ids.issubset(img_ids)

    def test_coco_write_to_file(self):
        ds = self._make_dataset()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ds.to_coco_json(output_path=path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert len(data["annotations"]) == 3
        finally:
            os.unlink(path)

    def test_no_annotations(self):
        images = [{"file_name": "empty.dcm", "width": 100, "height": 100}]
        ds = MockDataset(images)
        coco = ds.to_coco_json()
        assert len(coco["images"]) == 1
        assert len(coco["annotations"]) == 0
        assert len(coco["categories"]) == 0


class TestCocoImport:
    def test_from_coco_json_round_trip(self):
        images = [
            {"file_name": "img_001.dcm", "width": 200, "height": 300},
        ]
        annotations_map = {
            0: [
                Annotation(GeometryType.RECTANGLE, [10, 20, 50, 60], "mass"),
                Annotation(
                    GeometryType.POLYGON,
                    [(5, 5), (15, 5), (15, 15), (5, 15)],
                    "calc",
                ),
            ],
        }
        ds = MockDataset(images, annotations_map)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ds.to_coco_json(output_path=path)
            result = BaseDataset.from_coco_json(path)

            assert len(result["images"]) == 1
            assert 1 in result["annotations"]
            anns = result["annotations"][1]
            assert len(anns) == 2

            # Rectangle was exported as segmentation polygon (4 corners)
            # so it comes back as POLYGON
            labels = {a.label for a in anns}
            assert "mass" in labels
            assert "calc" in labels

            assert len(result["categories"]) == 2
        finally:
            os.unlink(path)

    def test_from_coco_json_fallback_to_bbox(self):
        """When segmentation is empty, fall back to bbox."""
        coco = {
            "info": {},
            "licenses": [],
            "categories": [{"id": 1, "name": "lesion"}],
            "images": [{"id": 1, "file_name": "test.dcm", "width": 100, "height": 100}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [[]],
                    "bbox": [10, 20, 30, 40],
                    "area": 1200,
                    "iscrowd": 0,
                }
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(coco, f)
            path = f.name
        try:
            result = BaseDataset.from_coco_json(path)
            anns = result["annotations"][1]
            assert len(anns) == 1
            assert anns[0].shape == GeometryType.RECTANGLE
            # bbox [10,20,30,40] => [x_min, y_min, x_min+w, y_min+h] = [10,20,40,60]
            assert anns[0].coordinates == [10, 20, 40, 60]
        finally:
            os.unlink(path)
