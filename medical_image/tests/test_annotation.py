import pytest

from medical_image.utils.annotation import Annotation, GeometryType


class TestGeometryTypeEnum:
    def test_rectangle_exists(self):
        assert GeometryType.RECTANGLE is not None

    def test_ellipse_exists(self):
        assert GeometryType.ELLIPSE is not None

    def test_polygon_exists(self):
        assert GeometryType.POLYGON is not None

    def test_bounding_box_alias(self):
        assert GeometryType.BOUNDING_BOX is GeometryType.RECTANGLE


class TestAnnotationCenter:
    def test_rectangle_center(self):
        ann = Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "mass")
        assert ann.center == (20.0, 30.0)

    def test_ellipse_center(self):
        ann = Annotation(GeometryType.ELLIPSE, [50.0, 60.0, 10.0, 20.0], "calc")
        assert ann.center == (50.0, 60.0)

    def test_polygon_center(self):
        ann = Annotation(
            GeometryType.POLYGON,
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            "region",
        )
        assert ann.center == (5.0, 5.0)

    def test_bounding_box_alias_center(self):
        ann = Annotation(GeometryType.BOUNDING_BOX, [0, 0, 100, 100], "test")
        assert ann.center == (50.0, 50.0)


class TestAnnotationBoundingBox:
    def test_rectangle_bbox(self):
        ann = Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "mass")
        assert ann.get_bounding_box() == [10, 20, 30, 40]

    def test_ellipse_bbox(self):
        ann = Annotation(GeometryType.ELLIPSE, [50.0, 60.0, 10.0, 20.0], "calc")
        assert ann.get_bounding_box() == [40, 40, 60, 80]

    def test_polygon_bbox(self):
        ann = Annotation(
            GeometryType.POLYGON,
            [(5, 10), (15, 5), (25, 20), (10, 25)],
            "region",
        )
        assert ann.get_bounding_box() == [5, 5, 25, 25]


class TestAnnotationROI:
    def test_bbox_roi_no_padding(self):
        ann = Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "mass")
        roi = ann.get_roi()
        assert roi["type"] == "bbox"
        assert roi["coordinates"] == [10, 20, 30, 40]

    def test_bbox_roi_with_padding(self):
        ann = Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "mass")
        roi = ann.get_roi(padding=5)
        assert roi["coordinates"] == [5, 15, 35, 45]

    def test_rectangle_roi_type(self):
        ann = Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "mass")
        roi = ann.get_roi(roi_type="rectangle")
        assert roi["type"] == "rectangle"
        assert roi["coordinates"] == [10, 20, 30, 40]

    def test_ellipse_roi_type(self):
        ann = Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "mass")
        roi = ann.get_roi(roi_type="ellipse")
        assert roi["type"] == "ellipse"
        assert roi["coordinates"]["center"] == (20.0, 30.0)
        assert roi["coordinates"]["radii"] == (10.0, 10.0)

    def test_roi_clamping(self):
        ann = Annotation(GeometryType.RECTANGLE, [0, 0, 50, 50], "mass")
        roi = ann.get_roi(padding=10, image_shape=(40, 40))
        assert roi["coordinates"] == [0, 0, 40, 40]

    def test_invalid_roi_type(self):
        ann = Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "mass")
        with pytest.raises(ValueError, match="Unknown roi_type"):
            ann.get_roi(roi_type="circle")


class TestAnnotationSerialization:
    def test_rectangle_round_trip(self):
        ann = Annotation(
            GeometryType.RECTANGLE, [10, 20, 30, 40], "mass", {"key": "val"}
        )
        d = ann.to_dict()
        restored = Annotation.from_dict(d)
        assert restored.shape == GeometryType.RECTANGLE
        assert restored.coordinates == [10, 20, 30, 40]
        assert restored.label == "mass"
        assert restored.center == ann.center
        assert restored.metadata == {"key": "val"}

    def test_polygon_round_trip(self):
        coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
        ann = Annotation(GeometryType.POLYGON, coords, "region")
        d = ann.to_dict()
        # Coordinates should be lists in dict
        assert d["coordinates"] == [[0, 0], [10, 0], [10, 10], [0, 10]]
        restored = Annotation.from_dict(d)
        assert restored.shape == GeometryType.POLYGON
        assert restored.coordinates == coords

    def test_ellipse_round_trip(self):
        ann = Annotation(GeometryType.ELLIPSE, [50.0, 60.0, 10.0, 20.0], "calc")
        d = ann.to_dict()
        restored = Annotation.from_dict(d)
        assert restored.shape == GeometryType.ELLIPSE
        assert restored.coordinates == [50.0, 60.0, 10.0, 20.0]

    def test_to_dict_contains_center_and_bbox(self):
        ann = Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "mass")
        d = ann.to_dict()
        assert "center" in d
        assert "bounding_box" in d
        assert d["center"] == [20.0, 30.0]
        assert d["bounding_box"] == [10, 20, 30, 40]


class TestAnnotationValidation:
    def test_rectangle_invalid_length(self):
        with pytest.raises(ValueError, match="Rectangle"):
            Annotation(GeometryType.RECTANGLE, [1, 2, 3], "bad")

    def test_ellipse_invalid_length(self):
        with pytest.raises(ValueError, match="Ellipse"):
            Annotation(GeometryType.ELLIPSE, [1, 2], "bad")

    def test_polygon_too_few_points(self):
        with pytest.raises(ValueError, match="Polygon"):
            Annotation(GeometryType.POLYGON, [(0, 0), (1, 1)], "bad")


class TestAnnotationRepr:
    def test_repr_no_error(self):
        ann = Annotation(GeometryType.RECTANGLE, [10, 20, 30, 40], "mass")
        r = repr(ann)
        assert "mass" in r
        assert "RECTANGLE" in r
        assert "center" in r
