import unittest

import pytest


class TestDicom:

    @pytest.mark.parametrize("s1, s2, expected_distance", mock_levenshtien())
    def test_levenshtein_distance(self, s1, s2, expected_distance):
        # Create DocElement instances
        doc_element1 = DocElement(0, 0, 0, 0, ContentType.TEXT, s1)
        doc_element2 = DocElement(0, 0, 0, 0, ContentType.TEXT, s2)

        # Compute Levenshtein distance
        distance = TextUtils.levenshtein_distance(doc_element1, doc_element2)

        # Check if the computed distance matches the expected distance
        assert distance == expected_distance, f"Distance for {s1} and {s2} is incorrect"