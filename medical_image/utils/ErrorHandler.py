class AppError(Exception):
    """Base exception for all application-specific errors."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self):
        if self.code:
            return f"[Error {self.code}] {self.message}"
        return self.message


class FileNotFoundAppError(AppError):
    pass


class InvalidPixelDataError(AppError):
    pass


class LengthMismatchError(AppError):
    pass


class InvalidInputTypeError(AppError):
    pass


class UnsupportedFileTypeError(AppError):
    pass


class DicomDataNotLoadedError(AppError):
    pass


class EmptyDatasetError(AppError):
    pass


class AnnotationTypeError(AppError):
    pass


class ErrorMessages:
    @staticmethod
    def file_not_found(path):
        return FileNotFoundAppError(f"Unable to locate file at: {path}", code=404)

    @staticmethod
    def length_mismatch(
        expected_len, actual_len, expected_name="coordinates", actual_name="classes"
    ):
        return LengthMismatchError(
            f"Length mismatch: {expected_name} ({expected_len}) and {actual_name} ({actual_len}) must be the same length.",
            code=400,
        )

    @staticmethod
    def invalid_pixel_data():
        return InvalidPixelDataError(
            "pixel_data is not a valid Torch Tensor.", code=422
        )

    @staticmethod
    def invalid_input_type(expected_type, received_type):
        return InvalidInputTypeError(
            f"Invalid input type: expected {expected_type}, got {received_type}.",
            code=422,
        )

    @staticmethod
    def input_none(expected_type):
        return InvalidInputTypeError(f"{expected_type} must be provided.", code=422)

    @staticmethod
    def unsupported_file_type(extension, expected=".dcm"):
        return UnsupportedFileTypeError(
            f"Unsupported file type: '{extension}'. Only {expected} files are supported.",
            code=415,
        )

    @staticmethod
    def dicom_data_not_loaded():
        return DicomDataNotLoadedError(
            "DICOM data has not been loaded yet. Call load() method first.", code=500
        )

    @staticmethod
    def empty_dataset():
        return EmptyDatasetError(
            "Dataset is Empty or not loaded properly. Call `load_dataset` method first.",
            code=500,
        )

    @staticmethod
    def annotation_type_not_recognized(received, expected=None):
        if expected is None:
            expected = {
                "AnnotationType.BOUNDING_BOX",
                "AnnotationType.POLYGON",
                "AnnotationType.MASK",
            }

        expected_names = ", ".join(sorted(e.name for e in expected))
        return AnnotationTypeError(
            f"Unrecognized annotation type: '{received}'. Supported types are: {expected_names}.",
            code=415,
        )
