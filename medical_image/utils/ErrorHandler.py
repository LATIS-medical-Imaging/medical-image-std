class ErrorMessages:
    @staticmethod
    def length_mismatch(expected_len, actual_len, expected_name="coordinates", actual_name="classes"):
        return (
            f"Length mismatch: {expected_name} ({expected_len}) and {actual_name} ({actual_len}) "
            f"must be the same length."
        )

    @staticmethod
    def file_not_found(path):
        return FileNotFoundError(f"Unable to locate file at: {path}")

    @staticmethod
    def invalid_input_type(expected_type, received_type):
        return f"Invalid input type: expected {expected_type}, got {received_type}."