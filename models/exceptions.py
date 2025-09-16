from fastapi import status, HTTPException


class SourceTypeError(HTTPException):
    def __init__(
        self,
        detail="Unsupported source type.",
        status_code=status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(status_code, detail)


class PaperFetchError(HTTPException):
    def __init__(self, detail, status_code=status.HTTP_400_BAD_REQUEST):
        super().__init__(status_code, detail)


class ExtractionError(HTTPException):
    def __init__(
        self,
        detail="Failed to extract from the paper.",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(status_code, detail)
