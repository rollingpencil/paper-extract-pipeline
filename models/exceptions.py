from fastapi import HTTPException, status

from utils.logger import log


class SourceTypeError(HTTPException):
    def __init__(
        self,
        detail="Unsupported source type.",
        status_code=status.HTTP_400_BAD_REQUEST,
    ):
        log.warning(f"SourceTypeError: {detail}")
        super().__init__(status_code, detail)


class PaperFetchError(HTTPException):
    def __init__(self, detail, status_code=status.HTTP_400_BAD_REQUEST):
        log.warning(f"PaperFetchError: {detail}")
        super().__init__(status_code, detail)


class ExtractionError(HTTPException):
    def __init__(
        self,
        detail="Failed to extract from the paper.",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        log.warning(f"ExtractionError: {detail}")
        super().__init__(status_code, detail)


class PaperAlreadyExistsError(HTTPException):
    def __init__(self, detail, status_code=status.HTTP_400_BAD_REQUEST):
        log.warning(f"PaperAlreadyExistsError: {detail}")
        super().__init__(status_code, detail)
