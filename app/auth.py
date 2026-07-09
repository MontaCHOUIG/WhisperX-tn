from fastapi import Header, HTTPException, status
from .config import settings


async def require_api_key(x_api_key: str = Header(default=None)):
    if settings.API_KEY is None:
        return  # auth disabled — dev mode only, do not run this in production
    if x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
        )
