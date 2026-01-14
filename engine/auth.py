"""
Clerk Authentication Middleware for FastAPI

This module handles JWT verification for Clerk-issued tokens.
"""

import os
import logging
from typing import Optional
from functools import lru_cache

import httpx
from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from jwt import PyJWKClient, PyJWKClientError
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


class ClerkUser(BaseModel):
    """Authenticated user information extracted from Clerk JWT."""
    user_id: str
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    image_url: Optional[str] = None
    

class ClerkConfig:
    """Clerk configuration loaded from environment."""
    
    def __init__(self):
        self.secret_key = os.environ.get("CLERK_SECRET_KEY")
        self.publishable_key = os.environ.get("CLERK_PUBLISHABLE_KEY", "")
        
        # Extract Clerk frontend API from publishable key
        # Format: pk_test_xxxxx or pk_live_xxxxx
        # The JWKS URL uses the Clerk frontend API
        if self.publishable_key:
            # Clerk's JWKS endpoint
            self.jwks_url = None  # Will be set dynamically
        
        # Alternatively, set the issuer directly
        self.issuer = os.environ.get("CLERK_ISSUER")  # e.g., https://your-app.clerk.accounts.dev
        
        if self.issuer:
            self.jwks_url = f"{self.issuer}/.well-known/jwks.json"
        
        self.enabled = bool(self.secret_key or self.issuer)
        
        if not self.enabled:
            logger.warning(
                "Clerk authentication is DISABLED. "
                "Set CLERK_SECRET_KEY and CLERK_ISSUER environment variables to enable."
            )


# Global config instance
_config: Optional[ClerkConfig] = None


def get_clerk_config() -> ClerkConfig:
    """Get or create Clerk configuration."""
    global _config
    if _config is None:
        _config = ClerkConfig()
    return _config


# Cache the JWKS client to avoid repeated fetches
_jwks_client: Optional[PyJWKClient] = None


def get_jwks_client() -> Optional[PyJWKClient]:
    """Get or create JWKS client for verifying Clerk JWTs."""
    global _jwks_client
    config = get_clerk_config()
    
    if not config.jwks_url:
        return None
        
    if _jwks_client is None:
        _jwks_client = PyJWKClient(config.jwks_url, cache_keys=True)
    
    return _jwks_client


def verify_clerk_token(token: str) -> Optional[ClerkUser]:
    """
    Verify a Clerk JWT and extract user information.
    
    Args:
        token: The JWT token from the Authorization header
        
    Returns:
        ClerkUser object if valid, None if invalid
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    config = get_clerk_config()
    
    if not config.enabled:
        logger.warning("Auth check skipped - Clerk not configured")
        return None
    
    jwks_client = get_jwks_client()
    if not jwks_client:
        logger.error("JWKS client not available")
        raise HTTPException(status_code=500, detail="Authentication not configured")
    
    try:
        # Get the signing key from Clerk's JWKS
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        
        # Decode and verify the token
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "require": ["exp", "iat", "sub"],
            }
        )
        
        # Extract user info from claims
        user = ClerkUser(
            user_id=payload.get("sub"),
            email=payload.get("email"),
            first_name=payload.get("first_name"),
            last_name=payload.get("last_name"),
            image_url=payload.get("image_url"),
        )
        
        logger.debug(f"Authenticated user: {user.user_id}")
        return user
        
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    except PyJWKClientError as e:
        logger.error(f"JWKS client error: {e}")
        raise HTTPException(status_code=500, detail="Authentication service error")
    except Exception as e:
        logger.error(f"Unexpected auth error: {e}")
        raise HTTPException(status_code=500, detail="Authentication error")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> ClerkUser:
    """
    FastAPI dependency to get the current authenticated user.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: ClerkUser = Depends(get_current_user)):
            return {"user_id": user.user_id}
    """
    config = get_clerk_config()
    
    # If auth is not configured, reject all requests in production
    if not config.enabled:
        # In development, you might want to allow requests
        # For production, always require auth
        dev_mode = os.environ.get("DEV_MODE", "false").lower() == "true"
        if dev_mode:
            logger.warning("DEV_MODE: Returning mock user")
            return ClerkUser(user_id="dev_user_123", email="dev@example.com")
        raise HTTPException(
            status_code=503, 
            detail="Authentication not configured. Set CLERK_SECRET_KEY and CLERK_ISSUER."
        )
    
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = verify_clerk_token(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Optional[ClerkUser]:
    """
    FastAPI dependency to optionally get the current user.
    Returns None if not authenticated (instead of raising an exception).
    
    Usage:
        @app.get("/public-or-private")
        async def mixed_route(user: Optional[ClerkUser] = Depends(get_optional_user)):
            if user:
                return {"message": f"Hello {user.user_id}"}
            return {"message": "Hello anonymous"}
    """
    config = get_clerk_config()
    
    if not config.enabled or not credentials:
        return None
    
    try:
        return verify_clerk_token(credentials.credentials)
    except HTTPException:
        return None


# Rate limiting helper (basic implementation)
class RateLimiter:
    """Simple in-memory rate limiter per user."""
    
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self._requests: dict[str, list[float]] = {}
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if user is within rate limit."""
        import time
        now = time.time()
        minute_ago = now - 60
        
        # Get user's request timestamps
        if user_id not in self._requests:
            self._requests[user_id] = []
        
        # Filter to last minute
        self._requests[user_id] = [
            ts for ts in self._requests[user_id] if ts > minute_ago
        ]
        
        # Check limit
        if len(self._requests[user_id]) >= self.requests_per_minute:
            return False
        
        # Record this request
        self._requests[user_id].append(now)
        return True


# Global rate limiter
_rate_limiter = RateLimiter(requests_per_minute=30)


async def check_rate_limit(user: ClerkUser = Depends(get_current_user)) -> ClerkUser:
    """
    Dependency that checks rate limit after authentication.
    
    Usage:
        @app.post("/analyze")
        async def analyze(user: ClerkUser = Depends(check_rate_limit)):
            ...
    """
    if not _rate_limiter.is_allowed(user.user_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before making more requests."
        )
    return user
