"""Admin-API security: API key + IP allowlist + rate-limit + audit log.

Used as a router-level dependency on /admin/*:

    router = APIRouter(dependencies=[Depends(require_admin)])

Fail-closed: if ADMIN_API_KEY is unset, every admin call returns 503 so the
endpoints can never be reached unauthenticated by misconfiguration.
"""
import hmac
import logging
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

from app.core.config import settings

logger = logging.getLogger("admin.audit")

# In-memory sliding-window rate limit (per client IP). Single-replica service,
# so a process-local counter is sufficient.
_RATE_MAX = 30          # requests
_RATE_WINDOW = 60.0     # seconds
_hits: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    """Real client IP. Behind Railway's proxy, trust the first X-Forwarded-For."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _allowlist() -> set[str]:
    return {ip.strip() for ip in settings.ADMIN_IP_ALLOWLIST.split(",") if ip.strip()}


def _rate_limited(ip: str) -> bool:
    now = time.monotonic()
    q = _hits[ip]
    while q and now - q[0] > _RATE_WINDOW:
        q.popleft()
    if len(q) >= _RATE_MAX:
        return True
    q.append(now)
    return False


async def require_admin(
    request: Request,
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
):
    """FastAPI dependency guarding all /admin/* routes."""
    ip = _client_ip(request)
    path = request.url.path

    # Fail-closed: admin must be explicitly configured.
    if not settings.ADMIN_API_KEY:
        logger.warning("admin DENY (unconfigured) ip=%s path=%s", ip, path)
        raise HTTPException(status_code=503, detail="Admin API is not configured.")

    # Network layer: optional IP allowlist.
    allow = _allowlist()
    if allow and ip not in allow:
        logger.warning("admin DENY (ip) ip=%s path=%s", ip, path)
        raise HTTPException(status_code=403, detail="Forbidden.")

    # Rate limit (after IP check so allowlisted ops aren't starved by attackers).
    if _rate_limited(ip):
        logger.warning("admin DENY (rate) ip=%s path=%s", ip, path)
        raise HTTPException(status_code=429, detail="Too many requests.")

    # Constant-time key comparison.
    provided = x_admin_key or ""
    if not hmac.compare_digest(provided, settings.ADMIN_API_KEY):
        logger.warning("admin DENY (key) ip=%s path=%s", ip, path)
        raise HTTPException(status_code=401, detail="Unauthorized.")

    logger.info("admin OK ip=%s method=%s path=%s", ip, request.method, path)
