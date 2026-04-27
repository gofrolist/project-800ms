"""Internal-caller bearer auth for the retriever service.

Mirrors ``apps/api``'s ``X-Internal-Token`` pattern (see
``apps/api/routes/transcripts.py::_require_internal_token``):

  * Header: ``X-Internal-Token: <secret>``.
  * Empty config → 503 ``retriever_unconfigured`` (endpoint disabled,
    not silently open).
  * Wrong / missing token → 401 ``unauthenticated``.
  * `secrets.compare_digest` to dodge timing leaks on the comparison.

Closes issue #40 (RETRIEVER_INTERNAL_TOKEN) and partially #47 (the
remaining piece — making ``tenant_id`` come from a signed claim instead
of the request body — is filed as a follow-up). v1's threat model is
"prevent lateral movement on the docker-internal network from any
co-resident container without the secret"; that's what shared-secret
bearer covers.
"""

from __future__ import annotations

import secrets

from fastapi import Header

from config import get_settings
from errors import RetrieverUnconfigured, Unauthenticated


def require_internal_token(
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> None:
    """FastAPI dependency. Apply to internal-only routes.

    Raises typed errors so the existing exception handler emits the
    flat ``{error, message}`` envelope rather than FastAPI's default
    ``{"detail": ...}`` shape.
    """
    token = get_settings().retriever_internal_token
    if not token:
        # Fail closed at request-time, not boot-time: dev / unit tests
        # that don't need /retrieve still work, but the endpoint itself
        # refuses traffic when the secret is unset.
        raise RetrieverUnconfigured()
    if not x_internal_token or not secrets.compare_digest(x_internal_token, token):
        raise Unauthenticated()
