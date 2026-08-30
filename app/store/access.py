"""Who may see which document.

The visibility rule lives here and nowhere else. Spreading it across the
retrieval legs, the document listing and the file-serving endpoint is how one
of them ends up subtly different from the others, and the one that is different
is the one that leaks.

Deliberately separate from the `where=` metadata filter the stores already
take. `where=` is a caller's preference -- narrow this search to one source --
and callers may set, widen or drop it freely. An AccessScope is not a
preference: it is the boundary the caller is not allowed to widen. Keeping them
as one parameter means a caller who passes `where={"source": "x.md"}` silently
replaces the security filter with their own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# A document is visible to a user when it belongs to their company AND either
# they own it, or it has been shared with a department they are in and a
# manager has approved it.
SCOPE_PRIVATE = "private"
SCOPE_DEPARTMENT = "department"

STATUS_ACTIVE = "active"
STATUS_PENDING = "pending_approval"
STATUS_REJECTED = "rejected"


@dataclass(frozen=True)
class AccessScope:
    """The identity a query runs as.

    Frozen because a scope handed to the retrieval layer must be the same one
    the request was authenticated with -- a mutable scope is one an inner layer
    can widen.
    """
    company_id: str
    user_id: str
    department_ids: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        # An empty company_id or user_id would render the SQL predicate
        # trivially wide rather than trivially narrow, so refuse it here
        # instead of letting it become a silent "sees everything".
        if not self.company_id or not self.user_id:
            raise ValueError("AccessScope needs both company_id and user_id")
