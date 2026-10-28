"""Drop exited native-owner locals while retaining exception provenance and locations."""
import traceback


def clear_exception_frames(error):
    """Visit causes, contexts and group children once, including cyclic exception graphs.

    clear_frames leaves traceback nodes/locations intact and skips executing frames.
    Resource owners attached to exceptions are deliberately retained for cleanup/retry.
    """
    pending=[error];seen=set()
    while pending:
        current=pending.pop()
        if id(current) in seen:continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend(child for child in (
            current.__cause__,current.__context__,*getattr(current,"exceptions",())
        ) if isinstance(child,BaseException))
