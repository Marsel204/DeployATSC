"""Process-local handoff reservation shared by both dashboard runtimes."""
from contextlib import contextmanager
import threading
import time


class RuntimeConflict(ValueError):
    pass


class PreviewLease:
    """One reachable preview owner, retained through read and confirmed native release.

    Only the request thread releases initially, after its read exits. If release raises,
    one daemon retries with backoff. Public stop can retry failed worker scheduling
    after the request's release exits; it never cancels or races a native call.
    """
    def __init__(self,registry,identity):
        self.registry=registry;self.identity=identity;self.cap=None;self.release_error=None
        self.released=threading.Event();self.wake=threading.Event();self.thread=None
        self.retry_ready=threading.Event();self.retry_lock=threading.Lock();self.scheduling_error=None

    @property
    def resources_released(self):return self.released.is_set()

    def _release(self):
        if self.resources_released:return
        if self.cap is not None:self.cap.release()
        self.release_error=None
        # Removing the registered owner is the admission linearization point; release
        # has returned successfully. Signal waiters after that accounting update.
        with self.registry.lock:self.registry.previews.remove(self)
        self.released.set()

    def finish(self):
        try:self._release()
        except Exception as error:
            self.release_error=str(error)
            self.retry_ready.set()
            self._ensure_retry()
            raise

    def _ensure_retry(self):
        # A failed worker launch must retain accounting and remain publicly retryable.
        # retry_ready is set only after the request's native release has exited.
        with self.retry_lock:
            if self.resources_released or not self.retry_ready.is_set():return
            if self.thread is not None and self.thread.is_alive():return
            try:
                self.thread=threading.Thread(target=self._retry,daemon=True,name="Preview-release")
                self.thread.start()
                self.scheduling_error=None
            except Exception as error:
                self.thread=None;self.scheduling_error=str(error)

    def _retry(self):
        delay=.02
        while not self.resources_released:
            self.wake.wait(delay);self.wake.clear()
            try:self._release()
            except Exception as error:self.release_error=str(error);delay=min(.5,delay*2)

    def stop(self,timeout=1.):
        self._ensure_retry()
        self.wake.set()
        return self.released.wait(max(0,timeout))


class RuntimeOwnership:
    def __init__(self):
        self.lock=threading.Lock();self.reserved=False;self.owner=None;self.kind=None;self.previews=set()

    @property
    def leases(self):
        with self.lock:return len(self.previews)

    @contextmanager
    def reserve(self,kind):
        with self.lock:
            if self.reserved:raise RuntimeConflict('A runtime handoff is already in progress')
            if self.previews:raise RuntimeConflict('A preview capture is open; retry when it is released')
            if kind=='legacy' and self.kind=='observation' and self.owner is not None and not self.owner.resources_released:
                raise RuntimeConflict('Stop the observation and release its resources before starting legacy replay')
            self.reserved=True
        try:yield
        finally:
            with self.lock:self.reserved=False

    @contextmanager
    def lease(self,identity=None):
        """Held by a preview capture from open until release; shared among previews only."""
        with self.lock:
            if self.reserved:raise RuntimeConflict('A runtime start or handoff is in progress')
            if identity is not None and any(owner.identity==identity for owner in self.previews):
                raise RuntimeConflict('A preview still owns this source; retry after release')
            owner=PreviewLease(self,identity);self.previews.add(owner)
        try:yield owner
        except BaseException:
            # Preserve the read/open error; a concurrent cleanup error stays on its owner.
            try:owner.finish()
            except Exception:pass
            raise
        else:owner.finish()

    def preview_owners(self):
        with self.lock:return tuple(self.previews)

    def stop_previews(self,timeout=1.):
        deadline=time.monotonic()+max(0,timeout)
        owners=self.preview_owners()
        for owner in owners:owner.wake.set()
        for owner in owners:owner.stop(max(0,deadline-time.monotonic()))
        return not self.preview_owners()

    def current_owner(self):
        with self.lock:return self.kind,self.owner

    def claim(self,kind,owner):
        with self.lock:
            # A second live owner never displaces an unreleased one; admission rejects it first.
            current=self.owner
            if current is not None and current is not owner and not current.resources_released and owner is not None and not owner.resources_released:
                raise RuntimeConflict("Another runtime owner still owns resources")
            self.kind=kind;self.owner=owner


runtime_ownership=RuntimeOwnership()
