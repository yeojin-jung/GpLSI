"""Bounded-memory process-tree RSS sampling for full-cohort resource pilots.

The sampled process-tree sum includes shared resident pages once per process,
so it conservatively double-counts those pages. Sampling can miss short spikes;
resource.ru_maxrss additionally preserves the main process's lifetime high water.
This is not Slurm's accounting value, cgroup usage, GPU memory, or unique PSS.
"""
from __future__ import annotations

import os
import math
from pathlib import Path
import sys
import threading
import time


def _resource_self_highwater():
    try:
        import resource
        native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(native*(1 if sys.platform=='darwin' else 1024))
    except (ImportError,OSError,ValueError):
        return None


def _sample_psutil(module,pid):
    process=module.Process(pid)
    self_rss=int(process.memory_info().rss);total=self_rss;children_seen=0;missing=0
    # children(recursive=True) is a bounded process inventory, not a time series.
    children=process.children(recursive=True)
    visited={int(pid)}
    for child in children:
        if int(child.pid) in visited:continue
        visited.add(int(child.pid))
        try:
            total+=int(child.memory_info().rss);children_seen+=1
        except (OSError,module.Error):
            # Short-lived children can disappear between inventory and RSS read.
            missing+=1
    return {'self_rss_bytes':self_rss,'process_tree_rss_bytes':total,
            'descendants':children_seen,'unreadable_descendants':missing,'backend':'psutil'}


def _sample_linux_proc(pid,*,proc_root=Path('/proc'),page_size=None):
    proc_root=Path(proc_root)
    pages=int(os.sysconf('SC_PAGE_SIZE')) if page_size is None else int(page_size)
    def rss(process_id):
        fields=(proc_root/str(process_id)/'statm').read_text().split()
        if len(fields)<2:raise ValueError('Malformed /proc statm')
        return int(fields[1])*pages
    self_rss=rss(pid);total=self_rss;pending=[int(pid)];visited={int(pid)}
    descendants=0;missing=0
    while pending:
        parent=pending.pop()
        task_dir=proc_root/str(parent)/'task'
        try:
            # Include children spawned by any thread, not only the main TID.
            thread_dirs=list(task_dir.iterdir())
        except FileNotFoundError:
            if parent==pid:raise
            missing+=1;continue
        except PermissionError:
            missing+=1;continue
        children=set()
        for thread_dir in thread_dirs:
            try:children.update(int(value) for value in (thread_dir/'children').read_text().split())
            except FileNotFoundError:missing+=1
            except (PermissionError,ValueError):missing+=1
        for child in children:
            if child in visited:continue
            visited.add(child)
            try:child_rss=rss(child)
            except (FileNotFoundError,PermissionError,ValueError):missing+=1;continue
            total+=child_rss;descendants+=1;pending.append(child)
    return {'self_rss_bytes':self_rss,'process_tree_rss_bytes':total,
            'descendants':descendants,'unreadable_descendants':missing,'backend':'linux_proc'}


def _choose_sampler(pid):
    try:
        import psutil
    except ImportError:
        psutil=None
    if psutil is not None:
        return lambda:_sample_psutil(psutil,pid),'psutil'
    if sys.platform.startswith('linux') and Path('/proc/self/statm').exists():
        return lambda:_sample_linux_proc(pid),'linux_proc'
    return None,'unavailable'


class PeakMemory:
    """Start a one-second memory sampler; stop returns an idempotent report.

    Optional injected sampler/highwater callables support deterministic tests.
    A sampler returns current self/tree RSS and its backend/coverage metadata.
    No complete sampling history or process-object inventory is retained.
    """
    def __init__(self,interval_seconds=1.0,*,sampler=None,self_highwater_reader=None,
                 join_timeout_seconds=1.0,pid=None):
        if not math.isfinite(interval_seconds) or interval_seconds<=0:raise ValueError('Sampling interval must be finite and positive')
        if not 0<=join_timeout_seconds<=1:raise ValueError('Join timeout must lie between zero and one second')
        self.interval_seconds=float(interval_seconds);self.pid=os.getpid() if pid is None else int(pid)
        if sampler is None:self._sampler,self._backend=_choose_sampler(self.pid)
        else:self._sampler,self._backend=sampler,'injected'
        self._highwater=_resource_self_highwater if self_highwater_reader is None else self_highwater_reader
        self._join_timeout=float(join_timeout_seconds)
        self._stop_event=threading.Event();self._lock=threading.Lock();self._thread=None
        self._started=False;self._started_at=None;self._report=None
        self._max_self=0;self._max_tree=0;self._max_children=0
        self._samples=0;self._errors=0;self._unreadable_children=0;self._last_error=None

    def _sample(self):
        if self._sampler is None:return
        try:
            result=self._sampler()
            own=int(result['self_rss_bytes']);tree=int(result['process_tree_rss_bytes'])
            if own<0 or tree<own:raise ValueError('Invalid nonnegative self/tree RSS sample')
            with self._lock:
                self._max_self=max(self._max_self,own);self._max_tree=max(self._max_tree,tree)
                self._max_children=max(self._max_children,int(result.get('descendants',0)))
                self._unreadable_children+=int(result.get('unreadable_descendants',0))
                self._samples+=1;self._backend=result.get('backend',self._backend)
        except Exception as exc:
            # Resource instrumentation never turns an otherwise valid fit into
            # a failure; missing measurement is explicit in the returned report.
            with self._lock:
                self._errors+=1;self._last_error=f'{type(exc).__name__}: {exc}'[:500]

    def _run(self):
        # Always attempt the initial sample, even for a very short stage that
        # invokes stop immediately after start.
        while True:
            self._sample()
            if self._stop_event.wait(self.interval_seconds):return

    def start(self):
        if self._started or self._report is not None:return self
        self._started=True;self._started_at=time.monotonic()
        if self._sampler is not None:
            self._thread=threading.Thread(target=self._run,name='joint-v2-peak-memory',daemon=True)
            try:self._thread.start()
            except Exception as exc:
                self._errors+=1;self._last_error=f'{type(exc).__name__}: {exc}'[:500];self._thread=None
        return self

    def stop(self):
        if self._report is not None:return dict(self._report)
        self._stop_event.set()
        if self._thread is not None:self._thread.join(timeout=self._join_timeout)
        thread_alive=bool(self._thread is not None and self._thread.is_alive())
        try:highwater=self._highwater()
        except Exception:highwater=None
        with self._lock:
            if self._report is not None:return dict(self._report)
            own=max(self._max_self,int(highwater or 0))
            status=('stop_timeout' if thread_alive else 'unavailable' if not self._samples else
                    'partial' if self._errors or self._unreadable_children else 'ok')
            self._report={
                'peak_self_rss_bytes':own or None,
                'peak_process_tree_rss_bytes':self._max_tree if self._samples else None,
                'peak_sampled_self_rss_bytes':self._max_self if self._samples else None,
                'self_lifetime_highwater_bytes':int(highwater) if highwater is not None else None,
                'process_tree_measurement_available':bool(self._samples),
                'scope':'main process plus recursive descendants; sampled sum of per-process resident bytes',
                'shared_page_accounting':'sum conservatively double-counts pages shared by processes; not unique PSS',
                'sampling_limitation':'short-lived peaks between samples may be missed; main-process lifetime highwater is also recorded',
                'self_highwater_scope':'resource.RUSAGE_SELF since process start; may precede sampler start',
                'interval_seconds':self.interval_seconds,'backend':self._backend,'status':status,
                'samples':self._samples,'sampling_errors':self._errors,
                'unreadable_descendant_samples':self._unreadable_children,
                'maximum_observed_descendants':self._max_children,'last_error':self._last_error,
                'sampling_thread_still_alive':thread_alive,
                'join_timeout_seconds':self._join_timeout,
                'elapsed_seconds':time.monotonic()-self._started_at if self._started_at is not None else 0.0,
                'sampler_started':self._started}
        return dict(self._report)

    def __enter__(self):return self.start()
    def __exit__(self,exc_type,exc_value,traceback):
        self.stop()
        return False
