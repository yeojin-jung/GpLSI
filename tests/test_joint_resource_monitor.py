from types import SimpleNamespace
import threading
import time

import pytest

from gplsi_joint_v2.resource_monitor import PeakMemory,_sample_linux_proc,_sample_psutil


def test_fake_process_tree_counts_descendants_once_and_reports_race():
    class FakeError(Exception):pass
    class Process:
        def __init__(self,pid,rss,children=(),gone=False):
            self.pid=pid;self.rss=rss;self.descendants=children;self.gone=gone
        def memory_info(self):
            if self.gone:raise FakeError('vanished')
            return SimpleNamespace(rss=self.rss)
        def children(self,recursive=False):
            assert recursive;return self.descendants
    child=Process(2,200);grandchild=Process(3,300);gone=Process(4,0,gone=True)
    parent=Process(1,100,[child,grandchild,child,gone])
    module=SimpleNamespace(Process=lambda pid:parent,Error=FakeError)
    result=_sample_psutil(module,1)
    assert result['self_rss_bytes']==100
    assert result['process_tree_rss_bytes']==600
    assert result['descendants']==2 and result['unreadable_descendants']==1


def test_linux_proc_fallback_includes_children_of_nonmain_threads(tmp_path):
    for pid,resident,threads in [(1,3,{1:'2',10:'3'}),(2,5,{2:'4'}),(3,7,{3:''}),(4,11,{4:''})]:
        directory=tmp_path/str(pid);directory.mkdir()
        (directory/'statm').write_text(f'100 {resident} 0 0 0\n')
        for tid,children in threads.items():
            thread=directory/'task'/str(tid);thread.mkdir(parents=True)
            (thread/'children').write_text(children)
    result=_sample_linux_proc(1,proc_root=tmp_path,page_size=4096)
    assert result['self_rss_bytes']==3*4096
    assert result['process_tree_rss_bytes']==(3+5+7+11)*4096
    assert result['descendants']==3 and result['backend']=='linux_proc'


def test_peak_sampler_maxima_stop_idempotence_and_daemon_join():
    sampled=threading.Event();lock=threading.Lock();counter=0
    def sample():
        nonlocal counter
        with lock:counter+=1;n=counter
        if n>=3:sampled.set()
        return {'self_rss_bytes':n*10,'process_tree_rss_bytes':n*30,'descendants':2,'backend':'fake'}
    monitor=PeakMemory(.005,sampler=sample,self_highwater_reader=lambda:100)
    assert monitor.start() is monitor and monitor.start() is monitor
    assert sampled.wait(timeout=.5)
    report=monitor.stop();assert monitor.stop()==report
    assert report['peak_self_rss_bytes']>=100
    assert report['peak_process_tree_rss_bytes']>=90
    assert report['samples']>=3 and report['status']=='ok'
    assert report['sampling_thread_still_alive'] is False
    assert monitor._thread.daemon
    assert not any(isinstance(value,list) for value in report.values())


def test_resource_sampling_errors_do_not_crash_fit():
    def fail():raise RuntimeError('instrumentation unavailable')
    monitor=PeakMemory(.01,sampler=fail,self_highwater_reader=lambda:123).start()
    report=monitor.stop()
    assert report['status']=='unavailable'
    assert report['peak_self_rss_bytes']==123
    assert report['peak_process_tree_rss_bytes'] is None
    assert report['sampling_errors']>=1


def test_stop_timeout_is_bounded_and_explicit():
    started=threading.Event();release=threading.Event()
    def delayed():
        started.set();release.wait(timeout=2)
        return {'self_rss_bytes':1,'process_tree_rss_bytes':1}
    monitor=PeakMemory(sampler=delayed,self_highwater_reader=lambda:1,join_timeout_seconds=.01).start()
    try:
        assert started.wait(timeout=.5)
        begin=time.monotonic();report=monitor.stop();elapsed=time.monotonic()-begin
        assert elapsed<.5 and report['status']=='stop_timeout'
        assert report['sampling_thread_still_alive'] is True
        assert monitor.stop()==report
    finally:
        release.set();monitor._thread.join(timeout=.5)


def test_real_psutil_reports_positive_rss_when_available():
    psutil=pytest.importorskip('psutil')
    assert psutil.Process().memory_info().rss>0
    monitor=PeakMemory(.01).start()
    report=monitor.stop()
    assert report['peak_self_rss_bytes']>0
    assert report['backend']=='psutil'
    if report['process_tree_measurement_available']:
        assert report['peak_process_tree_rss_bytes']>0
        assert report['status'] in ['ok','partial']
    else:
        # Restricted macOS sandboxes can permit self RSS but deny the global
        # sysctl process inventory used by psutil.children(). Never invent zero.
        assert report['status']=='unavailable'
        assert report['peak_process_tree_rss_bytes'] is None
        assert report['last_error']
