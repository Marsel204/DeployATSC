"""Conservative, startup-only BLAS policy. Keep numerical imports out of this module."""
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class CpuPolicy:
    cpus:tuple[int,...]
    governor:str|None
    min_khz:int|None
    max_khz:int|None
    cur_khz:int|None

    @property
    def pinned(self):
        if not self.governor:return None
        if self.governor=='performance':return True
        if self.min_khz is None or self.max_khz is None:return None
        return self.min_khz==self.max_khz


@dataclass(frozen=True)
class BlasDecision:
    requested:str
    applied:str|None
    reason:str
    policies:tuple[CpuPolicy,...]


_last=BlasDecision('auto',None,'not_applied',())


def _read(path):
    try:return path.read_text().strip()
    except (OSError,ValueError):return None


def _frequency(path):
    try:
        value=int(_read(path))
        return value if value>0 else None
    except (TypeError,ValueError):return None


def read_cpu_policies(sysfs_root='/sys'):
    """One fact per shared cpufreq directory; offline CPUs do not affect policy."""
    groups={}
    try:
        for cpu in sorted((Path(sysfs_root)/'devices/system/cpu').glob('cpu[0-9]*')):
            if not cpu.name[3:].isdigit() or _read(cpu/'online')=='0':continue
            policy=cpu/'cpufreq'
            # Retain an unknown policy for an online CPU with missing cpufreq data.
            groups.setdefault(policy.resolve(),[]).append(int(cpu.name[3:]))
    except OSError:return ()
    return tuple(CpuPolicy(tuple(cpus),_read(path/'scaling_governor'),
        _frequency(path/'scaling_min_freq'),_frequency(path/'scaling_max_freq'),
        _frequency(path/'scaling_cur_freq')) for path,cpus in groups.items())


def clocks_pinned(policies):
    if any(p.pinned is False for p in policies):return False
    return True if policies and all(p.pinned is True for p in policies) else None


def apply_blas_policy(environ=None,sysfs_root=None):
    global _last
    # Nested launchers must retain the first process policy decision.
    if environ is None and _last.reason!='not_applied':return _last
    env=os.environ if environ is None else environ
    requested=env.get('ATSC_BLAS_THREADS','auto');policies=read_cpu_policies('/sys' if sysfs_root is None else sysfs_root)
    value=None
    if requested=='default':reason='default_requested'
    elif requested=='auto':
        pinned=clocks_pinned(policies)
        if pinned is True:value='1';reason='clocks_pinned'
        else:reason='clocks_unpinned' if pinned is False else 'clocks_unknown'
    elif requested.isdecimal() and int(requested)>0:value=str(int(requested));reason='explicit_threads'
    else:reason='invalid_policy'
    if 'OPENBLAS_NUM_THREADS' in env:value=None;reason='user_openblas_preserved'
    if value is not None:env.setdefault('OPENBLAS_NUM_THREADS',value)
    _last=BlasDecision(requested,value,reason,policies)
    return _last


def last_decision():return _last


def host_provenance(sysfs_root='/sys'):
    policies=read_cpu_policies(sysfs_root);decision=last_decision()
    return {'cpu_policies':[{'cpus':list(p.cpus),'governor':p.governor,'min_khz':p.min_khz,
        'max_khz':p.max_khz,'cur_khz':p.cur_khz} for p in policies],
        'clocks_pinned':clocks_pinned(policies),'runtime_tuning':{'policy':decision.requested,
        'openblas_num_threads':decision.applied,'reason':decision.reason},
        'openblas_num_threads':os.environ.get('OPENBLAS_NUM_THREADS')}


def performance_advice(host):
    if host.get('clocks_pinned') is False:
        policies=host['cpu_policies'];governors=', '.join(sorted({p['governor'] or 'unknown' for p in policies}))
        minima=[p['min_khz'] for p in policies if p['min_khz'] is not None]
        minimum=f'{min(minima)/1000:g}' if minima else 'unknown'
        return [f'CPU clocks are not pinned (governor {governors}, min {minimum} MHz): real-time playback may stutter. Run `sudo jetson_clocks` before launching; `sudo jetson_clocks --restore` undoes it.']
    if host.get('clocks_pinned') is True and host.get('openblas_num_threads')!='1':
        return ['OpenBLAS threads are not capped: about 3.5 CPU cores are wasted spinning. Launch with the default ATSC_BLAS_THREADS=auto or set OPENBLAS_NUM_THREADS=1.']
    return []
