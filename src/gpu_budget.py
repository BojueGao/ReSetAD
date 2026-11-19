"""Conservative allocated single-GPU wall time, including validation and CPU waits."""
from datetime import datetime
import json
import os
from pathlib import Path
import time
from .evidence import write_json, utc_now


class BudgetStop(Exception):
    pass


class GPUBudget:
    def __init__(self, path, hours):
        self.path = Path(path)
        if hours != 25:
            raise ValueError('This authorized development budget is exactly 25 GPU hours')
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {
            'maximum_gpu_hours':hours, 'accounting':'one allocated GPU times elapsed wall time; includes CPU evaluation waits', 'attempts':[]}
        if self.data['maximum_gpu_hours'] != hours:
            raise ValueError('Cannot change an existing budget')
        self.previous = 0.
        for attempt in self.data['attempts']:
            if attempt.get('ended_at') is None:
                stat = Path(f"/proc/{attempt['pid']}/stat")
                if stat.exists() and stat.read_text().split()[21] == attempt['start_ticks']:
                    raise RuntimeError('Previous budget owner still active')
                # Conservative after unclean loss: include all elapsed time through detection.
                attempt['ended_at'] = utc_now()
                attempt['seconds'] = max(attempt.get('seconds',0.),
                    (datetime.fromisoformat(attempt['ended_at'])-datetime.fromisoformat(attempt['started_at'])).total_seconds())
                attempt['unclean_end_conservative_accounting'] = True
            self.previous += attempt['seconds']
        self.started = time.monotonic()
        self.attempt = dict(pid=os.getpid(), start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
                            started_at=utc_now(), ended_at=None, seconds=0.)
        self.data['attempts'].append(self.attempt)
        self.last_write = 0.
        self.stopped = False
        self.poll()

    def consumed(self):
        return self.previous + (self.attempt['seconds'] if self.stopped else time.monotonic()-self.started)

    def poll(self):
        now = time.monotonic()
        self.attempt['seconds'] = now-self.started
        if now-self.last_write >= 10:
            self.data['consumed_gpu_hours'] = self.consumed()/3600
            write_json(self.path, self.data)
            self.last_write = now
        if self.consumed() >= self.data['maximum_gpu_hours']*3600-90:
            raise BudgetStop('25 GPU-hour budget reached; reserved 90 seconds for durable shutdown')

    def finish(self):
        self.attempt['seconds'] = time.monotonic()-self.started
        self.attempt['ended_at'] = utc_now()
        self.stopped = True
        self.data['consumed_gpu_hours'] = self.consumed()/3600
        write_json(self.path, self.data)
