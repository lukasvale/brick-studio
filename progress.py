"""Measured elapsed times and explicitly estimated progress for local jobs."""
import threading
import time


class ProgressReporter:
    def __init__(self, names, estimates, emit, interval=.4):
        self.names = names
        self.estimates = list(estimates)
        self.emit = emit
        self.interval = interval
        self.batch_start = time.monotonic()
        self.photo_start = self.batch_start
        self.index = 0
        self.completed = 0
        self.phase_name = 'Preparing'
        self.floor = 0
        self.packaging = False
        self.finished = False
        self.durations = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def begin_photo(self, index):
        self.index = index
        self.photo_start = time.monotonic()
        self.phase_name = 'Developing RAW'
        self.floor = 0
        self.send()

    def phase(self, label, fraction):
        self.phase_name = label
        self.floor = fraction
        self.send()

    def complete_photo(self, adapt=True):
        duration = time.monotonic() - self.photo_start
        self.durations.append(duration)
        previous = self.estimates[self.index]
        self.estimates[self.index] = duration
        # Adapt remaining comparable jobs, without treating cached work as
        # evidence that a fresh AI pass will take the same time.
        ratio = max(.3, min(3, duration / max(previous, .1)))
        for j in range(self.index + 1, len(self.estimates)):
            if adapt and .6 < self.estimates[j] / max(previous, .1) < 1.7:
                self.estimates[j] *= .65 + .35 * ratio
        self.completed += 1
        self.floor = 1
        self.send()
        return duration

    def begin_packaging(self):
        self.packaging = True
        self.phase_name = 'Aligning framing and packaging'
        self.pack_start = time.monotonic()
        self.send()

    def snapshot(self):
        now = time.monotonic()
        total = len(self.names)
        photo_elapsed = (self.durations[-1] if self.packaging and self.durations
                         else max(0, now - self.photo_start))
        expected = self.estimates[self.index] if total else 1
        fraction = 1 if self.floor == 1 else min(.96, max(self.floor, photo_elapsed/max(expected, .1)*.9))
        packing_estimate = max(1, total * .65)
        if self.finished:
            eta, batch_fraction = 0, 1
        elif self.packaging:
            elapsed = now-self.pack_start
            eta = max(1, packing_estimate-elapsed)
            batch_fraction = min(.99, .94 + .05*elapsed/packing_estimate)
        else:
            current_remaining = 0 if self.floor == 1 else max(1, expected-photo_elapsed)
            eta = current_remaining+sum(self.estimates[self.index+1:])+packing_estimate
            work_done = sum(self.estimates[:self.index])+expected*fraction
            batch_fraction = .94*work_done/max(.1,sum(self.estimates))
        return dict(stage='progress', phase=self.phase_name,
                    index=self.index, photo_number=self.index+1, completed=self.completed,
                    total=total, name=self.names[self.index] if total else '',
                    photo_elapsed=round(photo_elapsed,1), batch_elapsed=round(now-self.batch_start,1),
                    photo_progress=round(fraction,4), batch_progress=round(batch_fraction,4),
                    eta_seconds=round(eta,1), progress_is_estimate=not self.finished)

    def send(self):
        self.emit(**self.snapshot())

    def _loop(self):
        while not self._stop.wait(self.interval):
            self.send()

    def finish(self):
        self.finished = True
        self.phase_name = 'Complete'
        self._stop.set()
        self._thread.join(timeout=1)
        self.send()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
