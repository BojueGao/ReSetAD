"""Exact FP32 ranking through sparse radix counts; bounded-memory PRO, no subsampling."""
from pathlib import Path
import numpy as np
from scipy.ndimage import label


def ordered_keys(values):
    values = np.asarray(values, dtype=np.float32).copy()
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite ranking scores')
    values[values == 0] = 0  # IEEE -0 and +0 have the same rank.
    bits = values.view(np.uint32)
    return np.where(bits & np.uint32(0x80000000), ~bits, bits ^ np.uint32(0x80000000)).astype(np.uint32)


class ExactPixelRanks:
    """One count for each distinct representable FP32 value, never score quantization.

    Two sparse 32-GiB files have a bounded resident working set. Only touched score
    pages consume physical disk. Both files are reproducible registered metric scratch.
    """
    def __init__(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        self.counts = [np.memmap(root / f'{name}.u64', dtype=np.uint64, mode='w+', shape=(2**32,))
                       for name in ('negative', 'positive')]
        self.prefixes = set()
        self.totals = np.zeros(2, dtype=np.uint64)

    def add(self, labels, scores):
        y = np.asarray(labels, dtype=bool).ravel()
        values = np.asarray(scores, dtype=np.float32).ravel()
        if y.shape != values.shape:
            raise ValueError('Ranking geometry mismatch')
        # Sort per-image, aggregate equal FP32 keys, then add integer counts.
        keys, inverse, counts = np.unique(ordered_keys(values), return_inverse=True, return_counts=True)
        pos = np.bincount(inverse, weights=y.astype(np.uint8), minlength=len(keys)).astype(np.uint64)
        neg = counts.astype(np.uint64) - pos
        for table, increments in zip(self.counts, (neg, pos)):
            old = table[keys]
            if np.any(old > np.iinfo(np.uint64).max - increments):
                raise OverflowError('Exact count overflow')
            table[keys] = old + increments
        self.totals += np.array([int(neg.sum()), int(pos.sum())], dtype=np.uint64)
        self.prefixes.update(int(v) for v in np.unique(keys >> 16))

    def result(self, poll=lambda: None):
        negative, positive = map(int, self.totals)
        if not negative or not positive:
            return dict(auroc=None, ap=None, status='NA_ALL_POSITIVE' if not negative else 'INCOMPLETE',
                        reason='Ranking needs both label states', valid_pixels=negative+positive)
        tp = fp = 0
        auc = ap = 0.0
        for prefix in sorted(self.prefixes, reverse=True):
            poll()
            start = prefix << 16
            n = np.asarray(self.counts[0][start:start+65536])[::-1]
            p = np.asarray(self.counts[1][start:start+65536])[::-1]
            keep = (n != 0) | (p != 0)
            n, p = n[keep], p[keep]
            if not len(n):
                continue
            new_tp = tp + np.cumsum(p, dtype=np.uint64)
            new_fp = fp + np.cumsum(n, dtype=np.uint64)
            previous_tp = np.r_[float(tp), new_tp[:-1].astype(np.float64)]
            auc += float(np.sum(n.astype(np.float64) * (new_tp.astype(np.float64)+previous_tp) / 2))
            ap += float(np.sum(p.astype(np.float64) * new_tp.astype(np.float64) / (new_tp+new_fp)))
            tp, fp = int(new_tp[-1]), int(new_fp[-1])
        if (tp, fp) != (positive, negative):
            raise AssertionError('Lost exact ranking counts')
        return dict(auroc=auc/(positive*negative), ap=ap/positive, status='READY',
                    valid_pixels=positive+negative, positive_pixels=positive, negative_pixels=negative,
                    exact_fp32=True, occupied_radix_prefixes=len(self.prefixes))

    def close(self):
        for table in self.counts:
            table.flush()
            table._mmap.close()


class StreamingPRO:
    def __init__(self, low, high, thresholds=4096, limit=.30):
        self.levels = np.linspace(low, high, thresholds, dtype=np.float64)
        self.background = np.zeros(thresholds+1, dtype=np.int64)
        self.region_pro = np.zeros(thresholds, dtype=np.float64)
        self.regions = 0
        self.limit = limit

    def add(self, mask, score):
        mask, score = np.asarray(mask, dtype=bool), np.asarray(score, dtype=np.float32)
        if mask.shape != score.shape or not np.isfinite(score).all():
            raise ValueError('Invalid PRO geometry/numerics')
        # right means threshold <= score, precisely matching the >= decision rule.
        bins = np.searchsorted(self.levels, score, side='right')
        self.background += np.bincount(bins[~mask], minlength=len(self.background))
        components, count = label(mask, structure=np.ones((3,3)))
        for index in range(1, count+1):
            selected = bins[components == index]
            histogram = np.bincount(selected, minlength=len(self.background))
            self.region_pro += np.cumsum(histogram[::-1], dtype=np.int64)[::-1][1:] / len(selected)
        self.regions += count

    def result(self):
        if not self.regions:
            return dict(value=None, status='NA_NO_MASK', reason='No GT connected region')
        total = int(self.background.sum())
        if not total:
            return dict(value=None, status='INCOMPLETE', reason='No background pixels for FPR')
        fpr = np.cumsum(self.background[::-1], dtype=np.int64)[::-1][1:] / total
        pro = self.region_pro / self.regions
        fpr, pro = np.r_[0., fpr], np.r_[0., pro]
        order = np.argsort(fpr, kind='stable')
        fpr, pro = fpr[order], pro[order]
        x, starts = np.unique(fpr, return_index=True)
        y = np.maximum.reduceat(pro, starts)
        endpoint = float(np.interp(self.limit, x, y))
        keep = x < self.limit
        value = np.trapezoid(np.r_[y[keep], endpoint], np.r_[x[keep], self.limit]) / self.limit
        return dict(value=float(value), status='READY', regions=self.regions,
                    thresholds=len(self.levels), background_pixels=total)
