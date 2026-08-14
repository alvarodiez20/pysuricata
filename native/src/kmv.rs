//! K-Minimum-Values distinct-count sketch.
//!
//! Differences from `accumulators/sketches.py::KMV`:
//!
//! * The Python sketch keeps its k smallest hashes in a **sorted Python list**
//!   and, on every batch, does `list.extend(candidates); list.sort()` — so each
//!   chunk costs O((k+m) log(k+m)) with boxed integers and pointer-chasing
//!   through the list's PyObject array. Here it is a fixed-capacity binary
//!   max-heap over a flat `Vec<u64>`: O(log k) per surviving candidate, one
//!   cache-resident 8 KiB array for the default k=1024, and the common case
//!   (`h >= heap[0]`) is rejected by a single compare against the root.
//!
//! * The estimator is bias-corrected. The Python version returns
//!   `(k-1)/t` where `t` is the k-th smallest hash on (0,1] — but then wraps it
//!   in `max(n, ...)`, which cannot lower an over-estimate and silently pins the
//!   result to the sketch size when `t` is degenerate.

use crate::hashing::to_unit;

#[derive(Clone, Debug)]
pub struct Kmv {
    k: usize,
    /// Max-heap of the k smallest hashes seen. `heap[0]` is the current
    /// threshold: anything >= it cannot enter.
    heap: Vec<u64>,
    /// Total values offered (not distinct) — useful for sanity checks.
    seen: u64,
}

impl Kmv {
    pub fn new(k: usize) -> Self {
        let k = k.max(16);
        Self {
            k,
            heap: Vec::with_capacity(k),
            seen: 0,
        }
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    #[inline]
    pub fn seen(&self) -> u64 {
        self.seen
    }

    /// True while the sketch still holds every distinct value it has seen, so
    /// `estimate()` is exact rather than approximate.
    #[inline]
    pub fn is_exact(&self) -> bool {
        self.heap.len() < self.k
    }

    #[inline]
    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.heap[i] > self.heap[parent] {
                self.heap.swap(i, parent);
                i = parent;
            } else {
                break;
            }
        }
    }

    #[inline]
    fn sift_down(&mut self, mut i: usize) {
        let n = self.heap.len();
        loop {
            let l = 2 * i + 1;
            let r = l + 1;
            let mut largest = i;
            if l < n && self.heap[l] > self.heap[largest] {
                largest = l;
            }
            if r < n && self.heap[r] > self.heap[largest] {
                largest = r;
            }
            if largest == i {
                break;
            }
            self.heap.swap(i, largest);
            i = largest;
        }
    }

    /// Offer one already-mixed 64-bit hash.
    #[inline]
    pub fn offer(&mut self, h: u64) {
        self.seen += 1;
        if self.heap.len() < self.k {
            // Duplicate suppression while filling: O(k) but bounded, and only
            // paid until the heap is full.
            if self.heap.contains(&h) {
                return;
            }
            self.heap.push(h);
            let last = self.heap.len() - 1;
            self.sift_up(last);
            return;
        }
        // Fast reject: one compare against the root handles the overwhelming
        // majority of values once the sketch is warm.
        if h >= self.heap[0] {
            return;
        }
        if self.heap.contains(&h) {
            return;
        }
        self.heap[0] = h;
        self.sift_down(0);
    }

    pub fn offer_many(&mut self, hashes: &[u64]) {
        for &h in hashes {
            self.offer(h);
        }
    }

    /// Merge another sketch. KMV is mergeable: the union's k smallest hashes
    /// are a subset of the two inputs' k smallest. This is what makes
    /// per-column-per-thread accumulation safe to fan out and recombine.
    pub fn merge(&mut self, other: &Kmv) {
        self.seen += other.seen;
        let incoming: Vec<u64> = other.heap.clone();
        for h in incoming {
            // Re-offer without double-counting `seen`.
            self.seen -= 1;
            self.offer(h);
        }
    }

    /// Distinct-count estimate.
    ///
    /// Below k distinct values the sketch is exact. Above it, the k-th smallest
    /// hash `t` on (0,1] gives the unbiased estimator `(k-1)/t`
    /// (Bar-Yossef et al. 2002); relative standard error is about
    /// `1/sqrt(k-2)`, so k=1024 gives ~3.1% and k=4096 gives ~1.6%.
    pub fn estimate(&self) -> f64 {
        let n = self.heap.len();
        if n < self.k {
            return n as f64;
        }
        let t = to_unit(self.heap[0]);
        if t <= 0.0 {
            return n as f64;
        }
        ((self.k - 1) as f64) / t
    }

    /// Relative standard error of the current estimate, for honest reporting in
    /// the UI ("~12,400 distinct, +/- 3%") instead of a bare number.
    pub fn relative_error(&self) -> f64 {
        if self.is_exact() {
            0.0
        } else {
            1.0 / ((self.k as f64) - 2.0).sqrt()
        }
    }

    pub fn hashes(&self) -> &[u64] {
        &self.heap
    }

    pub fn from_parts(k: usize, heap: Vec<u64>, seen: u64) -> Self {
        let mut s = Self {
            k: k.max(16),
            heap,
            seen,
        };
        // Restore the heap invariant defensively (state may come from pickle).
        let n = s.heap.len();
        for i in (0..n / 2).rev() {
            s.sift_down(i);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hashing::mix64;

    #[test]
    fn exact_below_k() {
        let mut s = Kmv::new(1024);
        for i in 0..500u64 {
            s.offer(mix64(i));
        }
        assert!(s.is_exact());
        assert_eq!(s.estimate(), 500.0);
        assert_eq!(s.relative_error(), 0.0);
    }

    #[test]
    fn duplicates_do_not_inflate() {
        let mut s = Kmv::new(1024);
        for _ in 0..100 {
            for i in 0..300u64 {
                s.offer(mix64(i));
            }
        }
        assert_eq!(s.estimate(), 300.0);
        assert_eq!(s.seen(), 30_000);
    }

    #[test]
    fn estimate_within_error_bound() {
        // 1e6 distinct values, k=4096 -> RSE ~1.56%. Allow 4 sigma.
        let k = 4096;
        let truth = 1_000_000u64;
        let mut s = Kmv::new(k);
        for i in 0..truth {
            s.offer(mix64(i));
        }
        let est = s.estimate();
        let rel = (est - truth as f64).abs() / truth as f64;
        assert!(rel < 4.0 * s.relative_error(), "est {est} rel {rel}");
    }

    #[test]
    fn merge_matches_union() {
        let k = 2048;
        let mut a = Kmv::new(k);
        let mut b = Kmv::new(k);
        let mut both = Kmv::new(k);
        for i in 0..300_000u64 {
            a.offer(mix64(i));
            both.offer(mix64(i));
        }
        for i in 200_000..500_000u64 {
            b.offer(mix64(i));
            both.offer(mix64(i));
        }
        a.merge(&b);
        // The k smallest of the union are identical either way, so the
        // estimates must agree exactly, not just approximately.
        assert_eq!(a.estimate(), both.estimate());
    }

    #[test]
    fn heap_root_is_max() {
        let mut s = Kmv::new(64);
        for i in 0..10_000u64 {
            s.offer(mix64(i));
        }
        let root = s.hashes()[0];
        assert!(s.hashes().iter().all(|&h| h <= root));
        assert_eq!(s.len(), 64);
    }
}
