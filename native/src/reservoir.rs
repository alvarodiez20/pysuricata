//! Reservoir sampling, Algorithm L (Li, 1994), with a per-instance RNG.
//!
//! Two problems with `accumulators/sketches.py::ReservoirSampler.add_many`:
//!
//! 1. **It is not uniform.** It draws
//!    `np.random.randint(1, seen + len(arr) + 1, size=len(arr))` — one uniform
//!    over the *post-batch* count for every element of the batch. True
//!    reservoir sampling requires each element i to be accepted with
//!    probability k/(seen+i), a denominator that grows *within* the batch.
//!    Using the final denominator for all of them under-weights early elements
//!    and over-weights late ones, and the bias grows with chunk size. Every
//!    quantile, the median, the IQR, the MAD, the outlier counts and the
//!    histogram in the report are computed from this reservoir.
//!
//! 2. **It uses the global RNG**, which `report.py` seeds process-wide with
//!    `np.random.seed(...)`. That makes every `profile()` call reset the
//!    caller's global RNG state, and makes per-column threading unreproducible.
//!
//! Algorithm L fixes the correctness problem and is also much cheaper: instead
//! of drawing one random number per element, it draws a geometric skip and
//! jumps over the elements it will not select. Expected draws for n elements
//! into a reservoir of k is O(k * (1 + ln(n/k))) — for n=10M, k=20k that is
//! ~145k draws instead of 10M.

/// xoshiro256++ — small, fast, and reproducible from an explicit seed.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn seed_from_u64(seed: u64) -> Self {
        // splitmix64 expansion, the author-recommended way to fill the state.
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
        }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform on [0, 1). 53 bits of mantissa, the standard construction.
    #[inline(always)]
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Uniform on [0, 1), excluding exactly 0 so `ln()` is finite.
    #[inline(always)]
    fn next_f64_open(&mut self) -> f64 {
        loop {
            let u = self.next_f64();
            if u > 0.0 {
                return u;
            }
        }
    }

    #[inline(always)]
    pub fn next_below(&mut self, bound: u64) -> u64 {
        // Lemire's debiased bounded generation.
        if bound == 0 {
            return 0;
        }
        let mut x = self.next_u64();
        let mut m = (x as u128).wrapping_mul(bound as u128);
        let mut l = m as u64;
        if l < bound {
            let t = bound.wrapping_neg() % bound;
            while l < t {
                x = self.next_u64();
                m = (x as u128).wrapping_mul(bound as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64
    }
}

#[derive(Clone, Debug)]
pub struct ReservoirL {
    k: usize,
    buf: Vec<f64>,
    seen: u64,
    /// Algorithm L's running acceptance scale.
    w: f64,
    /// Index of the next element that will be selected.
    next_select: u64,
    rng: Rng,
}

impl ReservoirL {
    pub fn new(k: usize, seed: u64) -> Self {
        let k = k.max(1);
        let mut rng = Rng::seed_from_u64(seed);
        let w = (rng.next_f64_open().ln() / k as f64).exp();
        let skip = (rng.next_f64_open().ln() / (1.0 - w).max(f64::MIN_POSITIVE).ln()).floor();
        Self {
            k,
            buf: Vec::with_capacity(k),
            seen: 0,
            w,
            next_select: k as u64 + skip as u64 + 1,
            rng,
        }
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    #[inline]
    pub fn seen(&self) -> u64 {
        self.seen
    }

    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.buf
    }

    /// Ratio of the population to the sample, so callers can scale sample-based
    /// counts (outliers, zero-runs) back to population estimates honestly.
    pub fn scale(&self) -> f64 {
        if self.buf.is_empty() {
            1.0
        } else {
            (self.seen as f64) / (self.buf.len() as f64)
        }
    }

    fn advance_skip(&mut self) {
        self.w *= (self.rng.next_f64_open().ln() / self.k as f64).exp();
        let denom = (1.0 - self.w).max(f64::MIN_POSITIVE).ln();
        let skip = (self.rng.next_f64_open().ln() / denom).floor();
        self.next_select = self
            .next_select
            .saturating_add(skip as u64)
            .saturating_add(1);
    }

    /// Feed a batch. Finite values only — NaN/inf are skipped, matching what
    /// the quantile consumers expect.
    pub fn add_many(&mut self, values: &[f64]) {
        for &x in values {
            if !x.is_finite() {
                continue;
            }
            self.seen += 1;
            if self.buf.len() < self.k {
                self.buf.push(x);
                continue;
            }
            if self.seen == self.next_select {
                let idx = self.rng.next_below(self.k as u64) as usize;
                self.buf[idx] = x;
                self.advance_skip();
            }
        }
    }

    /// Merge two reservoirs into an unbiased sample of the union.
    ///
    /// Each slot of the result is drawn from `self` with probability
    /// n_self/(n_self+n_other), which preserves uniformity over the union.
    pub fn merge(&mut self, other: &ReservoirL) {
        if other.seen == 0 {
            return;
        }
        if self.seen == 0 {
            self.buf = other.buf.clone();
            self.seen = other.seen;
            return;
        }
        let total = self.seen + other.seen;
        let target = self.k.min(total as usize);
        let mut out = Vec::with_capacity(target);
        let (mut i, mut j) = (0usize, 0usize);
        for _ in 0..target {
            let take_self = self.rng.next_below(total) < self.seen;
            if take_self && i < self.buf.len() {
                out.push(self.buf[i]);
                i += 1;
            } else if j < other.buf.len() {
                out.push(other.buf[j]);
                j += 1;
            } else if i < self.buf.len() {
                out.push(self.buf[i]);
                i += 1;
            } else {
                break;
            }
        }
        self.buf = out;
        self.seen = total;
    }

    /// Sorted copy of the sample, for quantile extraction.
    pub fn sorted(&self) -> Vec<f64> {
        let mut v = self.buf.clone();
        v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        v
    }

    /// Linear-interpolation quantile over the sample, matching numpy's default
    /// ("linear") method so results are directly comparable to `np.percentile`.
    pub fn quantile_from_sorted(sorted: &[f64], q: f64) -> f64 {
        if sorted.is_empty() {
            return f64::NAN;
        }
        if sorted.len() == 1 {
            return sorted[0];
        }
        let pos = q.clamp(0.0, 1.0) * ((sorted.len() - 1) as f64);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            return sorted[lo];
        }
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_then_bounds() {
        let mut r = ReservoirL::new(1000, 42);
        let data: Vec<f64> = (0..100_000).map(|i| i as f64).collect();
        r.add_many(&data);
        assert_eq!(r.values().len(), 1000);
        assert_eq!(r.seen(), 100_000);
        assert!((r.scale() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn reproducible_from_seed() {
        let data: Vec<f64> = (0..50_000).map(|i| (i % 997) as f64).collect();
        let mut a = ReservoirL::new(500, 7);
        let mut b = ReservoirL::new(500, 7);
        a.add_many(&data);
        b.add_many(&data);
        assert_eq!(a.values(), b.values());

        let mut c = ReservoirL::new(500, 8);
        c.add_many(&data);
        assert_ne!(a.values(), c.values());
    }

    #[test]
    fn sample_is_uniform_over_position() {
        // The bug in the Python version shows up exactly here: if early
        // elements are under-weighted, the mean of the sampled *positions*
        // drifts above the population mean of n/2.
        let n = 200_000usize;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut total = 0.0;
        let trials = 40;
        for seed in 0..trials {
            let mut r = ReservoirL::new(2_000, seed as u64);
            r.add_many(&data);
            let mean: f64 = r.values().iter().sum::<f64>() / r.values().len() as f64;
            total += mean;
        }
        let avg = total / trials as f64;
        let expected = (n as f64 - 1.0) / 2.0;
        // Standard error of the mean of a uniform sample of 2000 from
        // U(0, 200k) is ~1292; across 40 trials, ~204. Allow 5 sigma.
        assert!(
            (avg - expected).abs() < 5.0 * 204.0,
            "avg {avg} expected {expected}"
        );
    }

    #[test]
    fn chunked_matches_whole_distributionally() {
        let n = 100_000usize;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let mut whole = ReservoirL::new(1_000, 3);
        whole.add_many(&data);

        let mut merged = ReservoirL::new(1_000, 3);
        for chunk in data.chunks(7_000) {
            let mut part = ReservoirL::new(1_000, 3);
            part.add_many(chunk);
            merged.merge(&part);
        }
        assert_eq!(merged.seen(), n as u64);

        let mw = whole.values().iter().sum::<f64>() / whole.values().len() as f64;
        let mm = merged.values().iter().sum::<f64>() / merged.values().len() as f64;
        // Both estimate n/2 with SE ~ n/sqrt(12*1000) ~= 913. Allow 5 sigma
        // on the difference of two independent estimates.
        assert!((mw - mm).abs() < 5.0 * 1291.0, "{mw} vs {mm}");
    }

    #[test]
    fn quantiles_track_truth() {
        let n = 500_000usize;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut r = ReservoirL::new(20_000, 11);
        r.add_many(&data);
        let sorted = r.sorted();
        for (q, truth) in [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)] {
            let est = ReservoirL::quantile_from_sorted(&sorted, q);
            let exp = truth * (n as f64 - 1.0);
            assert!(
                (est - exp).abs() / (n as f64) < 0.02,
                "q{q}: {est} vs {exp}"
            );
        }
    }

    #[test]
    fn skips_non_finite() {
        let mut r = ReservoirL::new(10, 1);
        r.add_many(&[1.0, f64::NAN, 2.0, f64::INFINITY, 3.0]);
        assert_eq!(r.seen(), 3);
        assert_eq!(r.values().len(), 3);
    }
}
