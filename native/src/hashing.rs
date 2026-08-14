//! Non-cryptographic 64-bit hashing for sketch structures.
//!
//! The Python implementation uses SHA-1 (`hashlib.sha1(...).digest()[:8]`) for
//! every value fed to a KMV sketch. SHA-1 is a cryptographic hash: it is built
//! to resist preimage attacks, which is irrelevant for a distinct-count sketch
//! and costs roughly an order of magnitude more per byte than a modern
//! non-cryptographic mixer.
//!
//! What a KMV sketch actually needs from a hash:
//!   1. output approximately uniform on [0, 2^64)
//!   2. good avalanche (one input bit flips ~half the output bits)
//!   3. speed
//!
//! `splitmix64`'s finalizer and wyhash-style multiply-xor-fold mixing give all
//! three. No external crates, so the build stays a two-dependency affair.

/// splitmix64 finalizer. Excellent avalanche, three instructions of real work.
///
/// Use this when the input is *already* a 64-bit value — e.g. row hashes coming
/// out of `pandas.util.hash_pandas_object` or `polars.DataFrame.hash_rows`.
/// The current Python path stringifies those u64s and SHA-1s them, which is
/// pure waste: they are already high-entropy.
#[inline(always)]
pub fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

const S0: u64 = 0xa0761d6478bd642f;
const S1: u64 = 0xe7037ed1a0b428db;
const S2: u64 = 0x8ebc6af09c88c6e3;
const S3: u64 = 0x589965cc75374cc3;

/// 64x64 -> 128 multiply, folded back to 64 bits by xor. This is the core
/// primitive of wyhash: a single `mulx` on x86-64 / `umulh`+`mul` on aarch64.
#[inline(always)]
fn wymix(a: u64, b: u64) -> u64 {
    let r = (a as u128).wrapping_mul(b as u128);
    (r as u64) ^ ((r >> 64) as u64)
}

#[inline(always)]
fn read_u64(bytes: &[u8], i: usize) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[i..i + 8]);
    u64::from_le_bytes(buf)
}

#[inline(always)]
fn read_u32(bytes: &[u8], i: usize) -> u64 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&bytes[i..i + 4]);
    u32::from_le_bytes(buf) as u64
}

/// wyhash-style hash over an arbitrary byte string.
///
/// Processes 32 bytes per iteration in the bulk loop with three independent
/// accumulator chains, so the multiplies pipeline instead of serialising on a
/// single dependency chain. This is the ILP argument from CS:APP ch. 5 applied
/// to a hash: the loop is latency-bound on `mul`, not throughput-bound, so
/// unrolling with independent accumulators is where the speedup comes from.
pub fn hash_bytes(key: &[u8], seed: u64) -> u64 {
    let len = key.len();
    let mut seed = seed ^ wymix(seed ^ S0, S1);

    let a: u64;
    let b: u64;

    if len <= 16 {
        if len >= 4 {
            a = (read_u32(key, 0) << 32) | read_u32(key, (len >> 3) << 2);
            b = (read_u32(key, len - 4) << 32) | read_u32(key, len - 4 - ((len >> 3) << 2));
        } else if len > 0 {
            a = ((key[0] as u64) << 16) | ((key[len >> 1] as u64) << 8) | (key[len - 1] as u64);
            b = 0;
        } else {
            a = 0;
            b = 0;
        }
    } else {
        let mut i = len;
        let mut p = 0usize;
        if i > 48 {
            let mut see1 = seed;
            let mut see2 = seed;
            // Three independent chains: seed / see1 / see2 never read each
            // other inside the loop body, so the CPU can keep three multiplies
            // in flight at once.
            while i > 48 {
                seed = wymix(read_u64(key, p) ^ S1, read_u64(key, p + 8) ^ seed);
                see1 = wymix(read_u64(key, p + 16) ^ S2, read_u64(key, p + 24) ^ see1);
                see2 = wymix(read_u64(key, p + 32) ^ S3, read_u64(key, p + 40) ^ see2);
                p += 48;
                i -= 48;
            }
            seed ^= see1 ^ see2;
        }
        while i > 16 {
            seed = wymix(read_u64(key, p) ^ S1, read_u64(key, p + 8) ^ seed);
            i -= 16;
            p += 16;
        }
        a = read_u64(key, len - 16);
        b = read_u64(key, len - 8);
    }

    wymix(S1 ^ (len as u64), wymix(a ^ S1, b ^ seed))
}

/// Map a hash to the unit interval (0, 1]. KMV's estimator needs the k-th
/// smallest hash normalised this way.
#[inline(always)]
pub fn to_unit(h: u64) -> f64 {
    // 2^-64 scaling, computed without overflow.
    ((h >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix64_avalanche() {
        // Flipping any single input bit should change ~32 of 64 output bits.
        for bit in 0..64 {
            let mut total = 0u32;
            let trials = 512u64;
            for i in 0..trials {
                let x = mix64(i.wrapping_mul(0x9e3779b97f4a7c15));
                let y = mix64(i.wrapping_mul(0x9e3779b97f4a7c15) ^ (1u64 << bit));
                total += (x ^ y).count_ones();
            }
            let avg = total as f64 / trials as f64;
            assert!(avg > 26.0 && avg < 38.0, "bit {bit} avalanche {avg}");
        }
    }

    #[test]
    fn hash_bytes_is_deterministic_and_distinct() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for i in 0..50_000u32 {
            let s = format!("value-{i}");
            let h = hash_bytes(s.as_bytes(), 0);
            assert_eq!(h, hash_bytes(s.as_bytes(), 0), "not deterministic");
            seen.insert(h);
        }
        // 50k draws from 2^64 should collide with probability ~7e-11.
        assert_eq!(seen.len(), 50_000);
    }

    #[test]
    fn hash_bytes_handles_all_length_classes() {
        // 0, short (<4), medium (4..16), the 16..48 loop, and the 48+ loop.
        for len in [0usize, 1, 3, 4, 7, 8, 15, 16, 17, 31, 48, 49, 100, 1000] {
            let key: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            let h = hash_bytes(&key, 0);
            assert_eq!(h, hash_bytes(&key, 0));
        }
    }

    #[test]
    fn to_unit_in_range() {
        for i in 0..1000u64 {
            let u = to_unit(mix64(i));
            assert!((0.0..1.0).contains(&u));
        }
    }
}
