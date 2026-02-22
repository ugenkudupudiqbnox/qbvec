use std::sync::atomic::{AtomicU64, Ordering};

/// Tracks estimated memory consumption for a collection.
///
/// Memory is estimated conservatively as `num_vectors * dimension * 4` bytes
/// (each f32 is 4 bytes) plus overhead per vector.
pub struct MemoryBudget {
    /// Hard limit in bytes (0 = unlimited).
    limit_bytes: u64,
    /// Current estimated usage in bytes.
    used_bytes: AtomicU64,
    /// Fixed overhead (bytes) per stored vector.
    overhead_per_vector: u64,
}

impl MemoryBudget {
    const DEFAULT_OVERHEAD: u64 = 256;

    /// Create a new budget with the given hard limit (`0` = unlimited).
    pub fn new(limit_bytes: u64) -> Self {
        Self {
            limit_bytes,
            used_bytes: AtomicU64::new(0),
            overhead_per_vector: Self::DEFAULT_OVERHEAD,
        }
    }

    /// Record the addition of one vector of the given dimension.
    pub fn record_add(&self, dimension: usize) {
        let delta = (dimension as u64) * 4 + self.overhead_per_vector;
        self.used_bytes.fetch_add(delta, Ordering::Relaxed);
    }

    /// Return the current estimated memory usage in bytes.
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes.load(Ordering::Relaxed)
    }

    /// Return the configured limit in bytes (`0` = unlimited).
    pub fn limit_bytes(&self) -> u64 {
        self.limit_bytes
    }

    /// `true` when the limit is configured and has been exceeded.
    pub fn is_over_budget(&self) -> bool {
        self.limit_bytes > 0 && self.used_bytes() > self.limit_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracking() {
        let budget = MemoryBudget::new(1024);
        budget.record_add(4);
        // 4 * 4 + 256 = 272
        assert_eq!(budget.used_bytes(), 272);
        assert!(!budget.is_over_budget());
        // Add enough to exceed budget
        budget.record_add(4);
        // 272 * 2 = 544 – still under 1024
        assert!(!budget.is_over_budget());
    }

    #[test]
    fn test_unlimited_budget() {
        let budget = MemoryBudget::new(0);
        budget.record_add(128);
        assert!(!budget.is_over_budget());
    }
}
