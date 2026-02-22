use std::sync::atomic::{AtomicU64, Ordering};

/// Basic in-process telemetry counters.
///
/// All counters are lock-free atomics so they can be updated from any thread
/// without coordination overhead.
pub struct Metrics {
    /// Total vectors inserted across all collections.
    pub vectors_inserted: AtomicU64,
    /// Total search queries executed.
    pub searches_executed: AtomicU64,
    /// Total collections created since startup.
    pub collections_created: AtomicU64,
    /// Number of WAL entries written.
    pub wal_entries_written: AtomicU64,
}

impl Metrics {
    /// Create a zeroed `Metrics` instance.
    pub fn new() -> Self {
        Self {
            vectors_inserted: AtomicU64::new(0),
            searches_executed: AtomicU64::new(0),
            collections_created: AtomicU64::new(0),
            wal_entries_written: AtomicU64::new(0),
        }
    }

    /// Increment the vectors-inserted counter.
    pub fn record_insert(&self) {
        self.vectors_inserted.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the searches-executed counter.
    pub fn record_search(&self) {
        self.searches_executed.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the collections-created counter.
    pub fn record_collection_created(&self) {
        self.collections_created.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the WAL-entries-written counter.
    pub fn record_wal_write(&self) {
        self.wal_entries_written.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot current counters as a plain struct.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            vectors_inserted: self.vectors_inserted.load(Ordering::Relaxed),
            searches_executed: self.searches_executed.load(Ordering::Relaxed),
            collections_created: self.collections_created.load(Ordering::Relaxed),
            wal_entries_written: self.wal_entries_written.load(Ordering::Relaxed),
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// A point-in-time copy of all metric counters.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub vectors_inserted: u64,
    pub searches_executed: u64,
    pub collections_created: u64,
    pub wal_entries_written: u64,
}
