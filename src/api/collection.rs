use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{QbvecError, Result};
use crate::index::VectorIndex;
use crate::memory::MemoryBudget;
use crate::metrics::Metrics;
use crate::query::{merge_results, SearchHit};
use crate::segment::{ActiveSegment, SealedSegment};
use crate::storage::{CollectionMeta, MetadataStore, Wal, WalOperation};

/// Configuration stored as `config.json` inside the collection directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Human-readable name of the collection.
    pub name: String,
    /// Dimensionality of every vector in this collection.
    pub dimension: usize,
    /// Optional memory budget in bytes (0 = unlimited).
    pub memory_limit_bytes: u64,
}

/// A single isolated vector collection.
///
/// Each collection owns its directory, SQLite metadata database, WAL, and
/// an ordered list of segments.
pub struct Collection {
    /// The collection's on-disk configuration.
    pub config: CollectionConfig,
    /// Root directory for this collection.
    collection_dir: PathBuf,
    /// SQLite-backed metadata store.
    metadata: MetadataStore,
    /// Append-only write-ahead log.
    wal: Wal,
    /// The currently active (writable) segment.
    active_segment: Option<ActiveSegment>,
    /// Sealed (immutable) segments, ordered oldest-first.
    sealed_segments: Vec<SealedSegment>,
    /// Memory budget tracker.
    memory_budget: MemoryBudget,
    /// Reference to the engine-level metrics.
    metrics: Arc<Metrics>,
}

impl Collection {
    /// Create a brand-new collection on disk and return it.
    pub fn create(
        base_dir: &Path,
        config: CollectionConfig,
        metrics: Arc<Metrics>,
    ) -> Result<Self> {
        let collection_dir = base_dir.join(&config.name);
        if collection_dir.exists() {
            return Err(QbvecError::CollectionAlreadyExists(config.name.clone()));
        }
        std::fs::create_dir_all(collection_dir.join("segments"))?;

        // Write config.json
        let config_path = collection_dir.join("config.json");
        let config_file = std::fs::File::create(&config_path)?;
        serde_json::to_writer_pretty(config_file, &config)?;

        // Initialise metadata store
        let metadata = MetadataStore::open(collection_dir.join("metadata.db"))?;
        metadata.save_collection(&CollectionMeta {
            name: config.name.clone(),
            dimension: config.dimension,
            created_at: current_timestamp(),
        })?;

        // Initialise WAL
        let wal = Wal::open(collection_dir.join("wal.log"))?;

        // Create the first active segment
        let seg_id = new_segment_id();
        let seg_dir = collection_dir.join("segments").join(&seg_id);
        let active_segment = ActiveSegment::new(&seg_id, config.dimension, seg_dir);
        metadata.record_segment(&seg_id, false)?;

        let memory_budget = MemoryBudget::new(config.memory_limit_bytes);
        metrics.record_collection_created();

        Ok(Self {
            config,
            collection_dir,
            metadata,
            wal,
            active_segment: Some(active_segment),
            sealed_segments: Vec::new(),
            memory_budget,
            metrics,
        })
    }

    /// Open an existing collection from disk.
    pub fn open(base_dir: &Path, name: &str, metrics: Arc<Metrics>) -> Result<Self> {
        let collection_dir = base_dir.join(name);
        if !collection_dir.exists() {
            return Err(QbvecError::CollectionNotFound(name.to_string()));
        }

        // Load config
        let config_path = collection_dir.join("config.json");
        let config_file = std::fs::File::open(&config_path)?;
        let config: CollectionConfig = serde_json::from_reader(config_file)?;

        // Open metadata + WAL
        let metadata = MetadataStore::open(collection_dir.join("metadata.db"))?;
        let wal = Wal::open(collection_dir.join("wal.log"))?;

        // Reload segments
        let segment_records = metadata.list_segments()?;
        let mut sealed_segments: Vec<SealedSegment> = Vec::new();
        let mut active_segment: Option<ActiveSegment> = None;

        for (seg_id, sealed) in segment_records {
            let seg_dir = collection_dir.join("segments").join(&seg_id);
            if sealed {
                let sealed_seg = SealedSegment::load(&seg_id, &seg_dir)?;
                sealed_segments.push(sealed_seg);
            } else {
                // The active (unsealed) segment – reload if it was persisted
                let seg = if seg_dir.exists() {
                    let idx = crate::index::HnswIndex::load(&seg_dir)?;
                    ActiveSegment::from_index(
                        &seg_id,
                        config.dimension,
                        &seg_dir,
                        idx,
                    )
                } else {
                    ActiveSegment::new(&seg_id, config.dimension, &seg_dir)
                };
                active_segment = Some(seg);
            }
        }

        // If no active segment exists (e.g. all were sealed), create a new one.
        if active_segment.is_none() {
            let seg_id = new_segment_id();
            let seg_dir = collection_dir.join("segments").join(&seg_id);
            active_segment = Some(ActiveSegment::new(&seg_id, config.dimension, &seg_dir));
            metadata.record_segment(&seg_id, false)?;
        }

        let memory_budget = MemoryBudget::new(config.memory_limit_bytes);

        Ok(Self {
            config,
            collection_dir,
            metadata,
            wal,
            active_segment,
            sealed_segments,
            memory_budget,
            metrics,
        })
    }

    /// Insert a vector into this collection.
    pub fn add(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        let dimension = self.config.dimension;
        if vector.len() != dimension {
            return Err(QbvecError::DimensionMismatch {
                expected: dimension,
                got: vector.len(),
            });
        }

        let active = self
            .active_segment
            .as_ref()
            .ok_or_else(|| QbvecError::Segment("no active segment".into()))?;

        // Write to WAL first for crash safety
        self.wal.append(WalOperation::Insert {
            id,
            vector: vector.to_vec(),
            segment_id: active.id.clone(),
        })?;
        self.metrics.record_wal_write();

        active.add(id, vector)?;
        self.memory_budget.record_add(dimension);
        self.metrics.record_insert();
        Ok(())
    }

    /// Search this collection for the `k` nearest neighbours of `query`.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchHit>> {
        let mut all_hits: Vec<SearchHit> = Vec::new();

        // Search sealed segments
        for seg in &self.sealed_segments {
            let hits = seg.search(query, k)?;
            all_hits.extend(hits);
        }

        // Search the active segment
        if let Some(active) = &self.active_segment {
            let hits = active.search(query, k)?;
            all_hits.extend(hits);
        }

        self.metrics.record_search();
        Ok(merge_results(all_hits, k))
    }

    /// Seal the active segment and start a new one.
    pub fn rotate_segment(&mut self) -> Result<()> {
        if let Some(active) = self.active_segment.take() {
            let seg_id = active.id.clone();
            let seg_dir = self.collection_dir.join("segments").join(&seg_id);
            std::fs::create_dir_all(&seg_dir)?;
            let sealed = active.seal()?;
            self.metadata.record_segment(&seg_id, true)?;
            self.wal.append(WalOperation::SealSegment {
                segment_id: seg_id.clone(),
            })?;
            self.metrics.record_wal_write();
            self.sealed_segments.push(sealed);
        }

        let new_id = new_segment_id();
        let new_dir = self.collection_dir.join("segments").join(&new_id);
        self.active_segment = Some(ActiveSegment::new(&new_id, self.config.dimension, new_dir));
        self.metadata.record_segment(&new_id, false)?;
        Ok(())
    }

    /// Return the name of this collection.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Return estimated memory usage in bytes.
    pub fn memory_used_bytes(&self) -> u64 {
        self.memory_budget.used_bytes()
    }
}

// --- private helpers -------------------------------------------------------

fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn new_segment_id() -> String {
    Uuid::new_v4().to_string()
}
