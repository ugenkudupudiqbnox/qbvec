use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::api::collection::{Collection, CollectionConfig};
use crate::error::{QbvecError, Result};
use crate::metrics::{Metrics, MetricsSnapshot};
use crate::query::SearchHit;

/// Health information returned by [`QbvecEngine::health_check`].
#[derive(Debug, Clone)]
pub struct HealthStatus {
    /// The engine is healthy when this is `true`.
    pub ok: bool,
    /// Human-readable message.
    pub message: String,
    /// Current metric counters.
    pub metrics: MetricsSnapshot,
}

/// The central in-process vector engine.
///
/// `QbvecEngine` manages one or more isolated collections stored under a
/// common base directory.  The engine is thread-safe: multiple callers may
/// hold a shared `&QbvecEngine` and perform reads concurrently; writes are
/// serialised at the collection level via `RwLock`.
pub struct QbvecEngine {
    /// Root directory that holds one sub-directory per collection.
    base_dir: PathBuf,
    /// Live collection handles, keyed by collection name.
    ///
    /// Outer `RwLock` guards the map itself; inner `RwLock` guards each
    /// individual collection (multiple readers / single writer).
    collections: RwLock<HashMap<String, Arc<RwLock<Collection>>>>,
    /// Engine-level metrics shared across all collections.
    metrics: Arc<Metrics>,
}

impl QbvecEngine {
    /// Create (or reopen) an engine whose data lives under `base_dir`.
    ///
    /// Any collections that were previously created are *not* automatically
    /// reopened; call [`open_collection`](Self::open_collection) explicitly.
    pub fn open(base_dir: impl AsRef<Path>) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_dir)?;
        Ok(Self {
            base_dir,
            collections: RwLock::new(HashMap::new()),
            metrics: Arc::new(Metrics::new()),
        })
    }

    /// Create a brand-new collection.
    ///
    /// Returns `Err(CollectionAlreadyExists)` if a collection with that name
    /// already exists on disk.
    pub fn create_collection(&self, config: CollectionConfig) -> Result<()> {
        if config.dimension == 0 {
            return Err(QbvecError::InvalidConfig(
                "dimension must be greater than zero".into(),
            ));
        }
        let name = config.name.clone();
        let collection =
            Collection::create(&self.base_dir, config, Arc::clone(&self.metrics))?;

        let mut map = self
            .collections
            .write()
            .map_err(|_| QbvecError::Storage("collections lock poisoned".into()))?;
        map.insert(name, Arc::new(RwLock::new(collection)));
        Ok(())
    }

    /// Open an existing collection from disk and register it with this engine.
    ///
    /// Returns `Err(CollectionNotFound)` if the collection directory does not
    /// exist.
    pub fn open_collection(&self, name: &str) -> Result<()> {
        let collection =
            Collection::open(&self.base_dir, name, Arc::clone(&self.metrics))?;

        let mut map = self
            .collections
            .write()
            .map_err(|_| QbvecError::Storage("collections lock poisoned".into()))?;
        map.insert(name.to_string(), Arc::new(RwLock::new(collection)));
        Ok(())
    }

    /// Return a cloned `Arc` handle to the named collection.
    pub fn get_collection(&self, name: &str) -> Result<Arc<RwLock<Collection>>> {
        let map = self
            .collections
            .read()
            .map_err(|_| QbvecError::Storage("collections lock poisoned".into()))?;
        map.get(name)
            .cloned()
            .ok_or_else(|| QbvecError::CollectionNotFound(name.to_string()))
    }

    /// Insert a vector into the named collection.
    pub fn add(&self, collection: &str, id: u64, vector: &[f32]) -> Result<()> {
        let handle = self.get_collection(collection)?;
        let mut col = handle
            .write()
            .map_err(|_| QbvecError::Storage("collection lock poisoned".into()))?;
        col.add(id, vector)
    }

    /// Search the named collection for the `k` nearest neighbours of `query`.
    pub fn search(&self, collection: &str, query: &[f32], k: usize) -> Result<Vec<SearchHit>> {
        let handle = self.get_collection(collection)?;
        let col = handle
            .read()
            .map_err(|_| QbvecError::Storage("collection lock poisoned".into()))?;
        col.search(query, k)
    }

    /// Return a basic health status for the engine.
    pub fn health_check(&self) -> HealthStatus {
        let metrics = self.metrics.snapshot();
        HealthStatus {
            ok: true,
            message: format!(
                "qbvec {} healthy – {} collection(s) loaded",
                crate::ENGINE_VERSION,
                self.collections
                    .read()
                    .map(|m| m.len())
                    .unwrap_or(0)
            ),
            metrics,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_engine() -> (QbvecEngine, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let engine = QbvecEngine::open(dir.path()).unwrap();
        (engine, dir)
    }

    fn cfg(name: &str, dim: usize) -> CollectionConfig {
        CollectionConfig {
            name: name.to_string(),
            dimension: dim,
            memory_limit_bytes: 0,
        }
    }

    #[test]
    fn test_create_and_search() {
        let (engine, _dir) = make_engine();
        engine.create_collection(cfg("my_col", 4)).unwrap();

        engine.add("my_col", 1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        engine.add("my_col", 2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        engine.add("my_col", 3, &[0.0, 0.0, 1.0, 0.0]).unwrap();

        let hits = engine.search("my_col", &[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, 1);
    }

    #[test]
    fn test_duplicate_collection_error() {
        let (engine, _dir) = make_engine();
        engine.create_collection(cfg("dup", 4)).unwrap();
        let err = engine.create_collection(cfg("dup", 4)).unwrap_err();
        assert!(matches!(err, QbvecError::CollectionAlreadyExists(_)));
    }

    #[test]
    fn test_collection_not_found() {
        let (engine, _dir) = make_engine();
        let err = engine.search("ghost", &[1.0], 1).unwrap_err();
        assert!(matches!(err, QbvecError::CollectionNotFound(_)));
    }

    #[test]
    fn test_health_check() {
        let (engine, _dir) = make_engine();
        let status = engine.health_check();
        assert!(status.ok);
    }

    #[test]
    fn test_zero_dimension_rejected() {
        let (engine, _dir) = make_engine();
        let err = engine.create_collection(cfg("bad", 0)).unwrap_err();
        assert!(matches!(err, QbvecError::InvalidConfig(_)));
    }
}
