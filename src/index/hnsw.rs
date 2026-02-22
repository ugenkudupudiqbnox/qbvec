use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

use hnsw_rs::prelude::{DistL2, Hnsw, Neighbour};
use serde::{Deserialize, Serialize};

use crate::error::{QbvecError, Result};
use crate::query::SearchHit;

use super::traits::VectorIndex;

// HNSW construction parameters – tuned for correctness, not maximum performance.
const MAX_NB_CONNECTION: usize = 16;
const MAX_ELEMENTS: usize = 100_000;
const MAX_LAYERS: usize = 16;
const EF_CONSTRUCTION: usize = 200;
const EF_SEARCH: usize = 50;

/// The on-disk snapshot format for an `HnswIndex`.
///
/// We keep a mirror of all inserted vectors so we can rebuild the HNSW graph
/// after a reload without depending on the internal hnsw_rs serialisation
/// format (which ties to reference lifetimes).
#[derive(Serialize, Deserialize)]
struct HnswSnapshot {
    dimension: usize,
    vectors: Vec<(u64, Vec<f32>)>,
}

/// HNSW-backed implementation of [`VectorIndex`].
///
/// The implementation is thread-safe: `hnsw_rs` guards its internal structure
/// with `RwLock` already, and we guard the vector mirror with our own `RwLock`.
pub struct HnswIndex {
    /// The underlying hnsw_rs structure.
    ///
    /// `'static` because all inserted data is owned (copied in on insert).
    inner: Hnsw<'static, f32, DistL2>,
    /// Expected dimensionality; validated on every `add`.
    dimension: usize,
    /// Mirror of every inserted vector – used for persistence / reload.
    mirror: RwLock<HashMap<u64, Vec<f32>>>,
}

impl HnswIndex {
    /// Create a new, empty HNSW index for vectors of the given `dimension`.
    pub fn new(dimension: usize) -> Self {
        let inner = Hnsw::<f32, DistL2>::new(
            MAX_NB_CONNECTION,
            MAX_ELEMENTS,
            MAX_LAYERS,
            EF_CONSTRUCTION,
            DistL2 {},
        );
        Self {
            inner,
            dimension,
            mirror: RwLock::new(HashMap::new()),
        }
    }
}

impl VectorIndex for HnswIndex {
    fn add(&self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(QbvecError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }
        // hnsw_rs uses usize as the id type; on all supported 64-bit targets
        // usize == u64, so this cast is lossless.
        self.inner.insert((vector, id as usize));

        let mut mirror = self
            .mirror
            .write()
            .map_err(|_| QbvecError::Index("mirror lock poisoned".into()))?;
        mirror.insert(id, vector.to_vec());
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchHit>> {
        if query.len() != self.dimension {
            return Err(QbvecError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }
        let neighbours = self.inner.search(query, k, EF_SEARCH);
        let hits = neighbours
            .into_iter()
            .map(|n: Neighbour| SearchHit {
                id: n.get_origin_id() as u64,
                score: n.get_distance(),
                segment_id: String::new(), // filled in by the query layer
                engine_version: crate::ENGINE_VERSION.to_string(),
            })
            .collect();
        Ok(hits)
    }

    fn persist(&self, path: &Path) -> Result<()> {
        let mirror = self
            .mirror
            .read()
            .map_err(|_| QbvecError::Index("mirror lock poisoned".into()))?;
        let snapshot = HnswSnapshot {
            dimension: self.dimension,
            vectors: mirror.iter().map(|(&id, v)| (id, v.clone())).collect(),
        };
        let snapshot_path = path.join("hnsw_snapshot.json");
        let file = std::fs::File::create(&snapshot_path).map_err(|e| {
            QbvecError::Index(format!("cannot create snapshot at {}: {e}", snapshot_path.display()))
        })?;
        serde_json::to_writer(file, &snapshot)?;
        Ok(())
    }

    fn load(path: &Path) -> Result<Self> {
        let snapshot_path = path.join("hnsw_snapshot.json");
        let file = std::fs::File::open(&snapshot_path).map_err(|e| {
            QbvecError::Index(format!(
                "cannot open snapshot at {}: {e}",
                snapshot_path.display()
            ))
        })?;
        let snapshot: HnswSnapshot = serde_json::from_reader(file)?;

        let index = Self::new(snapshot.dimension);
        for (id, vector) in &snapshot.vectors {
            index.add(*id, vector)?;
        }
        Ok(index)
    }

    fn len(&self) -> usize {
        self.inner.get_nb_point()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_vec(dim: usize, val: f32) -> Vec<f32> {
        vec![val; dim]
    }

    #[test]
    fn test_add_and_search() {
        let dim = 4;
        let idx = HnswIndex::new(dim);
        for i in 0..10u64 {
            idx.add(i, &make_vec(dim, i as f32)).unwrap();
        }
        assert_eq!(idx.len(), 10);

        let hits = idx.search(&make_vec(dim, 5.0), 3).unwrap();
        assert!(!hits.is_empty());
        // nearest neighbour of [5,5,5,5] should be id 5 (exact match)
        assert_eq!(hits[0].id, 5);
    }

    #[test]
    fn test_dimension_mismatch() {
        let idx = HnswIndex::new(4);
        let err = idx.add(0, &[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, QbvecError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_persist_and_reload() {
        let dir = tempdir().unwrap();
        let dim = 3;
        let idx = HnswIndex::new(dim);
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0]).unwrap();
        idx.persist(dir.path()).unwrap();

        let loaded = HnswIndex::load(dir.path()).unwrap();
        assert_eq!(loaded.len(), 2);
        let hits = loaded.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, 1);
    }
}
