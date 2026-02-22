use std::path::Path;

use crate::error::Result;
use crate::query::SearchHit;

/// A single entry returned from a vector search.
///
/// This is a narrower alias used inside the index layer (no engine metadata yet).
pub type IndexHit = SearchHit;

/// Trait that every vector index implementation must satisfy.
///
/// All methods take `&self` rather than `&mut self` so the index can be shared
/// across read operations behind an `Arc<RwLock<…>>` while still allowing
/// concurrent inserts through internal synchronisation (hnsw_rs uses `RwLock`
/// internally).
pub trait VectorIndex: Sized + Send + Sync {
    /// Insert a vector with the given id.
    fn add(&self, id: u64, vector: &[f32]) -> Result<()>;

    /// Search for the `k` nearest neighbours of `query`.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<IndexHit>>;

    /// Persist the index to the given directory path.
    ///
    /// The callee chooses its own file names inside `path`.
    fn persist(&self, path: &Path) -> Result<()>;

    /// Load an index that was previously persisted to `path`.
    fn load(path: &Path) -> Result<Self>;

    /// Return the number of vectors currently stored in the index.
    fn len(&self) -> usize;

    /// Return `true` when no vectors have been inserted.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
