use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{QbvecError, Result};
use crate::index::{HnswIndex, VectorIndex};
use crate::query::SearchHit;

/// A mutable segment that accepts new vector writes.
pub struct ActiveSegment {
    pub id: String,
    index: Arc<HnswIndex>,
    segment_dir: PathBuf,
}

impl ActiveSegment {
    /// Create a new, empty active segment.
    pub fn new(id: impl Into<String>, dimension: usize, segment_dir: impl AsRef<Path>) -> Self {
        let id = id.into();
        let segment_dir = segment_dir.as_ref().to_path_buf();
        Self {
            id,
            index: Arc::new(HnswIndex::new(dimension)),
            segment_dir,
        }
    }

    /// Reconstruct an active segment from a previously loaded index.
    ///
    /// Used when reopening a collection that was closed while the segment was
    /// still active (not yet sealed).
    pub fn from_index(
        id: impl Into<String>,
        _dimension: usize,
        segment_dir: impl AsRef<Path>,
        index: HnswIndex,
    ) -> Self {
        Self {
            id: id.into(),
            index: Arc::new(index),
            segment_dir: segment_dir.as_ref().to_path_buf(),
        }
    }

    /// Insert a vector.
    pub fn add(&self, id: u64, vector: &[f32]) -> Result<()> {
        self.index.add(id, vector)
    }

    /// Search this segment.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchHit>> {
        let mut hits = self.index.search(query, k)?;
        for hit in &mut hits {
            hit.segment_id = self.id.clone();
        }
        Ok(hits)
    }

    /// Number of vectors in this segment.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// `true` when the segment is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Persist this segment to its directory and return a [`SealedSegment`].
    pub fn seal(self) -> Result<SealedSegment> {
        std::fs::create_dir_all(&self.segment_dir).map_err(|e| {
            QbvecError::Segment(format!(
                "cannot create segment dir {}: {e}",
                self.segment_dir.display()
            ))
        })?;
        self.index.persist(&self.segment_dir)?;
        Ok(SealedSegment {
            id: self.id,
            index: self.index,
            segment_dir: self.segment_dir,
        })
    }
}

/// An immutable (sealed) segment – read-only after creation.
pub struct SealedSegment {
    pub id: String,
    index: Arc<HnswIndex>,
    segment_dir: PathBuf,
}

impl SealedSegment {
    /// Load a sealed segment from its directory.
    pub fn load(id: impl Into<String>, segment_dir: impl AsRef<Path>) -> Result<Self> {
        let id = id.into();
        let segment_dir = segment_dir.as_ref().to_path_buf();
        let index = HnswIndex::load(&segment_dir)?;
        Ok(Self {
            id,
            index: Arc::new(index),
            segment_dir,
        })
    }

    /// Search this segment.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchHit>> {
        let mut hits = self.index.search(query, k)?;
        for hit in &mut hits {
            hit.segment_id = self.id.clone();
        }
        Ok(hits)
    }

    /// Number of vectors in this segment.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// `true` when the segment is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Path to the segment directory.
    pub fn dir(&self) -> &Path {
        &self.segment_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_active_segment_add_and_search() {
        let dir = tempdir().unwrap();
        let seg = ActiveSegment::new("seg-0", 3, dir.path().join("seg-0"));
        seg.add(1, &[1.0, 0.0, 0.0]).unwrap();
        seg.add(2, &[0.0, 1.0, 0.0]).unwrap();
        let hits = seg.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, 1);
        assert_eq!(hits[0].segment_id, "seg-0");
    }

    #[test]
    fn test_seal_and_load() {
        let dir = tempdir().unwrap();
        let seg_dir = dir.path().join("seg-0");
        let seg = ActiveSegment::new("seg-0", 3, &seg_dir);
        seg.add(10, &[1.0, 2.0, 3.0]).unwrap();
        let sealed = seg.seal().unwrap();
        assert_eq!(sealed.len(), 1);

        let reloaded = SealedSegment::load("seg-0", &seg_dir).unwrap();
        assert_eq!(reloaded.len(), 1);
        let hits = reloaded.search(&[1.0, 2.0, 3.0], 1).unwrap();
        assert_eq!(hits[0].id, 10);
    }
}
