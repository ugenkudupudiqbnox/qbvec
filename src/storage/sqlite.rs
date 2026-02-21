use std::path::Path;
use std::sync::Mutex;

use rusqlite::{params, Connection};

use crate::error::Result;

/// Metadata persisted in SQLite for a single collection.
#[derive(Debug, Clone)]
pub struct CollectionMeta {
    pub name: String,
    pub dimension: usize,
    pub created_at: i64,
}

/// Thin wrapper around a `rusqlite::Connection` for collection metadata.
///
/// The connection is guarded by a `Mutex` so that `MetadataStore` is both
/// `Send` and `Sync` and can live inside `Arc<RwLock<Collection>>`.
pub struct MetadataStore {
    conn: Mutex<Connection>,
}

impl MetadataStore {
    /// Open (or create) the metadata database at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path.as_ref())?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.initialize()?;
        Ok(store)
    }

    /// Upsert the collection metadata row.
    pub fn save_collection(&self, meta: &CollectionMeta) -> Result<()> {
        let conn = self.conn.lock().expect("metadata mutex poisoned");
        conn.execute(
            "INSERT OR REPLACE INTO collection_meta (name, dimension, created_at)
             VALUES (?1, ?2, ?3)",
            params![meta.name, meta.dimension as i64, meta.created_at],
        )?;
        Ok(())
    }

    /// Load the collection metadata row (if it exists).
    pub fn load_collection(&self) -> Result<Option<CollectionMeta>> {
        let conn = self.conn.lock().expect("metadata mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT name, dimension, created_at FROM collection_meta LIMIT 1",
        )?;
        let mut rows = stmt.query([])?;
        if let Some(row) = rows.next()? {
            Ok(Some(CollectionMeta {
                name: row.get(0)?,
                dimension: row.get::<_, i64>(1)? as usize,
                created_at: row.get(2)?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Record that a segment exists (by id).
    pub fn record_segment(&self, segment_id: &str, sealed: bool) -> Result<()> {
        let conn = self.conn.lock().expect("metadata mutex poisoned");
        conn.execute(
            "INSERT OR REPLACE INTO segments (segment_id, sealed) VALUES (?1, ?2)",
            params![segment_id, sealed as i64],
        )?;
        Ok(())
    }

    /// List all segment ids.
    pub fn list_segments(&self) -> Result<Vec<(String, bool)>> {
        let conn = self.conn.lock().expect("metadata mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT segment_id, sealed FROM segments ORDER BY rowid",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? != 0))
        })?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    // --- private helpers ---

    fn initialize(&self) -> Result<()> {
        let conn = self.conn.lock().expect("metadata mutex poisoned");
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             CREATE TABLE IF NOT EXISTS collection_meta (
                 name        TEXT    NOT NULL,
                 dimension   INTEGER NOT NULL,
                 created_at  INTEGER NOT NULL,
                 PRIMARY KEY (name)
             );
             CREATE TABLE IF NOT EXISTS segments (
                 segment_id  TEXT    NOT NULL,
                 sealed      INTEGER NOT NULL DEFAULT 0,
                 PRIMARY KEY (segment_id)
             );",
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_save_and_load_collection() {
        let dir = tempdir().unwrap();
        let store = MetadataStore::open(dir.path().join("metadata.db")).unwrap();

        let meta = CollectionMeta {
            name: "test_col".to_string(),
            dimension: 128,
            created_at: 1234567890,
        };
        store.save_collection(&meta).unwrap();

        let loaded = store.load_collection().unwrap().unwrap();
        assert_eq!(loaded.name, "test_col");
        assert_eq!(loaded.dimension, 128);
    }

    #[test]
    fn test_record_and_list_segments() {
        let dir = tempdir().unwrap();
        let store = MetadataStore::open(dir.path().join("metadata.db")).unwrap();

        store.record_segment("seg-0", false).unwrap();
        store.record_segment("seg-1", true).unwrap();

        let segs = store.list_segments().unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].0, "seg-0");
        assert!(!segs[0].1);
        assert!(segs[1].1);
    }
}
