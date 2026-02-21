use thiserror::Error;

/// Top-level error type for qbvec.
#[derive(Debug, Error)]
pub enum QbvecError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Collection '{0}' already exists")]
    CollectionAlreadyExists(String),

    #[error("Collection '{0}' not found")]
    CollectionNotFound(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Index error: {0}")]
    Index(String),

    #[error("Segment error: {0}")]
    Segment(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("WAL error: {0}")]
    Wal(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Convenient Result alias used throughout qbvec.
pub type Result<T> = std::result::Result<T, QbvecError>;
