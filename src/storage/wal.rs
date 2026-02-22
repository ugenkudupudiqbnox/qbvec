use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{QbvecError, Result};

/// A single entry in the Write-Ahead Log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// The kind of operation being logged.
    pub operation: WalOperation,
}

/// Operations that can be written to the WAL.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WalOperation {
    /// A vector was inserted.
    Insert {
        id: u64,
        vector: Vec<f32>,
        segment_id: String,
    },
    /// A segment was sealed (made read-only).
    SealSegment { segment_id: String },
    /// A checkpoint was recorded (all entries before this are durable).
    Checkpoint { sequence: u64 },
}

/// Append-only Write-Ahead Log for crash recovery.
///
/// Each line in the file is a newline-delimited JSON `WalEntry`.
pub struct Wal {
    writer: BufWriter<File>,
    path: std::path::PathBuf,
    next_sequence: u64,
}

impl Wal {
    /// Open (or create) a WAL at the given path.
    ///
    /// Replays existing entries to determine the next sequence number.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let next_sequence = if path.exists() {
            Self::last_sequence(&path)? + 1
        } else {
            0
        };

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| QbvecError::Wal(format!("cannot open WAL at {}: {}", path.display(), e)))?;

        Ok(Self {
            writer: BufWriter::new(file),
            path,
            next_sequence,
        })
    }

    /// Append an operation to the WAL and flush to disk.
    pub fn append(&mut self, operation: WalOperation) -> Result<u64> {
        let seq = self.next_sequence;
        let entry = WalEntry { sequence: seq, operation };
        let line = serde_json::to_string(&entry)?;
        self.writer
            .write_all(line.as_bytes())
            .map_err(|e| QbvecError::Wal(format!("write failed: {e}")))?;
        self.writer
            .write_all(b"\n")
            .map_err(|e| QbvecError::Wal(format!("write failed: {e}")))?;
        self.writer
            .flush()
            .map_err(|e| QbvecError::Wal(format!("flush failed: {e}")))?;
        self.next_sequence += 1;
        Ok(seq)
    }

    /// Read all entries from the WAL (for replay on startup).
    pub fn read_all(path: impl AsRef<Path>) -> Result<Vec<WalEntry>> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(vec![]);
        }
        let file = File::open(path)
            .map_err(|e| QbvecError::Wal(format!("cannot open WAL: {e}")))?;
        let reader = BufReader::new(file);
        use std::io::BufRead;
        let mut entries = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| QbvecError::Wal(format!("read line failed: {e}")))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let entry: WalEntry = serde_json::from_str(trimmed)
                .map_err(|e| QbvecError::Wal(format!("corrupt WAL entry: {e}")))?;
            entries.push(entry);
        }
        Ok(entries)
    }

    /// Return the path of this WAL file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    // --- private helpers ---

    fn last_sequence(path: &Path) -> Result<u64> {
        let entries = Self::read_all(path)?;
        Ok(entries.last().map(|e| e.sequence).unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_wal_append_and_replay() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal.log");

        let mut wal = Wal::open(&wal_path).unwrap();
        let seq0 = wal
            .append(WalOperation::Insert {
                id: 1,
                vector: vec![0.1, 0.2, 0.3],
                segment_id: "seg-0".to_string(),
            })
            .unwrap();
        let seq1 = wal
            .append(WalOperation::Checkpoint { sequence: seq0 })
            .unwrap();
        assert_eq!(seq0, 0);
        assert_eq!(seq1, 1);

        let entries = Wal::read_all(&wal_path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].sequence, 0);
        assert_eq!(entries[1].sequence, 1);
    }

    #[test]
    fn test_wal_sequence_continues_after_reopen() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal.log");

        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append(WalOperation::SealSegment {
                segment_id: "seg-0".to_string(),
            })
            .unwrap();
        }

        let mut wal2 = Wal::open(&wal_path).unwrap();
        let seq = wal2
            .append(WalOperation::Checkpoint { sequence: 0 })
            .unwrap();
        assert_eq!(seq, 1);
    }
}
