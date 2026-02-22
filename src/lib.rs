//! # qbvec
//!
//! An embedded, enterprise-grade vector memory engine designed to run
//! in-process as part of a larger system (Sentra).
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use qbvec::api::{CollectionConfig, QbvecEngine};
//!
//! let engine = QbvecEngine::open("/tmp/qbvec_data").unwrap();
//! engine.create_collection(CollectionConfig {
//!     name: "embeddings".to_string(),
//!     dimension: 384,
//!     memory_limit_bytes: 0,
//! }).unwrap();
//!
//! engine.add("embeddings", 1, &vec![0.0_f32; 384]).unwrap();
//! let hits = engine.search("embeddings", &vec![0.0_f32; 384], 5).unwrap();
//! ```

/// Semver version string of this engine build.
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod api;
pub mod error;
pub mod index;
pub mod memory;
pub mod metrics;
pub mod query;
pub mod segment;
pub mod storage;

// Re-export the most commonly used types at the crate root.
pub use api::{CollectionConfig, HealthStatus, QbvecEngine};
pub use error::{QbvecError, Result};
pub use query::SearchHit;
