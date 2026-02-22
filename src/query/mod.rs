/// The result of a single vector similarity search.
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// The user-supplied id of the matching vector.
    pub id: u64,
    /// Similarity / distance score (lower = closer for L2-based indices).
    pub score: f32,
    /// The segment that produced this result.
    pub segment_id: String,
    /// Version string of the engine that produced this result.
    pub engine_version: String,
}

/// Merges per-segment result lists into a single top-k list.
///
/// Results are sorted by ascending score (smaller distance first).
pub fn merge_results(mut results: Vec<SearchHit>, k: usize) -> Vec<SearchHit> {
    results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_results_sorts_and_truncates() {
        let hits = vec![
            SearchHit { id: 1, score: 3.0, segment_id: "s0".into(), engine_version: "0.1.0".into() },
            SearchHit { id: 2, score: 1.0, segment_id: "s0".into(), engine_version: "0.1.0".into() },
            SearchHit { id: 3, score: 2.0, segment_id: "s1".into(), engine_version: "0.1.0".into() },
        ];
        let merged = merge_results(hits, 2);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].id, 2);
        assert_eq!(merged[1].id, 3);
    }
}
