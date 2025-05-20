use super::{executor::ShmIndex, MemdIndex};
use crate::component::database::Chunk;
use anyhow::Result;

impl MemdIndex for ShmIndex {
    fn bulk_build_chunk(&mut self, chunks: &[Chunk]) -> Result<()> {
        self.bulk_build(chunks);
        Ok(())
    }

    fn insert_chunk(&mut self, chunks: &[Chunk]) -> Result<()> {
        self.insert(chunks)?;
        Ok(())
    }

    fn query_chunk(&mut self, query: &[f32], top_k: usize) -> Result<Vec<(usize, f32)>> {
        let matches = self.query(query, top_k)?;
        matches
            .get_topk()
            .iter()
            .map(|(_, key, conf_score, _)| {
                let id = *key as usize;
                let distance = *conf_score;
                Ok((id, distance))
            })
            .collect()
    }

    fn evaluate_memory_usage(&self) -> Result<usize> {
        Ok(self.memory_usage())
    }
}
