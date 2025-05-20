use super::MemdIndex;
use crate::component::database::Chunk;
use anyhow::Result;

impl MemdIndex for usearch::Index {
    fn bulk_build_chunk(&mut self, chunks: &[Chunk]) -> Result<()> {
        self.reset()?;
        self.reserve(chunks.len())?;
        for c in chunks {
            self.add(c.id as u64, c.content_vector.as_slice())?;
        }
        Ok(())
    }

    fn insert_chunk(&mut self, chunks: &[Chunk]) -> Result<()> {
        while self.capacity() < self.size() + chunks.len() {
            self.reserve(self.capacity() * 2)?;
        }
        for c in chunks {
            self.add(c.id as u64, c.content_vector.as_slice())?;
        }
        Ok(())
    }

    fn query_chunk(&mut self, query: &[f32], top_k: usize) -> Result<Vec<(usize, f32)>> {
        let matches = self.search(query, top_k)?;
        let keys = matches.keys;
        let distances = matches.distances;

        keys.iter()
            .zip(distances.iter())
            .map(|(k, d)| {
                let id = *k as usize;
                let distance = *d;
                Ok((id, distance))
            })
            .collect()
    }

    fn evaluate_memory_usage(&self) -> Result<usize> {
        Ok(self.memory_usage())
    }
}
