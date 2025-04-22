use anyhow::Result;
use std::path::PathBuf;

use super::{buffer_pool::BufferPool, index_file::IndexFile, page::RecordID};

#[derive(Clone, Debug)]
pub struct TopKBuffer {
    /// vector ID -> conf_score
    buffer: Vec<(usize, f32)>,
    /// top k
    k: usize,
}

impl TopKBuffer {
    pub fn new(k: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(k),
            k,
        }
    }

    pub fn push(&mut self, id: usize, score: f32) {
        if self.buffer.len() < self.k {
            self.buffer.push((id, score));
            // Total order comparison of floats is supported since Rust 1.62
            self.buffer.sort_by(|a, b| a.1.total_cmp(&b.1));
        } else if score > self.buffer[0].1 {
            self.buffer[0] = (id, score);
            self.buffer.sort_by(|a, b| a.1.total_cmp(&b.1));
        }
    }

    pub fn get_topk(&self) -> &[(usize, f32)] {
        &self.buffer
    }
}

/// TODO: this is to be replaced by FAISS because I don't want to write SIMD optimizations ...
fn calculate_similarity(vec0: &[f32], vec1: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..vec0.len() {
        sum += vec0[i] * vec1[i];
    }
    sum / (vec0.len() as f32)
}

/// reinterpret &[u8] as &[f32]
fn reinterpret_as_f32<'a>(slice: &'a [u8]) -> &'a [f32] {
    assert!(slice.len() % std::mem::size_of::<f32>() == 0);
    let len = slice.len() / std::mem::size_of::<f32>();
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f32, len) }
}

pub struct ShmIndex {
    buffer_pool: BufferPool,
}

#[derive(Clone, Debug)]
pub struct ShmIndexOptions {
    /// The path to the file that will be used for backing store.
    pub backed_file: PathBuf,
    /// The size of the buffer pool.
    pub pool_size: usize,
    /// The size of each page.
    pub page_size: usize,
    /// The size of each vector unit.
    pub vector_unit_size: usize,
}

impl ShmIndex {
    const MAGIC_HEADER: u64 = 0xDEADBEEFDEADBEEF;

    const METADATA_LENGTH: usize = size_of::<u32>() + size_of::<usize>() * 2;

    pub fn new(options: ShmIndexOptions) -> Result<Self> {
        let backed_file = IndexFile::create(
            options.backed_file,
            options.page_size,
            options.vector_unit_size,
        )?;
        let buffer_pool = BufferPool::new(
            backed_file,
            options.pool_size,
            options.page_size,
            options.vector_unit_size,
        );
        Ok(Self { buffer_pool })
    }

    pub fn load(file: PathBuf, pool_size: usize) -> Result<Self> {
        let backed_file = IndexFile::open(file)?;
        let page_size = backed_file.page_size;
        let vector_unit_size = backed_file.vector_unit_size;

        let buffer_pool = BufferPool::new(backed_file, pool_size, page_size, vector_unit_size);

        Ok(Self { buffer_pool })
    }

    pub fn persist(&mut self) -> Result<()> {
        self.buffer_pool.flush_all();
        Ok(())
    }

    /// insert one vector into the index
    pub fn insert(&mut self, vector_data: &[u8], vector_id: usize) -> Result<RecordID> {
        todo!()
    }

    pub fn query(&mut self, target_vector: &[u8]) -> Result<TopKBuffer> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_shm_index() {
        let mut index = ShmIndex::new(ShmIndexOptions {
            backed_file: PathBuf::from("test.bin"),
            pool_size: 10,
            page_size: 4096,
            vector_unit_size: 4,
        })
        .unwrap();
        let vector = 4.0_f32.to_le_bytes();
        let id = index.insert(&vector, 0).unwrap();
        println!("inserted rid: {:?}", id);
        assert_eq!(id.page_id, 0);
        assert_eq!(id.slot_id, 0);
        let target_vector = 4.0_f32.to_le_bytes();
        let topk = index.query(&target_vector).unwrap();
        println!("topk: {:?}", topk.get_topk());
    }
}
