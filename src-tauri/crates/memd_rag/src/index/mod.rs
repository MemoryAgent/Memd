use anyhow::Result;

use crate::component::database::Chunk;

pub mod page;

pub mod index_file;

pub mod buffer_pool;

pub mod executor;

pub trait MemdIndex {
    fn bulk_build_chunk(&mut self, chunks: &[Chunk]) -> Result<()>;
    fn insert_chunk(&mut self, chunks: &[Chunk]) -> Result<()>;
    // TODO: introduce interior mutability to get rid of this mutable reference.
    fn query_chunk(&mut self, query: &[f32], top_k: usize) -> Result<Vec<(usize, f32)>>;
    fn evaluate_memory_usage(&self) -> Result<usize>;
}

pub mod usearch_adapter;

pub mod shm_adapter;
