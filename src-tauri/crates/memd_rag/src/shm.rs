//!
//! Shm declares a vector index
//!
//! input: VecID + Vector
//!

use anyhow::{bail, Result};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{Read, Seek, Write},
    ops::{Deref, DerefMut},
    path::PathBuf,
};

type VecId = usize;

const INVALID_VECTOR_ID: usize = usize::MAX;

/// RecordID is the physical position of vector indices.
#[derive(Clone, Copy, Debug)]
pub struct RecordID {
    pub page_id: usize,
    pub page_offset: usize,
}

/// Index page is the basic data structure for storing vector indices, its layout
///
/// | Field     | description     |
/// |-----------|-----------------|
/// | page id           used for inner indices to reference the page
/// | vector unit size  self evident
/// | vector data       embedding data
/// | vector ids        ids for each embedding vector
#[derive(Debug, Clone)]
pub struct LeafPage {
    page_id: usize,
    vector_unit_size: usize,
    max_vectors: usize,
    current_vectors: usize,
    vector_data: Vec<u8>,
    // The alignments of id and data is somehow different.
    // vector_ids is the record id of the text for which the vector is encoded.
    vector_ids: Vec<VecId>,
}

fn read_usize(file: &mut File) -> Result<usize> {
    let mut buffer = [0u8; std::mem::size_of::<usize>()];
    file.read_exact(&mut buffer)?;
    Ok(usize::from_le_bytes(buffer))
}

fn read_vec_u8(file: &mut File, length: usize) -> Result<Vec<u8>> {
    let mut buffer = vec![0u8; length];
    file.read_exact(&mut buffer)?;
    Ok(buffer)
}

fn read_vec_usize(file: &mut File, count: usize) -> Result<Vec<usize>> {
    let mut vec = Vec::with_capacity(count);
    for _ in 0..count {
        vec.push(read_usize(file)?);
    }
    Ok(vec)
}

impl LeafPage {
    const HEADER_SIZE: usize =
        size_of::<LeafPage>() - size_of::<Vec<u8>>() - size_of::<Vec<VecId>>();

    fn calculate_max_vectors(page_size: usize, vector_unit_size: usize) -> usize {
        let page_data_size = page_size - LeafPage::HEADER_SIZE;
        let vector_id_size = size_of::<VecId>();
        page_data_size / (vector_unit_size + vector_id_size)
    }

    pub fn new(page_id: usize, max_vectors: usize, vector_unit_size: usize) -> LeafPage {
        let mut vector_data = Vec::new();
        vector_data.resize(max_vectors * vector_unit_size, 0xff);
        let mut vector_ids = Vec::new();
        vector_ids.resize(max_vectors, INVALID_VECTOR_ID);

        Self {
            page_id,
            max_vectors,
            vector_unit_size,
            current_vectors: 0,
            vector_data,
            vector_ids,
        }
    }

    pub fn memory_usage(&self) -> usize {
        self.vector_data.capacity() + size_of::<VecId>() * self.vector_ids.capacity()
    }

    pub fn bulk_append(&mut self, vectors: &[u8], ids: &[VecId]) -> Result<Vec<RecordID>> {
        let vector_count = ids.len();
        assert!(vector_count * self.vector_unit_size == vectors.len());

        if self.current_vectors * self.vector_unit_size + vectors.len() > self.vector_data.len() {
            bail!("not enough space");
        }

        let new_vector_count = self.current_vectors + vector_count;
        self.vector_data[self.current_vectors * self.vector_unit_size
            ..new_vector_count * self.vector_unit_size]
            .copy_from_slice(vectors);
        self.vector_ids[self.current_vectors..new_vector_count].copy_from_slice(ids);
        self.current_vectors = new_vector_count;

        Ok((0..vector_count)
            .map(|i| RecordID {
                page_id: self.page_id,
                page_offset: (self.current_vectors - vector_count + i) * self.vector_unit_size,
            })
            .collect())
    }

    pub fn append(&mut self, vector: &[u8], id: VecId) -> Result<RecordID> {
        let append_results = self.bulk_append(vector, &[id])?;
        assert_eq!(append_results.len(), 1);
        Ok(append_results[0])
    }

    pub fn get_vectors(&self) -> &[u8] {
        &self.vector_data[..self.current_vectors * self.vector_unit_size]
    }

    pub fn get_ids(&self) -> &[VecId] {
        &self.vector_ids[..self.current_vectors]
    }

    pub fn get_vector(&self, idx: usize) -> &[u8] {
        &self.vector_data[idx * self.vector_unit_size..(idx + 1) * self.vector_unit_size]
    }

    pub fn get_id(&self, idx: usize) -> VecId {
        self.vector_ids[idx]
    }

    pub fn to_zip_iter(&self) -> impl Iterator<Item = (VecId, &[u8])> {
        (0..self.current_vectors).map(|i| {
            let id = self.vector_ids[i];
            let vector =
                &self.vector_data[i * self.vector_unit_size..(i + 1) * self.vector_unit_size];
            (id, vector)
        })
    }

    pub fn persist(&self, f: &mut fs::File) {
        f.write(&self.page_id.to_le_bytes()).unwrap();
        f.write(&self.vector_unit_size.to_le_bytes()).unwrap();
        f.write(&self.max_vectors.to_le_bytes()).unwrap();
        f.write(&self.current_vectors.to_le_bytes()).unwrap();
        f.write(&self.vector_data.as_slice()).unwrap();
        for id in &self.vector_ids {
            f.write(&id.to_le_bytes()).unwrap();
        }
    }

    pub fn from_file(f: &mut fs::File) -> Result<LeafPage> {
        let page_id = read_usize(f)?;
        let vector_unit_size = read_usize(f)?;
        let max_vectors = read_usize(f)?;
        let current_vectors = read_usize(f)?;
        let vector_data = read_vec_u8(f, max_vectors * vector_unit_size)?;
        let vector_ids = read_vec_usize(f, max_vectors)?;
        Ok(LeafPage {
            page_id,
            vector_unit_size,
            max_vectors,
            current_vectors,
            vector_data,
            vector_ids,
        })
    }
}

#[test]
fn test_leaf_page() {
    let mut page = LeafPage::new(0, 10, 4);
    let vectors = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let ids = vec![0, 1];
    let _ = page.bulk_append(&vectors, &ids);
    assert_eq!(page.get_ids(), &[0, 1]);
    assert_eq!(page.get_vector(0), &[1, 2, 3, 4]);
}

#[test]
fn test_persist_load() {
    let mut page = LeafPage::new(0, 10, 4);
    let vectors = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let ids = vec![0, 1];
    let _ = page.bulk_append(&vectors, &ids);
    let mut file = File::create("test_page.bin").unwrap();
    page.persist(&mut file);
    file.flush().unwrap();
    file.sync_all().unwrap();
    let mut file = File::open("test_page.bin").unwrap();
    let loaded_page = LeafPage::from_file(&mut file).unwrap();
    assert_eq!(loaded_page.get_ids(), &[0, 1]);
    assert_eq!(loaded_page.get_vector(0), &[1, 2, 3, 4]);
    assert_eq!(loaded_page.get_vector(1), &[5, 6, 7, 8]);
    assert_eq!(loaded_page.vector_unit_size, 4);
    assert_eq!(loaded_page.max_vectors, 10);
    assert_eq!(loaded_page.current_vectors, 2);
    assert_eq!(loaded_page.page_id, 0);
    fs::remove_file("test_page.bin").unwrap();
}

#[test]
fn test_iterator() {
    let mut page = LeafPage::new(0, 10, 4);
    let vectors = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let ids = vec![0, 1];
    let _ = page.bulk_append(&vectors, &ids);
    let mut iter = page.to_zip_iter();
    assert_eq!(iter.next(), Some((0_usize, &[1_u8, 2, 3, 4][..])));
    assert_eq!(iter.next(), Some((1_usize, &[5_u8, 6, 7, 8][..])));
    assert_eq!(iter.next(), None);
}

#[derive(Clone, Debug, Default)]
pub struct FrameMetadata {
    /// How often the page is accessed.
    temperature: f32,
    /// How many times the page is pinned. pinned page will not be evicted.
    pin_count: i32,
    /// The page is dirty, it needs to be flushed to disk.
    dirty: bool,
}

/// Buffer pool swaps pages between memory and disk.
pub struct BufferPool {
    /// Persistent file.
    /// TODO: make this logic outside buffer pool.
    backed_file: PathBuf,
    /// The largest page id that will be allocated.
    maximum_page_id: usize,
    /// Page size is fixed for whole buffer pool.
    page_size: usize,
    /// TODO: make buffer pool less God object.
    vector_unit_size: usize,
    /// Memory page storage.
    page_storage: Vec<LeafPage>,
    /// page id to page frame index.
    page_table: HashMap<usize, usize>,
    /// list of free page frame ids.
    free_frames: Vec<usize>,
    /// page frame id to page frame metadata.
    metadata: HashMap<usize, FrameMetadata>,
}

impl BufferPool {
    pub fn new(
        backed_file: PathBuf,
        maximum_page_id: usize,
        pool_size: usize,
        page_size: usize,
        vector_unit_size: usize,
    ) -> BufferPool {
        let page_storage = {
            let mut v = Vec::new();
            let max_vectors = LeafPage::calculate_max_vectors(page_size, vector_unit_size);
            v.resize(pool_size, LeafPage::new(0, max_vectors, vector_unit_size));
            v
        };

        let free_page = (0..pool_size).collect();

        Self {
            backed_file,
            maximum_page_id,
            page_size,
            vector_unit_size,
            page_table: HashMap::new(),
            page_storage,
            free_frames: free_page,
            metadata: HashMap::new(),
        }
    }

    pub fn memory_usage(&self) -> usize {
        self.page_storage.iter().map(|p| p.memory_usage()).sum()
    }

    pub fn read_page(&self, page_id: usize) -> Result<LeafPage> {
        let file_size = self.backed_file.metadata()?.len();
        let seek_position = (ShmIndex::METADATA_LENGTH + page_id * self.page_size) as u64;
        if file_size < seek_position {
            bail!("page id exceed disk file limit");
        }
        let mut f = File::open(&self.backed_file)?;
        f.seek(std::io::SeekFrom::Start(seek_position))?;
        let page = LeafPage::from_file(&mut f)?;
        Ok(page)
    }

    pub fn pin_page(&mut self, page_id: usize) {
        if let Some(page) = self.page_table.get(&page_id) {
            self.metadata.get_mut(page).unwrap().pin_count += 1;
            self.metadata.get_mut(page).unwrap().temperature += 1.0;
        } else {
            panic!("page not found");
        }
    }

    pub fn unpin_page(&mut self, page_id: usize) {
        if let Some(page) = self.page_table.get(&page_id) {
            let metadata = self.metadata.get_mut(page).unwrap();
            assert!(metadata.pin_count > 0);
            metadata.pin_count -= 1;
        } else {
            panic!("page not found");
        }
    }

    pub fn mark_dirty(&mut self, page_id: usize) {
        if let Some(page) = self.page_table.get(&page_id) {
            self.metadata.get_mut(page).unwrap().dirty = true;
        } else {
            panic!("page not found");
        }
    }

    pub fn flush_frame(&mut self, frame_id: usize) {
        let metadata = self.metadata.get_mut(&frame_id).unwrap();
        let page_id = self.page_storage[frame_id].page_id;
        if metadata.dirty {
            let mut f = File::open(&self.backed_file).unwrap();
            f.seek(std::io::SeekFrom::Start(
                (ShmIndex::METADATA_LENGTH + page_id * self.page_size)
                    .try_into()
                    .unwrap(),
            ))
            .unwrap();
            self.page_storage[frame_id].persist(&mut f);
            metadata.dirty = false;
        }
        // TODO(SEVERE): add padding to PAGE_SIZE
    }

    pub fn flush_page(&mut self, page_id: usize) {
        if let Some(frame_id) = self.page_table.get(&page_id) {
            self.flush_frame(*frame_id);
        } else {
            panic!("page not found");
        }
    }

    pub fn flush_all(&mut self) {
        let all_frames = self
            .page_table
            .iter()
            .map(|(_, frame_id)| *frame_id)
            .collect::<Vec<_>>();
        for i in all_frames {
            self.flush_frame(i);
        }
    }

    pub fn get_evict_victim(&self) -> Option<usize> {
        let mut candidates = self
            .metadata
            .iter()
            .filter(|(_, metadata)| metadata.pin_count == 0)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return None;
        }
        candidates.sort_by(|(_, a), (_, b)| a.temperature.partial_cmp(&b.temperature).unwrap());

        let (frame_id, _) = candidates[0];
        Some(*frame_id)
    }

    pub fn claim_free_slot(&mut self) -> Option<usize> {
        if let Some(page_id) = self.free_frames.pop() {
            return Some(page_id);
        }
        if let Some(victim) = self.get_evict_victim() {
            let page_id = self.page_storage[victim].page_id;
            self.flush_page(page_id);
            self.page_table.remove(&page_id);
            self.metadata.remove(&page_id);
            return Some(victim);
        }
        None
    }
}

pub struct LeafPageGuard {
    page_id: usize,
    page_frame_id: usize,
    buffer_pool: *mut BufferPool,
}

impl LeafPageGuard {
    pub fn new(
        page_id: usize,
        page_frame_id: usize,
        buffer_pool: *mut BufferPool,
    ) -> LeafPageGuard {
        // SAFETY: single thread, buffer pool live longer than this method.
        unsafe {
            buffer_pool.as_mut().unwrap().pin_page(page_id);
        }
        LeafPageGuard {
            page_id,
            page_frame_id,
            buffer_pool,
        }
    }
}

impl Drop for LeafPageGuard {
    fn drop(&mut self) {
        // SAFETY: single thread, buffer pool live longer than page guard.
        unsafe {
            self.buffer_pool.as_mut().unwrap().unpin_page(self.page_id);
        }
    }
}

impl Deref for LeafPageGuard {
    type Target = LeafPage;

    fn deref(&self) -> &Self::Target {
        // SAFETY: single thread, buffer pool live longer than page guard.
        let page = unsafe { &(*self.buffer_pool).page_storage[self.page_frame_id] };
        assert!(page.page_id == self.page_id);
        page
    }
}

impl DerefMut for LeafPageGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: single thread, buffer pool live longer than page guard.
        let page = unsafe { &mut (*self.buffer_pool).page_storage[self.page_frame_id] };
        assert!(page.page_id == self.page_id);
        page
    }
}

impl BufferPool {
    pub fn fetch_page(&mut self, page_id: usize) -> LeafPageGuard {
        if let Some(frame_id) = self.page_table.get(&page_id) {
            return LeafPageGuard::new(page_id, *frame_id, self);
        }
        if let Some(free_frame_id) = self.free_frames.pop() {
            let page = self.read_page(page_id).unwrap();
            self.page_storage[free_frame_id] = page;
            self.page_table.insert(page_id, free_frame_id);
            self.metadata
                .insert(free_frame_id, FrameMetadata::default());
            return LeafPageGuard::new(page_id, free_frame_id, self);
        }
        panic!("no free page");
    }

    pub fn create_page(&mut self) -> usize {
        let page_id = self.maximum_page_id;
        self.maximum_page_id += 1;

        let max_vectors = (self.page_size - 32) / (self.vector_unit_size + 8);

        if let Some(frame_id) = self.claim_free_slot() {
            let page = LeafPage::new(page_id, max_vectors, self.vector_unit_size);
            self.page_storage[frame_id] = page;
            self.page_table.insert(page_id, frame_id);
            self.metadata.insert(frame_id, FrameMetadata::default());
            return page_id;
        }
        panic!("no free page");
    }
}

#[test]
fn bufferpool_new() {
    let mut buffer_pool = BufferPool::new("test.bin".into(), 0, 10, 4096, 4);
    assert_eq!(buffer_pool.page_size, 4096);
    assert_eq!(buffer_pool.vector_unit_size, 4);
    assert_eq!(buffer_pool.page_storage.capacity(), 10);
    assert_eq!(buffer_pool.free_frames.len(), 10);
    buffer_pool.flush_all();
}

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
        let backed_file = options.backed_file;
        if backed_file.exists() {
            fs::remove_file(&backed_file)?;
        }
        let buffer_pool = BufferPool::new(
            backed_file,
            0,
            options.pool_size,
            options.page_size,
            options.vector_unit_size,
        );
        Ok(Self { buffer_pool })
    }

    pub fn load(file: PathBuf, pool_size: usize) -> Result<Self> {
        if !file.exists() {
            bail!("file not found");
        }
        let mut f = File::open(&file)?;
        let mut header = [0u8; 8];
        f.read_exact(&mut header)?;
        let magic = u64::from_le_bytes(header);
        if magic != Self::MAGIC_HEADER {
            bail!("invalid file format");
        }
        let page_size = read_usize(&mut f)?;
        let vector_unit_size = read_usize(&mut f)?;
        let maximum_page_id = read_usize(&mut f)?;

        let buffer_pool = BufferPool::new(
            file,
            maximum_page_id,
            pool_size,
            page_size,
            vector_unit_size,
        );
        Ok(Self { buffer_pool })
    }

    pub fn persist(&mut self) -> Result<()> {
        let mut f = File::create(&self.buffer_pool.backed_file)?;
        f.write_all(&Self::MAGIC_HEADER.to_le_bytes())?;
        f.write_all(&(self.buffer_pool.page_size).to_le_bytes())?;
        f.write_all(&(self.buffer_pool.vector_unit_size).to_le_bytes())?;
        f.write_all(&(self.buffer_pool.maximum_page_id).to_le_bytes())?;
        self.buffer_pool.flush_all();
        Ok(())
    }

    /// insert one vector into the index
    pub fn insert(&mut self, vector_data: &[u8], vector_id: VecId) -> Result<RecordID> {
        let latest_page_id = if self.buffer_pool.maximum_page_id > 0 {
            self.buffer_pool.maximum_page_id - 1
        } else {
            self.buffer_pool.create_page()
        };

        // RAII scope for latest page
        let latest_adequate_page_id = {
            let latest_page = self.buffer_pool.fetch_page(latest_page_id);
            if latest_page.current_vectors >= latest_page.max_vectors {
                self.buffer_pool.create_page()
            } else {
                latest_page.page_id
            }
        };

        // RAII scope for usable page_id
        let rid = {
            let mut page = self.buffer_pool.fetch_page(latest_adequate_page_id);
            let rid = page.append(vector_data, vector_id)?;
            self.buffer_pool.mark_dirty(page.page_id);
            rid
        };

        Ok(rid)
    }

    pub fn query(&mut self, target_vector: &[u8]) -> Result<TopKBuffer> {
        let target_vector = reinterpret_as_f32(target_vector);
        let mut topk = TopKBuffer::new(10);
        for page_id in 0..self.buffer_pool.maximum_page_id {
            let page = self.buffer_pool.fetch_page(page_id);
            for (id, vector) in page.to_zip_iter() {
                let vector = reinterpret_as_f32(vector);
                let score = calculate_similarity(vector, target_vector);
                topk.push(id, score);
            }
        }
        Ok(topk)
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
        assert_eq!(id.page_offset, 0);
        let target_vector = 4.0_f32.to_le_bytes();
        let topk = index.query(&target_vector).unwrap();
        println!("topk: {:?}", topk.get_topk());
    }
}
