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

/// Index page is the basic data structure for storing vector indices, its layout
///
/// | Field     | description     |
/// |-----------|-----------------|
/// | page id           used for inner indices to reference the page
/// | vector unit size  self evident
/// | vector data       embedding data
/// | vector ids        ids for each embedding vector
#[derive(Debug)]
pub struct LeafPage {
    page_id: usize,
    vector_unit_size: usize,
    max_vectors: usize,
    current_vectors: usize,
    // TODO: consider zero copy
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

    pub fn append(&mut self, vectors: &[u8], ids: &[VecId]) {
        let vector_count = ids.len();
        assert!(vectors.len() % self.vector_unit_size == 0);
        assert!(vector_count * self.vector_unit_size == vectors.len());
        if self.current_vectors + vectors.len() > self.vector_data.len() {
            panic!("not enough space");
        }
        let new_vector_count = self.current_vectors + vector_count;
        self.vector_data[self.current_vectors * self.vector_unit_size
            ..new_vector_count * self.vector_unit_size]
            .copy_from_slice(vectors);
        self.vector_ids[self.current_vectors..new_vector_count].copy_from_slice(ids);
        self.current_vectors = new_vector_count;
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
        self.vector_ids
            .iter()
            .zip(self.vector_data.chunks_exact(self.vector_unit_size))
            .map(|(id, data)| (*id, data))
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
    page.append(&vectors, &ids);
    assert_eq!(page.get_ids(), &[0, 1]);
    assert_eq!(page.get_vector(0), &[1, 2, 3, 4]);
}

#[test]
fn test_persist_load() {
    let mut page = LeafPage::new(0, 10, 4);
    let vectors = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let ids = vec![0, 1];
    page.append(&vectors, &ids);
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

/// Internal Page is used for multi layer indexing.
/// TODO: implement internal page
/// TODO: make page type erased.
#[derive(Debug)]
pub struct InternalPage {
    vector_unit_size: usize,
    max_vectors: usize,
    vector_data: Vec<u8>,
    page_ids: Vec<usize>,
    vector_ids: Vec<VecId>,
}

pub enum PageContent {
    LeafPage(LeafPage),
    InternalPage(InternalPage),
}

pub struct Page {
    page_id: usize,
    content: PageContent,
}

#[derive(Clone, Debug, Default)]
pub struct PageMetadata {
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
    free_page: Vec<usize>,
    /// page frame id to page frame metadata.
    metadata: HashMap<usize, PageMetadata>,
}

impl BufferPool {
    pub fn new(
        backed_file: PathBuf,
        pool_size: usize,
        page_size: usize,
        vector_unit_size: usize,
    ) -> BufferPool {
        let page_storage = Vec::with_capacity(pool_size);
        let free_page = (0..pool_size).collect();
        Self {
            backed_file,
            maximum_page_id: 0,
            page_size,
            vector_unit_size,
            page_table: HashMap::new(),
            page_storage,
            free_page,
            metadata: HashMap::new(),
        }
    }

    pub fn memory_usage(&self) -> usize {
        self.page_storage.iter().map(|p| p.memory_usage()).sum()
    }

    pub fn read_page(&self, page_id: usize) -> Result<LeafPage> {
        let file_size = self.backed_file.metadata()?.len();
        if file_size < (self.page_size * page_id).try_into().unwrap() {
            bail!("page id exceed disk file limit");
        }
        let mut f = File::open(&self.backed_file)?;
        f.seek(std::io::SeekFrom::Start(
            (page_id * self.page_size).try_into().unwrap(),
        ))?;
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
    }

    pub fn flush_page(&mut self, page_id: usize) {
        if let Some(frame_id) = self.page_table.get(&page_id) {
            self.flush_frame(*frame_id);
        } else {
            panic!("page not found");
        }
    }

    pub fn flush_all(&mut self) {
        for i in 0..self.page_storage.len() {
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
        if let Some(page_id) = self.free_page.pop() {
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
    pub fn fetch_page(&mut self, page_id: usize) -> &LeafPage {
        if let Some(page) = self.page_table.get(&page_id) {
            return &self.page_storage[*page];
        }
        if let Some(free_page_id) = self.free_page.pop() {
            let page = self.read_page(page_id).unwrap();
            self.page_storage[free_page_id] = page;
            self.page_table.insert(page_id, free_page_id);
            return &self.page_storage[free_page_id];
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
            self.metadata.insert(frame_id, PageMetadata::default());
            return frame_id;
        }
        panic!("no free page");
    }
}

#[test]
fn bufferpool_new() {
    let mut buffer_pool = BufferPool::new("test.bin".into(), 10, 4096, 4);
    assert_eq!(buffer_pool.page_size, 4096);
    assert_eq!(buffer_pool.vector_unit_size, 4);
    assert_eq!(buffer_pool.page_storage.capacity(), 10);
    assert_eq!(buffer_pool.free_page.len(), 10);
    buffer_pool.flush_all();
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
    const MAGIC_HEADER: u32 = 0xDEADBEEF;

    const METADATA_LENGTH: usize = size_of::<u32>() + size_of::<usize>() * 2;

    pub fn new(options: ShmIndexOptions) -> Result<Self> {
        let backed_file = options.backed_file;
        if backed_file.exists() {
            fs::remove_file(&backed_file)?;
        }
        let buffer_pool = BufferPool::new(
            backed_file,
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
        let mut header = [0u8; 4];
        f.read_exact(&mut header)?;
        let magic = u32::from_le_bytes(header);
        if magic != Self::MAGIC_HEADER {
            bail!("invalid file format");
        }
        let page_size = read_usize(&mut f)?;
        let vector_unit_size = read_usize(&mut f)?;

        let buffer_pool = BufferPool::new(file, pool_size, page_size, vector_unit_size);
        Ok(Self { buffer_pool })
    }

    pub fn persist(&mut self) -> Result<()> {
        let mut f = File::create(&self.buffer_pool.backed_file)?;
        f.write_all(&Self::MAGIC_HEADER.to_le_bytes())?;
        f.write_all(&(self.buffer_pool.page_size as u32).to_le_bytes())?;
        f.write_all(&(self.buffer_pool.vector_unit_size as u32).to_le_bytes())?;
        self.buffer_pool.flush_all();
        Ok(())
    }
    
}
