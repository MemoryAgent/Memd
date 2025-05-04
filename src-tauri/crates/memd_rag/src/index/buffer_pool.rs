use anyhow::Result;

use std::collections::HashMap;

use crate::index::page::{create_internal_page_from_buffer, create_leaf_page_from_buffer};

use super::{index_file::IndexFile, page::read_page_id};

#[derive(Clone, Debug, Default)]
pub struct FrameMetadata {
    /// How often the frame is accessed.
    temperature: f32,
    /// How many times the frame is pinned. pinned frame will not be evicted.
    pin_count: i32,
    /// The page is dirty, it needs to be flushed to disk.
    dirty: bool,
}

/// Buffer pool swaps pages between memory and disk.
pub struct BufferPool {
    /// Persistent file.
    backed_file: IndexFile,
    /// Page size is fixed for whole buffer pool.
    page_size: usize,
    vector_unit_size: usize,
    /// Memory page storage.
    frame_storage: Vec<u8>,
    /// page id to frame index.
    page_table: HashMap<usize, usize>,
    /// list of free page frame ids.
    free_frames: Vec<usize>,
    /// frame id to frame metadata.
    metadata: HashMap<usize, FrameMetadata>,
}

impl BufferPool {
    pub fn new(
        backed_file: IndexFile,
        pool_size: usize,
        page_size: usize,
        vector_unit_size: usize,
    ) -> BufferPool {
        let page_storage = vec![0; pool_size * page_size];

        let free_frames = (0..pool_size).collect();

        Self {
            backed_file,
            page_size,
            vector_unit_size,
            page_table: HashMap::new(),
            frame_storage: page_storage,
            free_frames,
            metadata: HashMap::new(),
        }
    }

    pub fn memory_usage(&self) -> usize {
        self.frame_storage.len()
    }

    pub fn read_page(&mut self, page_id: usize, buf: &mut [u8]) -> Result<()> {
        self.backed_file.read_page(page_id, buf)
    }

    pub fn pin_page(&mut self, page_id: usize) {
        if let Some(frame) = self.page_table.get(&page_id) {
            let metadata = self.metadata.get_mut(frame).unwrap();
            metadata.pin_count += 1;
            metadata.temperature += 1.0;
        } else {
            panic!("page not found");
        }
    }

    pub fn unpin_page(&mut self, page_id: usize) {
        if let Some(frame) = self.page_table.get(&page_id) {
            let metadata = self.metadata.get_mut(frame).unwrap();
            assert!(metadata.pin_count > 0);
            metadata.pin_count -= 1;
        } else {
            panic!("page not found");
        }
    }

    pub fn mark_dirty(&mut self, page_id: usize) {
        if let Some(frame) = self.page_table.get(&page_id) {
            self.metadata.get_mut(frame).unwrap().dirty = true;
        } else {
            panic!("page not found");
        }
    }

    fn get_ith_frame(&self, frame_id: usize) -> &[u8] {
        let offset = frame_id * self.page_size;
        &self.frame_storage[offset..offset + self.page_size]
    }

    pub fn flush_frame(&mut self, frame_id: usize) {
        let metadata = self.metadata.get_mut(&frame_id).unwrap();
        if !metadata.dirty {
            return;
        }
        metadata.dirty = false;
        let f = &mut self.backed_file;
        let offset = frame_id * self.page_size;
        let frame = &self.frame_storage[offset..offset + self.page_size];
        f.write_page(frame).unwrap();
    }

    pub fn flush_page(&mut self, page_id: usize) {
        if let Some(frame_id) = self.page_table.get(&page_id) {
            self.flush_frame(*frame_id);
        } else {
            panic!("page not found");
        }
    }

    pub fn flush_all(&mut self) {
        let all_frames: Vec<usize> = self.page_table.values().map(|x| *x).collect();
        for frame_id in all_frames {
            self.flush_frame(frame_id);
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
        candidates.sort_by(|(_, a), (_, b)| a.temperature.total_cmp(&b.temperature));

        let (frame_id, _) = candidates[0];
        Some(*frame_id)
    }

    // before evicting a page, we need to cleanup its old data
    fn cleanup_frame(&mut self, frame_id: usize) {
        self.flush_frame(frame_id);

        let frame_data = {
            let offset = frame_id * self.page_size;
            &mut self.frame_storage[offset..offset + self.page_size]
        };
        let page_id = read_page_id(frame_data);

        frame_data.fill(0);
        self.page_table.remove(&page_id);
        self.metadata.remove(&frame_id);
    }

    pub fn claim_free_frame(&mut self) -> Option<usize> {
        if let Some(frame_id) = self.free_frames.pop() {
            return Some(frame_id);
        }
        if let Some(victim_frame) = self.get_evict_victim() {
            self.cleanup_frame(victim_frame);
            return Some(victim_frame);
        }
        None
    }
}

pub struct PageGuard {
    page_id: usize,
    page_frame_id: usize,
    buffer_pool: *mut BufferPool,
}

impl PageGuard {
    pub fn new(page_id: usize, page_frame_id: usize, buffer_pool: *mut BufferPool) -> PageGuard {
        // SAFETY: single thread, buffer pool live longer than this method.
        unsafe {
            buffer_pool.as_mut().unwrap().pin_page(page_id);
        }
        PageGuard {
            page_id,
            page_frame_id,
            buffer_pool,
        }
    }
}

impl Drop for PageGuard {
    fn drop(&mut self) {
        // SAFETY: single thread, buffer pool live longer than page guard.
        unsafe {
            self.buffer_pool.as_mut().unwrap().unpin_page(self.page_id);
        }
    }
}

impl PageGuard {
    pub fn get_page(&self) -> &[u8] {
        // SAFETY: single thread, buffer pool live longer than page guard.
        unsafe {
            self.buffer_pool
                .as_mut()
                .unwrap()
                .get_ith_frame(self.page_frame_id)
        }
    }
}

impl BufferPool {
    fn prepare_page(&mut self, page_id: usize, frame_id: usize) {
        self.page_table.insert(page_id, frame_id);
        self.metadata.insert(frame_id, FrameMetadata::default());
    }

    pub fn fetch_page(&mut self, page_id: usize) -> PageGuard {
        if let Some(frame_id) = self.page_table.get(&page_id) {
            return PageGuard::new(page_id, *frame_id, self);
        }
        if let Some(free_frame_id) = self.claim_free_frame() {
            let offset = free_frame_id * self.page_size;
            let mut frame = &mut self.frame_storage[offset..offset + self.page_size];
            self.backed_file.read_page(page_id, &mut frame).unwrap();
            self.prepare_page(page_id, free_frame_id);
            return PageGuard::new(page_id, free_frame_id, self);
        }
        panic!("no free page");
    }

    pub fn create_leaf_page(&mut self) -> usize {
        if let Some(frame_id) = self.claim_free_frame() {
            let page_id = self.backed_file.create_page().unwrap();
            let frame =
                &mut self.frame_storage[frame_id * self.page_size..(frame_id + 1) * self.page_size];

            create_leaf_page_from_buffer(&frame, page_id, self.vector_unit_size);

            self.prepare_page(page_id, frame_id);
            self.mark_dirty(page_id);
            return page_id;
        }
        panic!("no free page");
    }

    pub fn create_internal_page(&mut self, parent: usize) -> usize {
        if let Some(frame_id) = self.claim_free_frame() {
            let page_id = self.backed_file.create_page().unwrap();
            let frame =
                &mut self.frame_storage[frame_id * self.page_size..(frame_id + 1) * self.page_size];

            create_internal_page_from_buffer(&frame, page_id, self.vector_unit_size, parent);

            self.prepare_page(page_id, frame_id);
            self.mark_dirty(page_id);
            return page_id;
        }
        panic!("no free page");
    }

    pub fn print_status(&self) {
        println!("Buffer pool status:");
        println!("Page size: {}", self.page_size);
        println!("Vector unit size: {}", self.vector_unit_size);
        println!("Memory usage: {}", self.memory_usage());
        println!("Free frames: {:?}", self.free_frames);
        // line separate page table
        println!("Page table:");

        for (page_id, frame_id) in &self.page_table {
            println!("    {} -> {}", page_id, frame_id);
        }

        println!("Metadata:");

        for (frame_id, metadata) in &self.metadata {
            println!(
                "    Frame {}: temperature: {}, pin count: {}, dirty: {}",
                frame_id, metadata.temperature, metadata.pin_count, metadata.dirty
            );
        }

        println!("");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::remove_file;

    #[test]
    fn bufferpool_new() {
        let backed_file = IndexFile::create(
            "test.bin",
            4096,
            4,
            crate::index::executor::SummaryMethod::GMMCentroid,
        )
        .unwrap();
        let mut buffer_pool = BufferPool::new(backed_file, 10, 4096, 4);
        assert_eq!(buffer_pool.page_size, 4096);
        assert_eq!(buffer_pool.vector_unit_size, 4);
        assert_eq!(buffer_pool.frame_storage.capacity(), 10 * 4096);
        assert_eq!(buffer_pool.free_frames.len(), 10);
        buffer_pool.flush_all();
    }

    #[test]
    fn buffer_pool_print_status_with_pages() {
        let backed_file = IndexFile::create(
            "test.bin",
            4096,
            4,
            crate::index::executor::SummaryMethod::GMMCentroid,
        )
        .unwrap();
        let mut buffer_pool = BufferPool::new(backed_file, 10, 4096, 4);
        let _l = buffer_pool.create_leaf_page();
        buffer_pool.print_status();
        buffer_pool.flush_all();
        remove_file("test.bin").unwrap();
    }

    #[test]
    fn buffer_pool_raii_status() {
        let backed_file = IndexFile::create(
            "test.bin",
            4096,
            4,
            crate::index::executor::SummaryMethod::GMMCentroid,
        )
        .unwrap();
        let mut buffer_pool = BufferPool::new(backed_file, 10, 4096, 4);
        let page_id = buffer_pool.create_leaf_page();
        {
            let _guard = buffer_pool.fetch_page(page_id);
            buffer_pool.print_status();
        }
        {
            let _guard = buffer_pool.fetch_page(page_id);
            buffer_pool.print_status();
        }
        {
            let _guard = buffer_pool.fetch_page(page_id);
            let _guard2 = buffer_pool.fetch_page(page_id);
            buffer_pool.print_status();
        }
        buffer_pool.flush_all();
        buffer_pool.print_status();
        remove_file("test.bin").unwrap();
    }

    #[test]
    fn buffer_pool_evict() {
        let backed_file = IndexFile::create(
            "test.bin",
            4096,
            4,
            crate::index::executor::SummaryMethod::GMMCentroid,
        )
        .unwrap();
        let mut buffer_pool = BufferPool::new(backed_file, 10, 4096, 4);
        for _ in 0..10 {
            buffer_pool.create_leaf_page();
        }

        let _guards = (1..10)
            .map(|id| buffer_pool.fetch_page(id))
            .collect::<Vec<_>>();

        buffer_pool.print_status();

        let _guard = buffer_pool.create_internal_page(0);

        buffer_pool.print_status();

        let _guard2 = buffer_pool.fetch_page(0);

        buffer_pool.print_status();

        remove_file("test.bin").unwrap();
    }
}
