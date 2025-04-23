//!
//! The structure of a page:
//!
//! Common header
//! | page id (usize) | page type (u8) |
//!
//! Leaf page
//! | vector unit size (usize) | max vectors (usize) | current vectors (usize) |
//! | vector data (u8) | vector ids (usize) |
//!
//! Internal page
//! | vector unit size (usize) | max vectors (usize) | current vectors (usize) |
//! TODO: not parent page id. parent record ID or parent vec id?
//! | parent page id (usize) |
//! | vector data (u8) | vector ids (usize) | page ids (usize) |
//!
use anyhow::Result;
use std::marker::PhantomData;

/// VecId is the in-table key of vector index
type VecId = usize;

/// RecordID is the physical position of a record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordID {
    pub page_id: usize,
    pub slot_id: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageType {
    LeafPage,
    IndexPage,
}

impl PageType {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => PageType::LeafPage,
            1 => PageType::IndexPage,
            _ => panic!("Invalid page type"),
        }
    }

    pub fn to_u8(&self) -> u8 {
        match self {
            PageType::LeafPage => 0,
            PageType::IndexPage => 1,
        }
    }
}

/// Constants for page layout
const PAGE_ID_OFFSET: usize = 0;
const PAGE_ID_SIZE: usize = std::mem::size_of::<usize>();

const PAGE_TYPE_OFFSET: usize = PAGE_ID_SIZE;
const PAGE_TYPE_SIZE: usize = std::mem::size_of::<u8>();

const PAGE_COMMON_HEADER_SIZE: usize = PAGE_ID_SIZE + PAGE_TYPE_SIZE;

/// Leaf page begin
const LEAF_VECTOR_UNIT_SIZE_OFFSET: usize = PAGE_COMMON_HEADER_SIZE;
const LEAF_VECTOR_UNIT_SIZE_SIZE: usize = std::mem::size_of::<usize>();

const LEAF_MAX_VECTORS_OFFSET: usize = LEAF_VECTOR_UNIT_SIZE_OFFSET + LEAF_VECTOR_UNIT_SIZE_SIZE;
const LEAF_MAX_VECTORS_SIZE: usize = std::mem::size_of::<usize>();

const LEAF_CURRENT_VECTORS_OFFSET: usize = LEAF_MAX_VECTORS_OFFSET + LEAF_MAX_VECTORS_SIZE;
const LEAF_CURRENT_VECTORS_SIZE: usize = std::mem::size_of::<usize>();

const LEAF_HEADER_SIZE: usize = PAGE_COMMON_HEADER_SIZE
    + LEAF_VECTOR_UNIT_SIZE_SIZE
    + LEAF_MAX_VECTORS_SIZE
    + LEAF_CURRENT_VECTORS_SIZE;

const LEAF_DATA_OFFSET: usize = LEAF_HEADER_SIZE;

fn get_leaf_vector_ids_offset(max_vectors: usize, vector_unit_size: usize) -> usize {
    LEAF_DATA_OFFSET + max_vectors * vector_unit_size
}

pub fn calculate_max_vectors_in_leaf_page(page_size: usize, vector_unit_size: usize) -> usize {
    (page_size - LEAF_HEADER_SIZE) / (vector_unit_size + std::mem::size_of::<VecId>())
}

/// Internal page begin
const INTERNAL_VECTOR_UNIT_SIZE_OFFSET: usize = PAGE_COMMON_HEADER_SIZE;
const INTERNAL_VECTOR_UNIT_SIZE_SIZE: usize = std::mem::size_of::<usize>();

const INTERNAL_MAX_VECTORS_OFFSET: usize =
    INTERNAL_VECTOR_UNIT_SIZE_OFFSET + INTERNAL_VECTOR_UNIT_SIZE_SIZE;
const INTERNAL_MAX_VECTORS_SIZE: usize = std::mem::size_of::<usize>();

const INTERNAL_CURRENT_VECTORS_OFFSET: usize =
    INTERNAL_MAX_VECTORS_OFFSET + INTERNAL_MAX_VECTORS_SIZE;
const INTERNAL_CURRENT_VECTORS_SIZE: usize = std::mem::size_of::<usize>();

const INTERNAL_PARENT_PAGE_ID_OFFSET: usize =
    INTERNAL_CURRENT_VECTORS_OFFSET + INTERNAL_CURRENT_VECTORS_SIZE;
const INTERNAL_PARENT_PAGE_ID_SIZE: usize = std::mem::size_of::<usize>();

const INTERNAL_HEADER_SIZE: usize = PAGE_COMMON_HEADER_SIZE
    + INTERNAL_VECTOR_UNIT_SIZE_SIZE
    + INTERNAL_MAX_VECTORS_SIZE
    + INTERNAL_CURRENT_VECTORS_SIZE
    + INTERNAL_PARENT_PAGE_ID_SIZE;

const INTERNAL_DATA_OFFSET: usize = INTERNAL_HEADER_SIZE;

fn get_internal_vector_ids_offset(max_vectors: usize, vector_unit_size: usize) -> usize {
    INTERNAL_DATA_OFFSET + max_vectors * vector_unit_size
}

fn get_internal_child_page_ids_offset(max_vectors: usize, vector_unit_size: usize) -> usize {
    get_internal_vector_ids_offset(max_vectors, vector_unit_size) + max_vectors * size_of::<VecId>()
}

pub fn calculate_max_vectors_in_internal_page(page_size: usize, vector_unit_size: usize) -> usize {
    (page_size - INTERNAL_HEADER_SIZE) / (vector_unit_size + std::mem::size_of::<VecId>())
}

pub trait PageDescriptor {}

pub struct LeafPageDescriptor;
pub struct InternalPageDescriptor;

impl PageDescriptor for LeafPageDescriptor {}
impl PageDescriptor for InternalPageDescriptor {}

pub struct PageAccessor<T: PageDescriptor> {
    page_data: *mut u8,
    page_size: usize,
    marker: PhantomData<T>,
}

impl<T: PageDescriptor> PageAccessor<T> {
    fn read_u8(&self, offset: usize) -> u8 {
        assert!(offset < self.page_size);
        // SAFETY: PageReader is within the bufferpool.
        unsafe {
            let byte_offset = self.page_data.add(offset);
            *byte_offset
        }
    }

    fn read_usize(&self, offset: usize) -> usize {
        assert!(offset + PAGE_ID_SIZE <= self.page_size);
        // SAFETY: PageReader is within the bufferpool.
        unsafe {
            let byte_offset = self.page_data.add(offset);
            let mut buffer = [0u8; std::mem::size_of::<usize>()];
            buffer.copy_from_slice(std::slice::from_raw_parts(byte_offset, PAGE_ID_SIZE));
            usize::from_le_bytes(buffer)
        }
    }

    fn read_slice(&self, offset: usize, length: usize) -> &[u8] {
        assert!(offset + length <= self.page_size);
        // SAFETY: PageReader is within the bufferpool.
        unsafe {
            let byte_offset = self.page_data.add(offset);
            std::slice::from_raw_parts(byte_offset, length)
        }
    }

    fn write_u8(&mut self, offset: usize, value: u8) -> Result<()> {
        assert!(offset < self.page_size);
        // SAFETY: PageReader is within the bufferpool.
        unsafe {
            let byte_offset = self.page_data.add(offset);
            *byte_offset = value;
        }
        Ok(())
    }

    fn write_usize(&mut self, offset: usize, value: usize) -> Result<()> {
        // FIXME: why 8192 ????
        assert!(offset + PAGE_ID_SIZE <= self.page_size);
        // SAFETY: PageReader is within the bufferpool.
        unsafe {
            let byte_offset = self.page_data.add(offset);
            let bytes = value.to_le_bytes();
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), byte_offset, PAGE_ID_SIZE);
        }
        Ok(())
    }

    fn write_slice(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        assert!(offset + data.len() <= self.page_size);
        // SAFETY: PageReader is within the bufferpool.
        unsafe {
            let byte_offset = self.page_data.add(offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), byte_offset, data.len());
        }
        Ok(())
    }

    pub fn read_page_id(&self) -> usize {
        self.read_usize(PAGE_ID_OFFSET)
    }

    pub fn write_page_id(&mut self, page_id: usize) -> Result<()> {
        self.write_usize(PAGE_ID_OFFSET, page_id)
    }

    pub fn read_page_type(&self) -> PageType {
        let page_type = self.read_u8(PAGE_TYPE_OFFSET);
        PageType::from_u8(page_type)
    }

    pub fn write_page_type(&mut self, page_type: PageType) -> Result<()> {
        self.write_u8(PAGE_TYPE_OFFSET, page_type.to_u8())
    }

    pub fn print_common_header(&self) {
        let page_id = self.read_page_id();
        let page_type = self.read_page_type();
        println!("Page ID: {}, Page Type: {:?}", page_id, page_type);
    }
}

pub fn read_page_id(page: &[u8]) -> usize {
    assert!(page.len() >= PAGE_ID_SIZE);
    let page_id = &page[PAGE_ID_OFFSET..PAGE_ID_OFFSET + PAGE_ID_SIZE];
    usize::from_le_bytes(page_id.try_into().unwrap())
}

pub fn read_page_type(page: &[u8]) -> PageType {
    assert!(page.len() >= PAGE_TYPE_SIZE);
    let page_type = page[PAGE_TYPE_OFFSET];
    PageType::from_u8(page_type)
}

impl PageAccessor<LeafPageDescriptor> {
    pub fn read_vector_unit_size(&self) -> usize {
        self.read_usize(LEAF_VECTOR_UNIT_SIZE_OFFSET)
    }

    pub fn write_vector_unit_size(&mut self, vector_unit_size: usize) -> Result<()> {
        self.write_usize(LEAF_VECTOR_UNIT_SIZE_OFFSET, vector_unit_size)
    }

    pub fn read_max_vectors(&self) -> usize {
        self.read_usize(LEAF_MAX_VECTORS_OFFSET)
    }

    pub fn write_max_vectors(&mut self, max_vectors: usize) -> Result<()> {
        self.write_usize(LEAF_MAX_VECTORS_OFFSET, max_vectors)
    }

    pub fn read_current_vectors(&self) -> usize {
        self.read_usize(LEAF_CURRENT_VECTORS_OFFSET)
    }

    pub fn write_current_vectors(&mut self, current_vectors: usize) -> Result<()> {
        self.write_usize(LEAF_CURRENT_VECTORS_OFFSET, current_vectors)
    }

    pub fn read_vector_data(&self, idx: usize) -> &[u8] {
        let vector_unit_size = self.read_vector_unit_size();
        let offset = LEAF_DATA_OFFSET + idx * vector_unit_size;
        self.read_slice(offset, vector_unit_size)
    }

    pub fn write_vector_data(&mut self, idx: usize, data: &[u8]) -> Result<()> {
        assert!(idx < self.read_max_vectors());
        assert!(data.len() == self.read_vector_unit_size());
        let vector_unit_size = self.read_vector_unit_size();
        let offset = LEAF_DATA_OFFSET + idx * vector_unit_size;
        self.write_slice(offset, data)
    }

    pub fn read_vector_id(&self, idx: usize) -> VecId {
        let offset =
            get_leaf_vector_ids_offset(self.read_max_vectors(), self.read_vector_unit_size());
        let id_offset = offset + idx * size_of::<VecId>();
        self.read_usize(id_offset)
    }

    pub fn write_vector_id(&mut self, idx: usize, id: VecId) -> Result<()> {
        assert!(idx < self.read_max_vectors());
        let offset =
            get_leaf_vector_ids_offset(self.read_max_vectors(), self.read_vector_unit_size());
        let id_offset = offset + idx * size_of::<VecId>();
        self.write_usize(id_offset, id)
    }

    pub fn read_record(&self, idx: usize) -> (VecId, &[u8]) {
        let vector = self.read_vector_data(idx);
        let id = self.read_vector_id(idx);

        (id, vector)
    }

    pub fn iter(&self) -> impl Iterator<Item = (VecId, &[u8])> {
        (0..self.read_current_vectors()).map(move |idx| self.read_record(idx))
    }

    pub fn write_record(&mut self, idx: usize, id: VecId, data: &[u8]) -> Result<()> {
        assert!(idx < self.read_max_vectors());
        assert!(data.len() == self.read_vector_unit_size());
        self.write_vector_id(idx, id)?;
        self.write_vector_data(idx, data)
    }

    pub fn append_record(&mut self, id: VecId, data: &[u8]) -> Result<()> {
        let current_vectors = self.read_current_vectors();
        assert!(current_vectors < self.read_max_vectors());
        self.write_record(current_vectors, id, data)?;
        self.write_current_vectors(current_vectors + 1)
    }

    pub fn print_leaf_page_info(&self) {
        self.print_common_header();
        let vector_unit_size = self.read_vector_unit_size();
        let max_vectors = self.read_max_vectors();
        let current_vectors = self.read_current_vectors();
        println!(
            "Leaf Page: Vector Unit Size: {}, Max Vectors: {}, Current Vectors: {}",
            vector_unit_size, max_vectors, current_vectors
        );
    }
}

impl PageAccessor<InternalPageDescriptor> {
    pub fn read_vector_unit_size(&self) -> usize {
        self.read_usize(INTERNAL_VECTOR_UNIT_SIZE_OFFSET)
    }

    pub fn write_vector_unit_size(&mut self, vector_unit_size: usize) -> Result<()> {
        self.write_usize(INTERNAL_VECTOR_UNIT_SIZE_OFFSET, vector_unit_size)
    }

    pub fn read_max_vectors(&self) -> usize {
        self.read_usize(INTERNAL_MAX_VECTORS_OFFSET)
    }

    pub fn write_max_vectors(&mut self, max_vectors: usize) -> Result<()> {
        self.write_usize(INTERNAL_MAX_VECTORS_OFFSET, max_vectors)
    }

    pub fn read_current_vectors(&self) -> usize {
        self.read_usize(INTERNAL_CURRENT_VECTORS_OFFSET)
    }

    pub fn write_current_vectors(&mut self, current_vectors: usize) -> Result<()> {
        self.write_usize(INTERNAL_CURRENT_VECTORS_OFFSET, current_vectors)
    }

    pub fn read_parent_page_id(&self) -> usize {
        self.read_usize(INTERNAL_PARENT_PAGE_ID_OFFSET)
    }

    pub fn write_parent_page_id(&mut self, parent_page_id: usize) -> Result<()> {
        self.write_usize(INTERNAL_PARENT_PAGE_ID_OFFSET, parent_page_id)
    }

    pub fn read_vector_data(&self, idx: usize) -> &[u8] {
        let vector_unit_size = self.read_vector_unit_size();
        let offset = INTERNAL_DATA_OFFSET + idx * vector_unit_size;
        self.read_slice(offset, vector_unit_size)
    }

    pub fn write_vector_data(&mut self, idx: usize, data: &[u8]) -> Result<()> {
        assert!(idx < self.read_max_vectors());
        assert!(data.len() == self.read_vector_unit_size());
        let vector_unit_size = self.read_vector_unit_size();
        let offset = INTERNAL_DATA_OFFSET + idx * vector_unit_size;
        self.write_slice(offset, data)
    }

    pub fn read_vector_id(&self, idx: usize) -> VecId {
        let offset =
            get_internal_vector_ids_offset(self.read_max_vectors(), self.read_vector_unit_size());
        let id_offset = offset + idx * size_of::<VecId>();
        self.read_usize(id_offset)
    }

    pub fn write_vector_id(&mut self, idx: usize, id: VecId) -> Result<()> {
        assert!(idx < self.read_max_vectors());
        let offset =
            get_internal_vector_ids_offset(self.read_max_vectors(), self.read_vector_unit_size());
        let id_offset = offset + idx * size_of::<VecId>();
        self.write_usize(id_offset, id)
    }

    pub fn read_child_page_id(&self, idx: usize) -> usize {
        let offset = get_internal_child_page_ids_offset(
            self.read_max_vectors(),
            self.read_vector_unit_size(),
        );
        let id_offset = offset + idx * size_of::<VecId>();
        self.read_usize(id_offset)
    }

    pub fn write_child_page_id(&mut self, idx: usize, id: usize) -> Result<()> {
        assert!(idx < self.read_max_vectors());
        let offset = get_internal_child_page_ids_offset(
            self.read_max_vectors(),
            self.read_vector_unit_size(),
        );
        let id_offset = offset + idx * size_of::<VecId>();
        self.write_usize(id_offset, id)
    }

    pub fn read_record(&self, idx: usize) -> (VecId, &[u8], usize) {
        let vector = self.read_vector_data(idx);
        let id = self.read_vector_id(idx);
        let page_id = self.read_child_page_id(idx);

        (id, vector, page_id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (VecId, &[u8], usize)> {
        (0..self.read_current_vectors()).map(move |idx| self.read_record(idx))
    }

    pub fn write_record(
        &mut self,
        idx: usize,
        id: VecId,
        data: &[u8],
        page_id: usize,
    ) -> Result<()> {
        assert!(idx < self.read_max_vectors());
        assert_eq!(data.len(), self.read_vector_unit_size());
        self.write_vector_id(idx, id)?;
        self.write_vector_data(idx, data)?;
        self.write_child_page_id(idx, page_id)
    }

    pub fn append_record(&mut self, id: VecId, data: &[u8], page_id: usize) -> Result<()> {
        let current_vectors = self.read_current_vectors();
        assert!(current_vectors < self.read_max_vectors());
        self.write_record(current_vectors, id, data, page_id)?;
        self.write_current_vectors(current_vectors + 1)
    }

    pub fn print_internal_page_info(&self) {
        self.print_common_header();
        let vector_unit_size = self.read_vector_unit_size();
        let max_vectors = self.read_max_vectors();
        let current_vectors = self.read_current_vectors();
        let parent_page_id = self.read_parent_page_id();
        println!(
            "Internal Page: Vector Unit Size: {}, Max Vectors: {}, Current Vectors: {}, Parent Page ID: {}",
            vector_unit_size, max_vectors, current_vectors, parent_page_id
        );
    }
}

pub fn get_leaf_reader_from_buffer(page: &[u8]) -> PageAccessor<LeafPageDescriptor> {
    let page_reader = PageAccessor::<LeafPageDescriptor> {
        page_data: page.as_ptr() as *mut u8,
        page_size: page.len(),
        marker: PhantomData,
    };
    assert_eq!(page_reader.read_page_type(), PageType::LeafPage);
    page_reader
}

pub fn create_leaf_page_from_buffer(page: &[u8], page_id: usize, vector_unit_size: usize) {
    let mut page_reader = PageAccessor::<LeafPageDescriptor> {
        page_data: page.as_ptr() as *mut u8,
        page_size: page.len(),
        marker: PhantomData,
    };
    page_reader.write_page_id(page_id).unwrap();
    page_reader.write_page_type(PageType::LeafPage).unwrap();
    page_reader
        .write_vector_unit_size(vector_unit_size)
        .unwrap();

    let max_vectors = calculate_max_vectors_in_leaf_page(page.len(), vector_unit_size);
    page_reader.write_max_vectors(max_vectors).unwrap();

    page_reader.write_current_vectors(0).unwrap();
}

pub fn get_internal_reader_from_buffer(page: &[u8]) -> PageAccessor<InternalPageDescriptor> {
    let page_reader = PageAccessor::<InternalPageDescriptor> {
        page_data: page.as_ptr() as *mut u8,
        page_size: page.len(),
        marker: PhantomData,
    };
    assert_eq!(page_reader.read_page_type(), PageType::IndexPage);
    page_reader
}

pub fn create_internal_page_from_buffer(
    page: &[u8],
    page_id: usize,
    vector_unit_size: usize,
    parent_page_id: usize,
) {
    let mut page_reader = PageAccessor::<InternalPageDescriptor> {
        page_data: page.as_ptr() as *mut u8,
        page_size: page.len(),
        marker: PhantomData,
    };
    page_reader.write_page_id(page_id).unwrap();
    page_reader.write_page_type(PageType::IndexPage).unwrap();
    page_reader
        .write_vector_unit_size(vector_unit_size)
        .unwrap();

    let max_vectors = calculate_max_vectors_in_internal_page(page.len(), vector_unit_size);
    page_reader.write_max_vectors(max_vectors).unwrap();

    page_reader.write_current_vectors(0).unwrap();

    page_reader.write_parent_page_id(parent_page_id).unwrap();
}

pub fn print_page_information(page: &[u8]) {
    let page_type = PageType::from_u8(page[PAGE_TYPE_OFFSET]);
    match page_type {
        PageType::LeafPage => {
            let leaf_reader = get_leaf_reader_from_buffer(page);
            leaf_reader.print_leaf_page_info();
        }
        PageType::IndexPage => {
            let internal_reader = get_internal_reader_from_buffer(page);
            internal_reader.print_internal_page_info();
        }
    }
}

#[test]
fn test_page_reader() {
    let page_size = 1024;
    let page_data = vec![0u8; page_size];
    print_page_information(&page_data);
}
