//! Index file is the persistent storage for the index.
//!
//! Format:
//!
//! MAGIC_HEADER (u64)
//!
//! Page size (usize)
//!
//! Vector unit size (usize)
//!
//! Maximum page id (usize)
//!
//! TODO: these two methods can use u8 to store, but the priority is low.
//!
//! Summary method (usize)
//!
//! Cluster method (usize)

// constants

use anyhow::{bail, Result};
use candle_transformers::models::bert::BertModel;
use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    sync::{Arc, Mutex},
};
use tokenizers::Tokenizer;

use crate::component::llm::LLM;

use super::{
    executor::{ClusterMethod, SummaryMethod},
    page::read_page_id,
};

const MAGIC_HEADER: u64 = 0xDEADBEEFDEADBEEF;
const MAGIC_HEADER_OFFSET: usize = 0;
const MAGIC_HEADER_SIZE: usize = size_of_val(&MAGIC_HEADER);

const PAGE_SIZE_OFFSET: usize = MAGIC_HEADER_SIZE;
const PAGE_SIZE_SIZE: usize = size_of::<usize>();

const VECTOR_UNIT_SIZE_OFFSET: usize = PAGE_SIZE_OFFSET + PAGE_SIZE_SIZE;
const VECTOR_UNIT_SIZE_SIZE: usize = size_of::<usize>();

const MAX_PAGE_ID_OFFSET: usize = VECTOR_UNIT_SIZE_OFFSET + VECTOR_UNIT_SIZE_SIZE;
const MAX_PAGE_ID_SIZE: usize = size_of::<usize>();

const SUMMARY_METHOD_OFFSET: usize = MAX_PAGE_ID_OFFSET + MAX_PAGE_ID_SIZE;
const SUMMARY_METHOD_SIZE: usize = size_of::<usize>();

const CLUSTER_METHOD_OFFSET: usize = SUMMARY_METHOD_OFFSET + SUMMARY_METHOD_SIZE;
const CLUSTER_METHOD_SIZE: usize = size_of::<usize>();

const DATA_OFFSET: usize = CLUSTER_METHOD_OFFSET + CLUSTER_METHOD_SIZE;

pub struct IndexFile {
    // we don't need buffered writer because writes are batched to pages.
    file: File,
    // 4k to 32k
    pub page_size: usize,
    pub vector_unit_size: usize,
    // the next page id to be allocated.
    pub max_page_id: usize,
    // summary method is a rich enum...
    pub summary_method: usize,
    pub cluster_method: ClusterMethod,
}

fn write_u64(file: &mut File, offset: usize, value: u64) -> Result<()> {
    file.seek(SeekFrom::Start(offset as u64))?;
    file.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_usize(file: &mut File, offset: usize, value: usize) -> Result<()> {
    file.seek(SeekFrom::Start(offset as u64))?;
    file.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_slice(file: &mut File, offset: usize, data: &[u8]) -> Result<()> {
    file.seek(SeekFrom::Start(offset as u64))?;
    file.write_all(data)?;
    Ok(())
}

fn summary_method_to_usize(method: &SummaryMethod) -> usize {
    match method {
        SummaryMethod::LLM { .. } => 0,
        SummaryMethod::Centroid => 1,
    }
}

pub fn usize_to_summary_method(
    value: usize,
    llm: Option<Arc<Mutex<dyn LLM>>>,
    tokenizer: Option<Arc<Mutex<Tokenizer>>>,
    bert: Option<Arc<Mutex<BertModel>>>,
) -> Result<SummaryMethod> {
    match value {
        0 => Ok(SummaryMethod::LLM {
            llm: llm.unwrap().clone(),
            tokenizer: tokenizer.unwrap().clone(),
            bert: bert.unwrap().clone(),
        }),
        1 => Ok(SummaryMethod::Centroid),
        _ => bail!("Invalid summary method value: {}", value),
    }
}

fn cluster_method_to_u64(method: ClusterMethod) -> u64 {
    match method {
        ClusterMethod::GMM => 0,
        ClusterMethod::KMeans => 1,
        ClusterMethod::NoCluster => 2,
    }
}

fn u64_to_cluster_method(value: u64) -> Result<ClusterMethod> {
    match value {
        0 => Ok(ClusterMethod::GMM),
        1 => Ok(ClusterMethod::KMeans),
        2 => Ok(ClusterMethod::NoCluster),
        _ => bail!("Invalid clustering method value: {}", value),
    }
}

impl IndexFile {
    fn write_header(&mut self) -> Result<()> {
        write_u64(&mut self.file, 0, MAGIC_HEADER)?;
        write_usize(&mut self.file, PAGE_SIZE_OFFSET, self.page_size)?;
        write_usize(
            &mut self.file,
            VECTOR_UNIT_SIZE_OFFSET,
            self.vector_unit_size,
        )?;
        write_usize(&mut self.file, MAX_PAGE_ID_OFFSET, self.max_page_id)?;
        write_usize(&mut self.file, SUMMARY_METHOD_OFFSET, self.summary_method)?;
        write_u64(
            &mut self.file,
            CLUSTER_METHOD_OFFSET,
            cluster_method_to_u64(self.cluster_method),
        )?;
        Ok(())
    }

    pub fn write_page(&mut self, data: &[u8]) -> Result<()> {
        let page_id = read_page_id(data);
        if page_id > self.max_page_id {
            bail!("Page id {} is out of range", page_id);
        }
        let offset = DATA_OFFSET + page_id * self.page_size;
        write_slice(&mut self.file, offset, data)?;
        Ok(())
    }

    pub fn create(
        path: impl AsRef<Path>,
        page_size: usize,
        vector_unit_size: usize,
        summary_method: &SummaryMethod,
        cluster_method: ClusterMethod,
    ) -> Result<Self> {
        let file = File::create_new(path)?;
        let mut index_file = IndexFile {
            file,
            page_size,
            vector_unit_size,
            max_page_id: 0,
            summary_method: summary_method_to_usize(summary_method),
            cluster_method,
        };
        index_file.write_header()?;
        Ok(index_file)
    }

    fn verify_header(file: &mut File) -> Result<()> {
        let mut buf = [0; MAGIC_HEADER_SIZE];
        file.seek(SeekFrom::Start(MAGIC_HEADER_OFFSET as u64))?;
        file.read_exact(&mut buf)?;
        let magic_header = u64::from_le_bytes(buf);
        if magic_header != MAGIC_HEADER {
            bail!("Invalid index file header");
        }
        Ok(())
    }

    fn read_usize(file: &mut File, offset: usize) -> Result<usize> {
        let mut buf = [0; PAGE_SIZE_SIZE];
        file.seek(SeekFrom::Start(offset as u64))?;
        file.read_exact(&mut buf)?;
        Ok(usize::from_le_bytes(buf))
    }

    fn read_u64(file: &mut File, offset: usize) -> Result<u64> {
        let mut buf = [0; PAGE_SIZE_SIZE];
        file.seek(SeekFrom::Start(offset as u64))?;
        file.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = OpenOptions::new().write(true).read(true).open(path)?;
        Self::verify_header(&mut file)?;
        let page_size = Self::read_usize(&mut file, PAGE_SIZE_OFFSET)?;
        let vector_unit_size = Self::read_usize(&mut file, VECTOR_UNIT_SIZE_OFFSET)?;
        let max_page_id = Self::read_usize(&mut file, MAX_PAGE_ID_OFFSET)?;
        let summary_method = Self::read_usize(&mut file, SUMMARY_METHOD_OFFSET)?;
        let cluster_method =
            u64_to_cluster_method(Self::read_u64(&mut file, CLUSTER_METHOD_OFFSET)?)?;

        Ok(IndexFile {
            file,
            page_size,
            vector_unit_size,
            max_page_id,
            summary_method,
            cluster_method,
        })
    }

    pub fn print_metadata(&self) {
        println!("Page size: {}", self.page_size);
        println!("Vector unit size: {}", self.vector_unit_size);
        println!("Max page id: {}", self.max_page_id);
        println!("Summary method: {:?}", self.summary_method);
        println!("Cluster method: {:?}", self.cluster_method);
    }

    pub fn read_page(&mut self, page_id: usize, buf: &mut [u8]) -> Result<()> {
        if buf.len() != self.page_size {
            bail!(
                "Buffer size {} is not equal to page size {}",
                buf.len(),
                self.page_size
            );
        }
        if page_id > self.max_page_id {
            bail!("Page id {} is out of range", page_id);
        }
        let offset = DATA_OFFSET + page_id * self.page_size;
        self.file.seek(SeekFrom::Start(offset as u64))?;
        self.file.read_exact(buf)?;
        Ok(())
    }

    pub fn create_page(&mut self) -> Result<usize> {
        if self.max_page_id >= usize::MAX {
            bail!("Max page id is out of range");
        }
        let page_id = self.max_page_id;
        self.max_page_id += 1;
        write_usize(&mut self.file, MAX_PAGE_ID_OFFSET, self.max_page_id)?;
        Ok(page_id)
    }
}

#[test]
fn test_index_file() {
    let path = "test_index_file.bin";
    let page_size = 4096;
    let vector_unit_size = 32;
    let mut index_file = IndexFile::create(
        path,
        page_size,
        vector_unit_size,
        &SummaryMethod::Centroid,
        ClusterMethod::KMeans,
    )
    .unwrap();
    index_file.print_metadata();

    let page_id = index_file.create_page().unwrap();
    assert_eq!(page_id, 0);
    let data = vec![0; page_size];
    index_file.write_page(&data).unwrap();

    let mut buf = vec![0; page_size];
    index_file.read_page(page_id, &mut buf).unwrap();
    assert_eq!(buf, data);

    std::fs::remove_file(path).unwrap();
}
