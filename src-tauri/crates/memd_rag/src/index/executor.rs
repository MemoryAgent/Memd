use anyhow::{Context, Result};
use candle_transformers::models::bert::BertModel;
use rusqlite::Connection;
use std::{
    cmp::max,
    collections::BinaryHeap,
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tokenizers::Tokenizer;

use crate::{
    component::{
        bert::encode_single_sentence,
        database::{Chunk, Store},
        deepseek::extract_answer,
        llm::{LocalLlm, LLM},
    },
    method::cluster::{cluster_by_gmm_bic, cluster_by_kmeans, ClusterResult},
};

use super::{
    buffer_pool::BufferPool,
    index_file::{usize_to_summary_method, IndexFile},
    page::{
        self, get_internal_reader_from_buffer, get_leaf_reader_from_buffer, read_page_type,
        RecordID,
    },
};

/// TODO: make top-K buffer generic over content type and define intermediate data
/// structures for leaf and internal nodes.
#[derive(Clone, Debug)]
pub struct TopKBuffer {
    /// record id -> vector ID -> conf_score -> Option<child id>
    buffer: Vec<(RecordID, usize, f32, Option<usize>)>,
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

    pub fn push(&mut self, record_id: RecordID, id: usize, score: f32, child_id: Option<usize>) {
        if self.buffer.len() < self.k {
            self.buffer.push((record_id, id, score, child_id));
            // Total order comparison of floats is supported since Rust 1.62
            self.buffer.sort_by(|a, b| a.2.total_cmp(&b.2));
        } else if score > self.buffer[0].2 {
            self.buffer[0] = (record_id, id, score, child_id);
            self.buffer.sort_by(|a, b| a.2.total_cmp(&b.2));
        }
    }

    pub fn extend_from(&mut self, another_buffer: &TopKBuffer) {
        for (record_id, id, score, child_id) in &another_buffer.buffer {
            self.push(*record_id, *id, *score, *child_id);
        }
    }

    pub fn get_topk(&self) -> &[(RecordID, usize, f32, Option<usize>)] {
        &self.buffer
    }

    pub fn get_lowest_score(&self) -> f32 {
        self.buffer
            .first()
            .map(|(_, _, score, _)| *score)
            .unwrap_or(0.0)
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

#[test]
fn test_topk_buffer() {
    // Create a buffer with capacity 3
    let mut buffer = TopKBuffer::new(3);

    // Test empty buffer
    assert!(buffer.get_topk().is_empty());

    // Add first element
    buffer.push(
        RecordID {
            page_id: 1,
            slot_id: 1,
        },
        101,
        0.5,
        None,
    );
    assert_eq!(
        buffer.get_topk(),
        &[(
            RecordID {
                page_id: 1,
                slot_id: 1
            },
            101,
            0.5,
            None
        )]
    );

    // Add second element (lower score)
    buffer.push(
        RecordID {
            page_id: 1,
            slot_id: 2,
        },
        102,
        0.3,
        None,
    );
    assert_eq!(
        buffer.get_topk(),
        &[
            (
                RecordID {
                    page_id: 1,
                    slot_id: 2
                },
                102,
                0.3,
                None
            ),
            (
                RecordID {
                    page_id: 1,
                    slot_id: 1
                },
                101,
                0.5,
                None
            )
        ]
    );

    // Add third element (middle score)
    buffer.push(
        RecordID {
            page_id: 2,
            slot_id: 1,
        },
        103,
        0.4,
        None,
    );
    assert_eq!(
        buffer.get_topk(),
        &[
            (
                RecordID {
                    page_id: 1,
                    slot_id: 2
                },
                102,
                0.3,
                None
            ),
            (
                RecordID {
                    page_id: 2,
                    slot_id: 1
                },
                103,
                0.4,
                None
            ),
            (
                RecordID {
                    page_id: 1,
                    slot_id: 1
                },
                101,
                0.5,
                None
            )
        ]
    );

    // Add element that's too small (should be ignored)
    buffer.push(
        RecordID {
            page_id: 2,
            slot_id: 2,
        },
        104,
        0.2,
        None,
    );
    assert_eq!(
        buffer.get_topk(),
        &[
            (
                RecordID {
                    page_id: 1,
                    slot_id: 2
                },
                102,
                0.3,
                None
            ),
            (
                RecordID {
                    page_id: 2,
                    slot_id: 1
                },
                103,
                0.4,
                None
            ),
            (
                RecordID {
                    page_id: 1,
                    slot_id: 1
                },
                101,
                0.5,
                None
            )
        ]
    );

    // Add element that replaces the smallest
    buffer.push(
        RecordID {
            page_id: 3,
            slot_id: 1,
        },
        105,
        0.35,
        None,
    );
    assert_eq!(
        buffer.get_topk(),
        &[
            (
                RecordID {
                    page_id: 3,
                    slot_id: 1
                },
                105,
                0.35,
                None
            ),
            (
                RecordID {
                    page_id: 2,
                    slot_id: 1
                },
                103,
                0.4,
                None
            ),
            (
                RecordID {
                    page_id: 1,
                    slot_id: 1
                },
                101,
                0.5,
                None
            )
        ]
    );

    // Add element that would be largest
    buffer.push(
        RecordID {
            page_id: 3,
            slot_id: 2,
        },
        106,
        0.6,
        None,
    );
    assert_eq!(
        buffer.get_topk(),
        &[
            (
                RecordID {
                    page_id: 2,
                    slot_id: 1
                },
                103,
                0.4,
                None
            ),
            (
                RecordID {
                    page_id: 1,
                    slot_id: 1
                },
                101,
                0.5,
                None
            ),
            (
                RecordID {
                    page_id: 3,
                    slot_id: 2
                },
                106,
                0.6,
                None
            )
        ]
    );

    // Test with single element buffer
    let mut single_buffer = TopKBuffer::new(1);
    single_buffer.push(
        RecordID {
            page_id: 10,
            slot_id: 1,
        },
        110,
        0.1,
        None,
    );
    assert_eq!(
        single_buffer.get_topk(),
        &[(
            RecordID {
                page_id: 10,
                slot_id: 1
            },
            110,
            0.1,
            None
        )]
    );

    single_buffer.push(
        RecordID {
            page_id: 10,
            slot_id: 2,
        },
        111,
        0.2,
        None,
    );
    assert_eq!(
        single_buffer.get_topk(),
        &[(
            RecordID {
                page_id: 10,
                slot_id: 2
            },
            111,
            0.2,
            None
        )]
    );

    single_buffer.push(
        RecordID {
            page_id: 11,
            slot_id: 1,
        },
        112,
        0.05,
        None,
    );
    assert_eq!(
        single_buffer.get_topk(),
        &[(
            RecordID {
                page_id: 10,
                slot_id: 2
            },
            111,
            0.2,
            None
        )]
    );
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

/// reinterpret &[f32] as &[u8]
fn reinterpret_as_u8<'a>(slice: &'a [f32]) -> &'a [u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4) }
}

fn get_summary_prompt(text: &[String]) -> String {
    let mut prompt =
        "Please summarize the following text, include as many details as possible:\n".to_string();
    for (i, line) in text.iter().enumerate() {
        prompt.push_str(&format!("{}: {}\n", i + 1, line));
    }
    prompt.push_str("Summary:");
    prompt
}

// For a internal index, we need to store the summary and its embedding.
#[derive(Clone, Debug)]
pub struct InternalIndexEntry {
    pub summary: String,
    pub embedding: Vec<f32>,
}

// The database assigns vec_id after insertion.
#[derive(Clone, Debug)]
pub struct InternalIndexRow {
    pub vec_id: usize,
    pub summary: String,
    pub embedding: Vec<f32>,
}

// sqlite functions
fn insert_internal_index(
    conn: &mut rusqlite::Connection,
    index_entry: &InternalIndexEntry,
) -> Result<InternalIndexRow> {
    use crate::component::sqlite;
    conn.query_row(
        "INSERT INTO internal_vector_index (summary, embedding) VALUES (?, ?) RETURNING id",
        (
            &index_entry.summary,
            &sqlite::to_binary_string(&index_entry.embedding),
        ),
        |row| {
            let id: usize = row.get(0)?;
            Ok(InternalIndexRow {
                vec_id: id,
                summary: index_entry.summary.clone(),
                embedding: index_entry.embedding.clone(),
            })
        },
    )
    .with_context(|| {
        format!(
            "Failed to insert internal index entry: {}",
            index_entry.summary
        )
    })
}

fn insert_internal_relationship(
    conn: &mut rusqlite::Connection,
    parent_id: usize,
    child_id: i64,
) -> Result<()> {
    conn.execute(
        "INSERT INTO internal_vector_children (parent_id, child_id) VALUES (?, ?)",
        (parent_id, child_id),
    )
    .with_context(|| {
        format!(
            "Failed to insert internal index relationship: {} -> {}",
            parent_id, child_id
        )
    })?;
    Ok(())
}

/// LLM method uses large language model to summarize the text.
///
/// Centroid method does not summarize the text, but uses the centroid of vectors as the embedding
/// of internal index.
pub enum SummaryMethod {
    LLM {
        llm: Arc<Mutex<dyn LLM>>,
        tokenizer: Arc<Mutex<Tokenizer>>,
        bert: Arc<Mutex<BertModel>>,
    },
    Centroid,
}

impl std::fmt::Debug for SummaryMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LLM { .. } => write!(f, "llm"),
            Self::Centroid => write!(f, "Centroid"),
        }
    }
}

/// We could use two methods for clustering...
/// and there's also a option without clustering.
///
/// RAPTORS uses GMM, while Quake uses K-means
#[derive(Clone, Copy, Debug)]
pub enum ClusterMethod {
    GMM,
    KMeans,
    // If we use no cluster mode, the only work is to store all vectors in pages.
    NoCluster,
}

pub struct ShmIndex {
    buffer_pool: BufferPool,
    database: Arc<Mutex<Connection>>,
    summary_method: SummaryMethod,
    cluster_method: ClusterMethod,
    target_similarity: f32,
}

#[derive(Debug)]
pub struct ShmIndexOptions {
    /// The path to the file that will be used for backing store.
    pub backed_file: PathBuf,
    pub database: Arc<Mutex<Connection>>,
    /// The size of the buffer pool.
    pub pool_size: usize,
    /// The size of each page.
    pub page_size: usize,
    /// The size of each vector unit.
    pub vector_unit_size: usize,
    pub summary_method: SummaryMethod,
    pub cluster_method: ClusterMethod,
    pub target_similarity: f32,
}

impl Default for ShmIndexOptions {
    fn default() -> Self {
        let database_instance = Connection::open_in_memory().unwrap();

        let database = Arc::new(Mutex::new(database_instance));

        Self {
            backed_file: PathBuf::from("test_index.bin"),
            database,
            pool_size: 10,
            page_size: 4096,
            vector_unit_size: 4 * 368,
            summary_method: SummaryMethod::Centroid,
            cluster_method: ClusterMethod::KMeans,
            target_similarity: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct InternalChunk {
    summary: String,
    embedding: Vec<f32>,
    child_page_id: usize,
    vec_id: usize,
}

impl ShmIndex {
    /// The root page is 0 by default. If all vectors can reside in one page, then the root page
    /// is a leaf page, and the vector index simply traverse this page to find top-k vectors.
    ///
    /// Otherwise, in bulk insertion stage, if the size of embedding vectors exceeds a whole page,
    /// it is divided into many many clusters. These clusters are built to a hierarchy, whose final
    /// internal page is the root page.
    const ROOT_PAGE_ID: usize = 0;

    pub fn new(options: ShmIndexOptions) -> Result<Self> {
        let summary_method = options.summary_method;

        let backed_file = IndexFile::create(
            options.backed_file,
            options.page_size,
            options.vector_unit_size,
            &summary_method,
            options.cluster_method,
        )?;

        let buffer_pool = BufferPool::new(
            backed_file,
            options.pool_size,
            options.page_size,
            options.vector_unit_size,
        );

        let cluster_method = options.cluster_method;

        let target_similarity = options.target_similarity;

        let database = options.database;

        Ok(Self {
            buffer_pool,
            database,
            summary_method,
            cluster_method,
            target_similarity,
        })
    }

    pub fn load(
        file: PathBuf,
        pool_size: usize,
        target_similarity: f32,
        database: Arc<Mutex<Connection>>,
        llm: Option<Arc<Mutex<dyn LLM>>>,
        tokenizer: Option<Arc<Mutex<Tokenizer>>>,
        bert: Option<Arc<Mutex<BertModel>>>,
    ) -> Result<Self> {
        let backed_file = IndexFile::open(file)?;
        let page_size = backed_file.page_size;
        let vector_unit_size = backed_file.vector_unit_size;
        let summary_method =
            usize_to_summary_method(backed_file.summary_method, llm, tokenizer, bert)?;
        let cluster_method = backed_file.cluster_method;

        let buffer_pool = BufferPool::new(backed_file, pool_size, page_size, vector_unit_size);

        Ok(Self {
            buffer_pool,
            database,
            summary_method,
            cluster_method,
            target_similarity,
        })
    }

    pub fn persist(&mut self) -> Result<()> {
        self.buffer_pool.flush_all();
        Ok(())
    }

    fn is_empty_index(&self) -> bool {
        self.buffer_pool.backed_file.max_page_id == 0
    }

    fn get_latest_page(&self) -> usize {
        self.buffer_pool.backed_file.max_page_id - 1
    }

    fn bulk_build_no_cluster(&mut self, chunks: &[Chunk]) {
        for chunk in chunks.iter() {
            self.insert_no_cluster(
                reinterpret_as_u8(chunk.content_vector.as_slice()),
                chunk.id.try_into().unwrap(),
            )
            .unwrap();
        }
    }

    // this could be pre-calculated because it does not change
    // TODO: optimization, low priority
    fn decide_leaf_k(&mut self, length: usize) -> usize {
        max(length.div_ceil(self.buffer_pool.get_leaf_max_vectors()), 2)
    }

    fn decide_internal_k(&mut self, length: usize) -> usize {
        max(
            length.div_ceil(self.buffer_pool.get_internal_max_vectors()),
            2,
        )
    }

    // TODO: bulk build without LLM summary.

    // assume chunks is needed to split
    // every call into this function will build upper one level node
    // returns current level nodes
    fn cluster_bulk_build_recursive(&mut self, chunks: &[InternalChunk]) -> Vec<InternalChunk> {
        if chunks.len() <= self.buffer_pool.get_internal_max_vectors() {
            return chunks.to_vec();
        }

        let embeddings: Vec<&[f32]> = chunks
            .iter()
            .map(|chunk| chunk.embedding.as_slice())
            .collect();

        let min_clusters = self.decide_internal_k(embeddings.len());

        let ClusterResult {
            cluster_labels,
            centroids,
        } = match self.cluster_method {
            ClusterMethod::GMM => cluster_by_gmm_bic(&embeddings, min_clusters, min_clusters + 2),
            ClusterMethod::KMeans => cluster_by_kmeans(&embeddings, min_clusters),
            ClusterMethod::NoCluster => {
                unreachable!("using no cluster in cluster mode is unreachable.")
            }
        };

        // group chunks by cluster labels
        let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); centroids.len()];

        for (i, label) in cluster_labels.iter().enumerate() {
            clusters[*label].push(i);
        }

        let higher_level_internal_chunks: Vec<InternalChunk> = clusters
            .iter()
            .enumerate()
            .map(|(cluster_id, cluster)| {
                let (summary, embedding) = match &self.summary_method {
                    SummaryMethod::LLM {
                        llm,
                        tokenizer,
                        bert,
                    } => {
                        let texts: Vec<String> = cluster
                            .iter()
                            .map(|chunk_id| chunks[*chunk_id].summary.clone())
                            .collect();
                        let prompt = get_summary_prompt(&texts);
                        let summary = llm.lock().unwrap().llm_complete(&prompt).unwrap();

                        let mut tokenizer = tokenizer.lock().unwrap();
                        let bert = bert.lock().unwrap();

                        let embedding =
                            encode_single_sentence(&summary, &mut tokenizer, &bert).unwrap();

                        (summary, embedding.to_vec1().unwrap())
                    }
                    SummaryMethod::Centroid => {
                        ("no summary".to_string(), centroids.row(cluster_id).to_vec())
                    }
                };

                let entry = InternalIndexEntry { summary, embedding };

                let row = {
                    let mut conn = self.database.lock().unwrap();

                    insert_internal_index(&mut conn, &entry)
                }
                .unwrap();

                // TODO: maintain parent information
                let internal_page_id = self.buffer_pool.create_internal_page(0);
                let internal_page = self.buffer_pool.fetch_page(internal_page_id);
                let mut internal_page_accessor =
                    get_internal_reader_from_buffer(internal_page.get_page());

                for chunk in cluster.iter() {
                    let mut conn = self.database.lock().unwrap();

                    let current_chunk = &chunks[*chunk];

                    insert_internal_relationship(
                        &mut conn,
                        row.vec_id,
                        current_chunk.vec_id.try_into().unwrap(),
                    )
                    .unwrap();

                    internal_page_accessor
                        .append_record(
                            current_chunk.vec_id,
                            reinterpret_as_u8(&current_chunk.embedding),
                            current_chunk.child_page_id,
                        )
                        .unwrap();
                }

                InternalChunk {
                    summary: entry.summary.clone(),
                    embedding: entry.embedding.clone(),
                    child_page_id: internal_page_id,
                    vec_id: row.vec_id,
                }
            })
            .collect();

        // tail recursion
        self.cluster_bulk_build_recursive(&higher_level_internal_chunks)
    }

    fn cluster_bulk_build(&mut self, chunks: &[Chunk]) {
        // scenario one: one page can store all vectors. In this case we simply store all data to
        // this page.
        if chunks.len() < self.buffer_pool.get_leaf_max_vectors() {
            let leaf_page_id = self.buffer_pool.create_leaf_page();
            assert_eq!(leaf_page_id, Self::ROOT_PAGE_ID);
            let leaf_page = self.buffer_pool.fetch_page(leaf_page_id);
            let mut leaf_page_accessor = get_leaf_reader_from_buffer(leaf_page.get_page());

            for chunk in chunks.iter() {
                leaf_page_accessor
                    .append_record(
                        chunk.id.try_into().unwrap(),
                        reinterpret_as_u8(&chunk.content_vector),
                    )
                    .unwrap();
            }

            return ();
        }

        let embeddings: Vec<&[f32]> = chunks
            .iter()
            .map(|chunk| chunk.content_vector.as_slice())
            .collect();

        let min_clusters = self.decide_leaf_k(embeddings.len());

        let ClusterResult {
            cluster_labels,
            centroids,
        } = match self.cluster_method {
            ClusterMethod::GMM => cluster_by_gmm_bic(&embeddings, min_clusters, min_clusters + 2),
            ClusterMethod::KMeans => cluster_by_kmeans(&embeddings, min_clusters),
            ClusterMethod::NoCluster => {
                unreachable!("using no cluster in cluster mode is unreachable.")
            }
        };

        // group chunks by cluster labels
        let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); centroids.len()];

        for (i, label) in cluster_labels.iter().enumerate() {
            clusters[*label].push(i);
        }

        let first_internal_layer: Vec<InternalChunk> = clusters
            .iter()
            .enumerate()
            .map(|(cluster_id, cluster)| {
                let (summary, embedding) = match &self.summary_method {
                    SummaryMethod::LLM {
                        llm,
                        tokenizer,
                        bert,
                    } => {
                        let texts: Vec<String> = cluster
                            .iter()
                            .map(|chunk_id| chunks[*chunk_id].content.clone())
                            .collect();
                        let prompt = get_summary_prompt(&texts);
                        let summary = llm.lock().unwrap().llm_complete(&prompt).unwrap();

                        let mut tokenizer = tokenizer.lock().unwrap();
                        let bert = bert.lock().unwrap();

                        let embedding =
                            encode_single_sentence(&summary, &mut tokenizer, &bert).unwrap();

                        (summary, embedding.to_vec1().unwrap())
                    }
                    SummaryMethod::Centroid => {
                        ("no summary".to_string(), centroids.row(cluster_id).to_vec())
                    }
                };

                let index_entry = InternalIndexEntry { summary, embedding };
                let index_row = {
                    let mut conn = self.database.lock().unwrap();

                    insert_internal_index(&mut conn, &index_entry)
                }
                .unwrap();

                let leaf_page_id = self.buffer_pool.create_leaf_page();
                let leaf_page = self.buffer_pool.fetch_page(leaf_page_id);
                let mut leaf_page_accessor = get_leaf_reader_from_buffer(leaf_page.get_page());

                for chunk in cluster.iter() {
                    let mut conn = self.database.lock().unwrap();

                    let current_chunk = &chunks[*chunk];

                    insert_internal_relationship(&mut conn, index_row.vec_id, current_chunk.id)
                        .unwrap();

                    leaf_page_accessor
                        .append_record(
                            current_chunk.id.try_into().unwrap(),
                            reinterpret_as_u8(&current_chunk.content_vector),
                        )
                        .unwrap();
                }

                InternalChunk {
                    summary: index_entry.summary.clone(),
                    embedding: index_entry.embedding.clone(),
                    child_page_id: leaf_page_id,
                    vec_id: index_row.vec_id,
                }
            })
            .collect();

        // top level
        let internal_page_id = self.buffer_pool.create_internal_page(0);
        let internal_page = self.buffer_pool.fetch_page(internal_page_id);
        let mut internal_page_accessor = get_internal_reader_from_buffer(internal_page.get_page());

        let top_level_internal_chunks = self.cluster_bulk_build_recursive(&first_internal_layer);

        for chunk in top_level_internal_chunks.iter() {
            internal_page_accessor
                .append_record(
                    chunk.child_page_id,
                    reinterpret_as_u8(&chunk.embedding),
                    chunk.child_page_id,
                )
                .unwrap();
        }

        self.buffer_pool.flush_all();
    }

    pub fn bulk_build(&mut self, chunks: &[Chunk]) {
        // The problem is, the index is part of local_comps. Either we move out these components,
        // or we pass them as arguments.
        //
        // And for consistent interface with usearch, we need to hide them behind implementation details.
        match self.cluster_method {
            ClusterMethod::NoCluster => self.bulk_build_no_cluster(chunks),
            ClusterMethod::KMeans | ClusterMethod::GMM => self.cluster_bulk_build(chunks),
        }
    }

    fn review_page_capacity(&mut self, page_id: usize) -> bool {
        let page = self.buffer_pool.fetch_page(page_id);
        let page_data = page.get_page();
        let page_type = read_page_type(page_data);
        match page_type {
            page::PageType::LeafPage => {
                let leaf_page_accessor = get_leaf_reader_from_buffer(page_data);
                leaf_page_accessor.has_free_slot()
            }
            page::PageType::IndexPage => {
                let internal_page_accessor = get_internal_reader_from_buffer(page_data);
                internal_page_accessor.has_free_slot()
            }
        }
    }

    /// When the index is in no cluster mode, the new vector is added to the last page, and if
    /// that page is full, a new page is created.
    fn insert_no_cluster(&mut self, vector_data: &[u8], vector_id: usize) -> Result<RecordID> {
        let page_id = if self.is_empty_index() || !self.review_page_capacity(self.get_latest_page())
        {
            self.buffer_pool.create_leaf_page()
        } else {
            self.get_latest_page()
        };

        let page = self.buffer_pool.fetch_page(page_id);
        let page_data = page.get_page();
        let mut leaf_accessor = get_leaf_reader_from_buffer(page_data);
        let slot_id = leaf_accessor.append_record(vector_id.try_into().unwrap(), vector_data)?;
        Ok(slot_id)
    }

    fn insert_with_cluster(&mut self, vector_data: &[u8], vector_id: usize) -> Result<RecordID> {
        todo!()
    }

    /// insert one vector into the index
    fn insert_type_erased(&mut self, vector_data: &[u8], vector_id: usize) -> Result<RecordID> {
        match self.cluster_method {
            ClusterMethod::NoCluster => self.insert_no_cluster(vector_data, vector_id),
            ClusterMethod::KMeans | ClusterMethod::GMM => {
                self.insert_with_cluster(vector_data, vector_id)
            }
        }
    }

    fn insert_single(&mut self, embedding: &[f32], vector_id: usize) -> Result<RecordID> {
        let vector_data = reinterpret_as_u8(embedding);
        self.insert_type_erased(vector_data, vector_id)
    }

    pub fn insert(&mut self, chunks: &[Chunk]) -> Result<()> {
        for chunk in chunks.iter() {
            self.insert_single(
                chunk.content_vector.as_slice(),
                chunk.id.try_into().unwrap(),
            )?;
        }
        Ok(())
    }

    // query in one leaf page, should traverse the whole leaf. this is linear complexity.
    fn query_in_leaf_page(
        &mut self,
        target_vector: &[u8],
        page: &[u8],
        k: usize,
    ) -> Result<TopKBuffer> {
        let leaf_page_accessor = get_leaf_reader_from_buffer(page);
        let mut result_buffer = TopKBuffer::new(k);
        let page_id = leaf_page_accessor.read_page_id();

        for ((vec_id, embedding), slot_id) in leaf_page_accessor.iter().zip(0..) {
            let target_in_float = reinterpret_as_f32(target_vector);
            let page_vector_in_float = reinterpret_as_f32(embedding);
            let score = calculate_similarity(target_in_float, page_vector_in_float);
            // Leaf does not have child.
            // TODO: Use option in API is too ugly
            result_buffer.push(RecordID { page_id, slot_id }, vec_id, score, None);
        }

        Ok(result_buffer)
    }

    // If this index is built without internal pages, then we can only traverse the index to find
    // the top-k vectors.
    //
    // assert the root page is a leaf page.
    fn linear_query(&mut self, target_vector: &[u8], k: usize) -> Result<TopKBuffer> {
        let mut top_k_buffer = TopKBuffer::new(k);
        for i in Self::ROOT_PAGE_ID..self.buffer_pool.backed_file.max_page_id {
            let page = self.buffer_pool.fetch_page(i);
            let page_data = page.get_page();
            let page_results = self.query_in_leaf_page(target_vector, page_data, k)?;
            top_k_buffer.extend_from(&page_results);
        }
        Ok(top_k_buffer)
    }

    fn query_in_internal_page(
        &mut self,
        target_vector: &[u8],
        page: &[u8],
        k: usize,
    ) -> Result<TopKBuffer> {
        let internal_page_accessor = get_internal_reader_from_buffer(page);
        let mut result_buffer = TopKBuffer::new(k);
        let page_id = internal_page_accessor.read_page_id();

        for ((vec_id, embedding, child_page_id), slot_id) in internal_page_accessor.iter().zip(0..)
        {
            let target_in_float = reinterpret_as_f32(target_vector);
            let page_vector_in_float = reinterpret_as_f32(embedding);
            let score = calculate_similarity(target_in_float, page_vector_in_float);
            result_buffer.push(
                RecordID { page_id, slot_id },
                vec_id,
                score,
                Some(child_page_id),
            );
        }

        Ok(result_buffer)
    }

    /// query strategy: here use a priority queue, for every layer scan several targets.
    /// Until pq is empty or similarity metric is fulfilled.
    fn query_internal_from(
        &mut self,
        target_vector: &[u8],
        page_id: usize,
        k: usize,
        target_similarity: f32,
    ) -> Result<TopKBuffer> {
        // temp data structure for priority queue.
        #[derive(Debug, Clone)]
        struct QueryPage {
            score: f32,
            page_id: usize,
        }

        impl PartialEq for QueryPage {
            fn eq(&self, other: &Self) -> bool {
                self.score == other.score
            }
        }

        impl Eq for QueryPage {}

        impl PartialOrd for QueryPage {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.score.partial_cmp(&other.score)
            }
        }

        impl Ord for QueryPage {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.score.total_cmp(&other.score)
            }
        }

        let mut pq = BinaryHeap::<QueryPage>::new();
        let mut top_k_buffer = TopKBuffer::new(k);

        pq.push(QueryPage {
            score: 1.0,
            page_id,
        });

        // This is tree structured, so every page only enqueue once.
        while !pq.is_empty()
            && (top_k_buffer.len() < k || top_k_buffer.get_lowest_score() < target_similarity)
        {
            let QueryPage { page_id, .. } = pq.pop().unwrap(); // The loop invariant guarantees that pop.
            let current_page = self.buffer_pool.fetch_page(page_id);
            let current_page_data = current_page.get_page();
            let current_page_type = read_page_type(current_page_data);

            match current_page_type {
                page::PageType::LeafPage => {
                    let leaf_result =
                        self.query_in_leaf_page(target_vector, current_page_data, k)?;
                    top_k_buffer.extend_from(&leaf_result);
                }
                page::PageType::IndexPage => {
                    let internal_result =
                        self.query_in_internal_page(target_vector, current_page_data, k)?;
                    let top_k_internal = internal_result.get_topk();
                    for (_, _, score, child_page_id) in top_k_internal {
                        let page_id = child_page_id.unwrap(); // this is guaranteed by type of page.
                        pq.push(QueryPage {
                            score: *score,
                            page_id,
                        });
                    }
                }
            }
        }

        Ok(top_k_buffer)
    }

    fn clustered_query(&mut self, target_vector: &[u8], k: usize) -> Result<TopKBuffer> {
        let root_page = self.buffer_pool.fetch_page(Self::ROOT_PAGE_ID);
        let root_page_data = root_page.get_page();
        let root_page_type = read_page_type(root_page_data);
        match root_page_type {
            page::PageType::LeafPage => self.query_in_leaf_page(target_vector, root_page_data, k),
            page::PageType::IndexPage => self.query_internal_from(
                target_vector,
                Self::ROOT_PAGE_ID,
                k,
                self.target_similarity,
            ),
        }
    }

    /// Query finds the top-k vectors similar to target_vector in the index.
    fn query_type_erased(&mut self, target_vector: &[u8], k: usize) -> Result<TopKBuffer> {
        match self.cluster_method {
            ClusterMethod::NoCluster => self.linear_query(target_vector, k),
            ClusterMethod::KMeans | ClusterMethod::GMM => self.clustered_query(target_vector, k),
        }
    }

    pub fn query(&mut self, target_vector: &[f32], k: usize) -> Result<TopKBuffer> {
        let vector_data = reinterpret_as_u8(target_vector);
        self.query_type_erased(vector_data, k)
    }

    pub fn memory_usage(&self) -> usize {
        self.buffer_pool.memory_usage()
    }
}

impl Drop for ShmIndex {
    fn drop(&mut self) {
        self.persist().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use crate::component::{operation, LocalComponent};

    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_shm_index() {
        let mut index = ShmIndex::new(ShmIndexOptions {
            backed_file: PathBuf::from("test.bin"),
            pool_size: 10,
            page_size: 4096,
            vector_unit_size: 4,
            summary_method: SummaryMethod::Centroid,
            cluster_method: ClusterMethod::KMeans,
            target_similarity: 0.0,
            ..Default::default()
        })
        .unwrap();
        let vector = 4.0_f32.to_le_bytes();
        let id = index.insert_type_erased(&vector, 0).unwrap();
        println!("inserted rid: {:?}", id);
        assert_eq!(id.page_id, 0);
        assert_eq!(id.slot_id, 0);
        let target_vector = 4.0_f32.to_le_bytes();
        let topk = index.query_type_erased(&target_vector, 1).unwrap();
        println!("topk: {:?}", topk.get_topk());
    }

    #[test]
    fn test_bulk_build() {
        let mut index = ShmIndex::new(ShmIndexOptions {
            backed_file: PathBuf::from("test_bulk_build.bin"),
            pool_size: 10,
            page_size: 4096,
            vector_unit_size: 4 * 384,
            summary_method: SummaryMethod::Centroid,
            cluster_method: ClusterMethod::KMeans,
            target_similarity: 0.0,
            ..Default::default()
        })
        .unwrap();
        let mut local_comps = LocalComponent::default();

        let full_text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

        let inserted_full_text = local_comps
            .store
            .add_document(&operation::Document {
                name: "hi".to_string(),
                content: full_text.to_string(),
            })
            .unwrap();

        let text = vec![
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
        ];

        let chunks: Vec<Chunk> = text
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let embedding =
                    encode_single_sentence(text, &mut local_comps.tokenizer, &local_comps.bert)
                        .unwrap();
                operation::Chunk {
                    full_doc_id: inserted_full_text.id,
                    chunk_index: 0,
                    tokens: 20,
                    content: text.clone(),
                    embedding,
                }
            })
            .map(|chunk| local_comps.store.add_chunk(&chunk).unwrap())
            .collect();

        index.cluster_bulk_build(&chunks);

        index.persist().unwrap();
    }

    #[test]
    fn test_bulk_build_two_layer() {
        let mut index = ShmIndex::new(ShmIndexOptions {
            backed_file: PathBuf::from("test_bulk_build_two_layer.bin"),
            pool_size: 10,
            page_size: 4096,
            vector_unit_size: 4 * 384,
            summary_method: SummaryMethod::Centroid,
            cluster_method: ClusterMethod::KMeans,
            target_similarity: 0.0,
            ..Default::default()
        })
        .unwrap();
        let mut local_comps = LocalComponent::default();

        let full_text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

        let inserted_full_text = local_comps
            .store
            .add_document(&operation::Document {
                name: "hi".to_string(),
                content: full_text.to_string(),
            })
            .unwrap();

        let text = vec![
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
            "ccccccccccccccccccccccccccccccccccccccccccccc".to_string(),
            "ccccccccccccccccccccccccccccccccccccccccccccc".to_string(),
            "ddddddddddddddddddddddddddddddddddddddddddddd".to_string(),
            "ddddddddddddddddddddddddddddddddddddddddddddd".to_string(),
        ];

        let chunks: Vec<Chunk> = text
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let embedding =
                    encode_single_sentence(text, &mut local_comps.tokenizer, &local_comps.bert)
                        .unwrap();
                operation::Chunk {
                    full_doc_id: inserted_full_text.id,
                    chunk_index: 0,
                    tokens: 20,
                    content: text.clone(),
                    embedding,
                }
            })
            .map(|chunk| local_comps.store.add_chunk(&chunk).unwrap())
            .collect();

        index.cluster_bulk_build(&chunks);

        index.persist().unwrap();
    }
}
