use linfa::prelude::*;
use linfa_clustering::GaussianMixtureModel;
use linfa_clustering::KMeans;
use ndarray::{Array1, Array2};

pub fn cluster_by_kmeans(embeddings: &Vec<Vec<f32>>, k: usize) -> Array1<usize> {
    let data = Array2::from_shape_vec((embeddings.len(), embeddings[0].len()), embeddings.concat());

    let dataset = DatasetBase::from(data.unwrap());
    let kmeans = KMeans::params(k)
        .max_n_iterations(10)
        .tolerance(1e-5)
        .fit(&dataset)
        .unwrap();
    kmeans.predict(&dataset)
}

pub fn cluster_by_kmeans_slice(
    embeddings: &[f32],
    unit_vector_size: usize,
    k: usize,
) -> Array1<usize> {
    let data = embeddings
        .chunks_exact(unit_vector_size)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();

    cluster_by_kmeans(&data, k)
}

#[test]
fn test_cluster_by_kmeans() {
    let embeddings = vec![0.0, 1.0, 2.0, 3.0, 40.0, 41.0];
    let unit_vector_size = 2;
    let k = 2;

    let cluster_labels = cluster_by_kmeans_slice(&embeddings, unit_vector_size, k);
    assert_eq!(cluster_labels.len(), 3);
    println!("Cluster labels: {:?}", cluster_labels);
}

pub fn cluster_by_gmm(embeddings: &Vec<Vec<f32>>, k: usize) -> Array1<usize> {
    let data = Array2::from_shape_vec((embeddings.len(), embeddings[0].len()), embeddings.concat());

    let dataset = DatasetBase::from(data.unwrap());
    let gmm = GaussianMixtureModel::params(k)
        .max_n_iterations(10)
        .tolerance(1e-5)
        .fit(&dataset)
        .unwrap();

    gmm.predict(&dataset)
}

pub fn cluster_by_gmm_slice(
    embeddings: &[f32],
    unit_vector_size: usize,
    k: usize,
) -> Array1<usize> {
    let data = embeddings
        .chunks_exact(unit_vector_size)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();

    cluster_by_gmm(&data, k)
}

#[test]
fn test_cluster_by_gmm() {
    let embeddings = vec![0.0, 1.0, 2.0, 3.0, 40.0, 41.0];
    let unit_vector_size = 2;
    let k = 2;

    let cluster_labels = cluster_by_gmm_slice(&embeddings, unit_vector_size, k);
    assert_eq!(cluster_labels.len(), 3);
    println!("Cluster labels: {:?}", cluster_labels);
}
