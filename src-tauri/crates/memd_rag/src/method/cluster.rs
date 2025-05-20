use anyhow::Result;
use linfa::prelude::*;
use linfa_clustering::GaussianMixtureModel;
use linfa_clustering::KMeans;
use linfa_linalg::cholesky::Cholesky;
use linfa_linalg::triangular::SolveTriangularInplace;
use linfa_linalg::triangular::UPLO;
use ndarray::s;
use ndarray::Array;
use ndarray::Array3;
use ndarray::Axis;
use ndarray::Zip;
use ndarray::{Array1, Array2};
use tracing::info;

/// For every clustering, we need to store the cluster labels and the centroids.
///
/// For KMeans, the centroids are the cluster centers.
/// For GMM, the centroids are the means of the Gaussian distributions.
#[derive(Debug, Clone)]
pub struct ClusterResult {
    pub cluster_labels: Array1<usize>,
    pub centroids: Array2<f32>,
}

/// The Quake paper uses balanced k-means for clustering.
/// and let the K be $(the number of clusters) / \tau_s$
pub fn cluster_by_kmeans(embeddings: &Vec<&[f32]>, k: usize) -> ClusterResult {
    let data = Array2::from_shape_vec((embeddings.len(), embeddings[0].len()), embeddings.concat());

    let dataset = DatasetBase::from(data.unwrap());
    let kmeans = KMeans::params(k)
        .max_n_iterations(10)
        .tolerance(1e-5)
        .fit(&dataset)
        .unwrap();

    let cluster_labels = kmeans.predict(&dataset);
    let centroids = kmeans.centroids().clone();

    ClusterResult {
        cluster_labels,
        centroids,
    }
}

pub fn cluster_by_kmeans_slice(
    embeddings: &[f32],
    unit_vector_size: usize,
    k: usize,
) -> ClusterResult {
    let data = embeddings
        .chunks_exact(unit_vector_size)
        .map(|chunk| chunk)
        .collect::<Vec<_>>();

    cluster_by_kmeans(&data, k)
}

#[test]
fn test_cluster_by_kmeans() {
    let embeddings = vec![0.0, 1.0, 2.0, 3.0, 40.0, 41.0];
    let unit_vector_size = 2;
    let k = 2;

    let cluster_labels = cluster_by_kmeans_slice(&embeddings, unit_vector_size, k);
    assert_eq!(cluster_labels.cluster_labels.len(), 3);
    println!("Cluster labels: {:?}", cluster_labels);
}

/// transform gmm to [0, 1] to improve numerical stability.
fn gmm_standardize(samples: &Array2<f32>) -> Array2<f32> {
    let data_mean = samples.mean_axis(Axis(0)).unwrap();
    let data_std = samples.std_axis(Axis(0), 0.0);
    let standized_data = (samples - &data_mean) / &data_std;
    standized_data
}

fn compute_precisions_cholesky_full(covariances: &Array3<f32>) -> Result<Array3<f32>> {
    let n_clusters = covariances.shape()[0];
    let n_features = covariances.shape()[1];
    let mut precisions_chol = Array::zeros((n_clusters, n_features, n_features));
    for (k, covariance) in covariances.outer_iter().enumerate() {
        let sol = {
            let decomp = covariance.cholesky()?;
            decomp.solve_triangular_into(Array::eye(n_features), UPLO::Lower)?
        };

        precisions_chol.slice_mut(s![k, .., ..]).assign(&sol.t());
    }
    Ok(precisions_chol)
}

fn compute_log_det_cholesky_full(matrix_chol: &Array3<f32>, n_features: usize) -> Array1<f32> {
    let n_clusters = matrix_chol.shape()[0];
    let log_diags = &matrix_chol
        .to_owned()
        .into_shape((n_clusters, n_features * n_features))
        .unwrap()
        .slice(s![.., ..; n_features+1])
        .to_owned()
        .mapv(|x| x.ln());
    log_diags.sum_axis(Axis(1))
}

fn estimate_log_weights(gmm: &GaussianMixtureModel<f32>) -> Array1<f32> {
    gmm.weights().mapv(|x| x.ln())
}

fn estimate_log_gaussian_prob(
    gmm: &GaussianMixtureModel<f32>,
    observations: &Array2<f32>,
) -> Result<Array2<f32>> {
    let n_samples = observations.nrows();
    let n_features = observations.ncols();
    let means = gmm.means();
    let n_clusters = means.nrows();

    let precisions_chol = compute_precisions_cholesky_full(gmm.covariances())?;

    // GmmCovarType = full
    // det(precision_chol) is half of det(precision)
    let log_det = compute_log_det_cholesky_full(&precisions_chol, n_features);
    let mut log_prob: Array2<f32> = Array::zeros((n_samples, n_clusters));
    Zip::indexed(means.rows())
        .and(precisions_chol.outer_iter())
        .for_each(|k, mu, prec_chol| {
            let diff = (&observations.to_owned() - &mu).dot(&prec_chol);
            log_prob
                .slice_mut(s![.., k])
                .assign(&diff.mapv(|v| v * v).sum_axis(Axis(1)))
        });
    let log_gaussian = log_prob
        .mapv(|v| -0.5 * (v + n_features as f32 * f32::ln(2. * std::f32::consts::PI)))
        + log_det;

    let log_weights = estimate_log_weights(gmm);

    Ok(log_gaussian + log_weights)
}

pub fn score_samples(model: &GaussianMixtureModel<f32>, observations: &Array2<f32>) -> Array1<f32> {
    let weighted_log_probs = estimate_log_gaussian_prob(&model, &observations).unwrap();

    weighted_log_probs.map_axis(Axis(1), |row| {
        let max_val = row.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        let sum_exp = row.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln();
        max_val + sum_exp
    })
}

pub fn calculate_log_likelihood_score(
    model: &GaussianMixtureModel<f32>,
    observations: &Array2<f32>,
) -> f32 {
    score_samples(model, observations).sum()
}

pub fn cluster_by_gmm_bic(
    embeddings: &Vec<&[f32]>,
    min_clusters: usize,
    max_clusters: usize,
) -> ClusterResult {
    let raw_data =
        Array2::from_shape_vec((embeddings.len(), embeddings[0].len()), embeddings.concat())
            .unwrap();

    let data = gmm_standardize(&raw_data);

    let dataset = DatasetBase::from(data.clone());

    let calculate_gauss_bic = |clusters: usize| -> (GaussianMixtureModel<f32>, f32) {
        let model: GaussianMixtureModel<f32> = GaussianMixtureModel::params(clusters)
            .max_n_iterations(100)
            .init_method(linfa_clustering::GmmInitMethod::KMeans)
            .reg_covariance(1e-1)
            .tolerance(1e-1)
            .fit(&dataset)
            .unwrap();

        let n_samples = data.nrows() as f32;
        let n_features = data.ncols();

        let observations = &dataset.records;
        let log_likelihood = calculate_log_likelihood_score(&model, observations);

        let n_params = clusters as f32 * n_features as f32
            + clusters as f32 * n_features as f32 * (n_features as f32 + 1.0) / 2.0
            + (clusters - 1) as f32;

        let bic = n_params * n_samples.ln() - 2.0 * log_likelihood;

        info!(
            "For splitting to {} clusters, the log likelihood is {}, params is {}, bic is {}.",
            clusters, log_likelihood, n_params, bic
        );

        (model, bic)
    };

    let mut cluster_and_scores = (min_clusters..=max_clusters)
        .map(|cluster| (cluster, calculate_gauss_bic(cluster)))
        .collect::<Vec<_>>();

    cluster_and_scores.sort_by(|x, y| x.1 .1.total_cmp(&y.1 .1));

    let best_gmm = &cluster_and_scores.first().unwrap().1 .0;

    let cluster_labels = best_gmm.predict(&dataset);
    let centroids = best_gmm.centroids().clone();

    ClusterResult {
        cluster_labels,
        centroids,
    }
}

pub fn cluster_by_gmm_bic_slice(
    embeddings: &[f32],
    unit_vector_size: usize,
    min_clusters: usize,
    max_clusters: usize,
) -> ClusterResult {
    let data = embeddings
        .chunks_exact(unit_vector_size)
        .map(|chunk| chunk)
        .collect::<Vec<_>>();

    cluster_by_gmm_bic(&data, min_clusters, max_clusters)
}

pub fn cluster_by_gmm(embeddings: &Vec<Vec<f32>>, k: usize) -> ClusterResult {
    let data = Array2::from_shape_vec((embeddings.len(), embeddings[0].len()), embeddings.concat());

    let dataset = DatasetBase::from(data.unwrap());
    let gmm = GaussianMixtureModel::params(k)
        .reg_covariance(1e-1)
        .max_n_iterations(100)
        .tolerance(1e-3)
        .fit(&dataset)
        .unwrap();

    let cluster_labels = gmm.predict(&dataset);
    let centroids = gmm.centroids().clone();

    ClusterResult {
        cluster_labels,
        centroids,
    }
}

pub fn cluster_by_gmm_slice(
    embeddings: &[f32],
    unit_vector_size: usize,
    k: usize,
) -> ClusterResult {
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
    let k = 1;

    let cluster_labels = cluster_by_gmm_slice(&embeddings, unit_vector_size, k);
    assert_eq!(cluster_labels.cluster_labels.len(), 3);
    println!("Cluster labels: {:?}", cluster_labels);
}

#[test]
fn test_cluster_by_gmm_bic() {
    tracing_subscriber::fmt::init();

    let embeddings = vec![
        0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 39.0, 39.0, 40.0, 41.0, 43.0, 44.0,
    ];
    let unit_vector_size = 2;

    let cluster_results = cluster_by_gmm_bic_slice(&embeddings, unit_vector_size, 1, 3);
    println!("{:?}", cluster_results)
}
