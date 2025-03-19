use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// [`Timer`] is used to record the processing time of this program.
pub struct Timer {
    session_started: SystemTime,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            session_started: SystemTime::now(),
        }
    }

    pub fn read(self) -> Result<Duration> {
        self.session_started
            .elapsed()
            .with_context(|| "Failed to read elapsed time")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestType {
    Store,
    Query,
    Chat,
}

#[test]
fn test_request_type_serialize() {
    let req_type = RequestType::Store;
    println!("{}", serde_json::to_string(&req_type).unwrap())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerRequestMetricData {
    pub kind: RequestType,
    pub time_cost: Duration,
    pub storage_memory_usage: usize,
    pub total_memory_usage_before: usize,
    pub total_memory_usage_after: usize,
}

/// Per request metric includes memory and cpu usage of each request in each method.
/// the index of vector is the request ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricData {
    pub request_metrics: Vec<PerRequestMetricData>,
    pub start_memory_usage: usize,
    pub end_memory_usage: usize,
}

pub struct Metrics {
    global_req_id: usize,
    metric_store: MetricData,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            global_req_id: 0,
            metric_store: MetricData {
                request_metrics: Vec::new(),
                start_memory_usage: 0,
                end_memory_usage: 0,
            },
        }
    }

    pub fn reset(&mut self) {
        self.global_req_id = 0;
        self.metric_store.request_metrics.clear();
        self.metric_store.start_memory_usage = 0;
        self.metric_store.end_memory_usage = 0;
    }

    pub fn add_store_metric(
        &mut self,
        total_memory_usage_before: usize,
        total_memory_usage_after: usize,
        storage_memory_usage: usize,
        time_cost: Duration,
    ) {
        self.global_req_id += 1;
        self.metric_store
            .request_metrics
            .push(PerRequestMetricData {
                kind: RequestType::Store,
                time_cost,
                storage_memory_usage,
                total_memory_usage_before,
                total_memory_usage_after,
            });
    }

    pub fn add_query_metric(
        &mut self,
        total_memory_usage_before: usize,
        total_memory_usage_after: usize,
        storage_memory_usage: usize,
        time_cost: Duration,
    ) {
        self.global_req_id += 1;
        self.metric_store
            .request_metrics
            .push(PerRequestMetricData {
                kind: RequestType::Query,
                time_cost,
                storage_memory_usage,
                total_memory_usage_before,
                total_memory_usage_after,
            });
    }

    pub fn add_chat_metric(
        &mut self,
        total_memory_usage_before: usize,
        total_memory_usage_after: usize,
        storage_memory_usage: usize,
        time_cost: Duration,
    ) {
        self.global_req_id += 1;
        self.metric_store
            .request_metrics
            .push(PerRequestMetricData {
                kind: RequestType::Chat,
                time_cost,
                storage_memory_usage,
                total_memory_usage_before,
                total_memory_usage_after,
            });
    }

    pub fn get_metrics(&self) -> &MetricData {
        &self.metric_store
    }
}

pub fn get_current_memory() -> Option<memory_stats::MemoryStats> {
    memory_stats::memory_stats()
}

#[test]
fn test_current_memory() {
    let memory = get_current_memory();
    println!("{:?}", memory)
}
