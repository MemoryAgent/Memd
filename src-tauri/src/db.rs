use anyhow::{Ok, Result};
use candle_core::Tensor;
use tracing::info;

#[derive(Clone)]
pub struct Memory {
    pub text: String,
    embedding: Tensor,
}

pub trait VecStore {
    fn add(&mut self, tensor: &Tensor, text: &str);

    fn query(&self, prompt: &Tensor) -> Result<Memory>;
}

pub struct InMemDB(Vec<Memory>);

impl InMemDB {
    pub fn new() -> InMemDB {
        InMemDB(vec![])
    }
}

impl VecStore for InMemDB {
    fn add(&mut self, tensor: &Tensor, text: &str) {
        info!("adding {:?} {:?}, db size is {:?}", tensor.shape(), text, self.0.len());
        self.0.push(Memory {
            text: text.to_string(),
            embedding: tensor.clone(),
        });
    }

    fn query(&self, prompt: &Tensor) -> Result<Memory> {
        let mut similarities = self
            .0
            .iter()
            .zip(0_usize..)
            .flat_map(|(mem, idx)| {
                Ok((
                    idx,
                    (&mem.embedding * prompt)?.sum_all()?.to_scalar::<f32>()?,
                ))
            })
            .collect::<Vec<(usize, f32)>>();
        similarities.sort_by(|u, v| u.1.total_cmp(&v.1));
        if let Some((idx, _)) = similarities.get(0) {
            Ok(self.0.get(*idx).unwrap().clone())
        } else {
            Ok(Memory {
                text: "No memory yet".to_string(),
                embedding: Tensor::zeros(4, candle_core::DType::F32, &candle_core::Device::Cpu)
                    .unwrap(),
            })
        }
    }
}
