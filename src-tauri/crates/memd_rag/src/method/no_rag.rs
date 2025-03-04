use anyhow::Result;

use super::component::{self, operation::Document};

pub(super) fn insert(_doc: &Document, _local_comps: &mut component::LocalComponent) -> Result<()> {
    Ok(())
}

pub(super) fn query(question: &str, local_comps: &mut component::LocalComponent) -> Result<String> {
    local_comps.llm.complete(&question)
}
