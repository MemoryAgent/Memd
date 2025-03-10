use anyhow::Result;

use super::{component::{self, operation::Document}, QueryResults};

pub(super) fn insert(_doc: &Document, _local_comps: &mut component::LocalComponent) -> Result<()> {
    Ok(())
}

pub fn query(
    _question: &str,
    _local_comps: &mut component::LocalComponent,
) -> Result<QueryResults> {
    Ok(QueryResults(Vec::new()))
}

pub(super) fn chat(question: &str, local_comps: &mut component::LocalComponent) -> Result<String> {
    local_comps.llm.complete(&question)
}
