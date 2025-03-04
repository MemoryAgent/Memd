use super::component;
use anyhow::Result;

pub async fn insert(
    doc: &component::operation::Document,
    local_comps: &mut component::LocalComponent,
) -> Result<()> {
    Ok(())
}

pub async fn query(question: &str, local_comps: &mut component::LocalComponent) -> Result<String> {
    Ok("".to_string())
}
