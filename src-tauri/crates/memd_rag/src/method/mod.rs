pub use crate::component;
use anyhow::Result;
use tracing::info;

pub enum RAGMethods {
    HippoRAG,
    MemdAgent,
    Raptor,
    ReadAgent,
}

pub async fn add_local(
    text: Vec<String>,
    local_comps: &mut component::LocalComponent,
) -> Result<()> {
    let encoded =
        component::bert::encode_sentence(&text, &mut local_comps.tokenizer, &local_comps.bert)?;
    encoded.iter().zip(&text).for_each(|(t, txt)| {
        local_comps.cache.add(t, &txt);
    });
    Ok(())
}

pub async fn insert(
    doc: &component::operation::Document,
    local_comps: &mut component::LocalComponent,
) -> Result<()> {
    let stored_doc = local_comps.store.add_document(&doc.name)?;
    let chunks = component::operation::chunk_document(
        doc,
        stored_doc.id,
        512,
        128,
        &mut local_comps.tokenizer,
        &local_comps.bert,
    )
    .await?;
    for chunk in &chunks {
        let stored_chunk = local_comps.store.add_chunk(chunk)?;
        info!("inserted chunk {:?}", stored_chunk);

        let entities = component::operation::chunk_extract_entity(
            chunk,
            &local_comps.llm,
            &mut local_comps.tokenizer,
            &local_comps.bert,
        )
        .await?;

        if entities.is_empty() {
            continue;
        }

        let stored_entites = entities
            .iter()
            .map(|entity| local_comps.store.add_entity(entity, &stored_chunk))
            .collect::<Result<Vec<component::database::Entity>>>()?;
        info!("inserted entities {:?}", stored_entites);

        let relations =
            component::operation::chunk_extract_relation(chunk, &entities, &local_comps.llm)
                .await?;

        if relations.is_empty() {
            continue;
        }

        let entity_name_to_id = stored_entites
            .iter()
            .map(|entity| (entity.name.clone(), entity.id))
            .collect::<std::collections::HashMap<_, _>>();
        let relations = relations
            .iter()
            .map(|relation| local_comps.store.add_relation(relation, &entity_name_to_id))
            .collect::<Result<Vec<_>>>()?;
        info!("inserted relation {:?}", relations);
    }
    Ok(())
}

#[tokio::test]
async fn test_insert() {
    tracing_subscriber::fmt::init();
    let doc = component::operation::Document {
        name: "test".to_string(),
        content: "Beijing is the capital of China.".to_string(),
    };
    let mut local_comps = component::LocalComponent::default();
    insert(&doc, &mut local_comps).await.unwrap();
}

pub async fn query(question: &str, local_comps: &mut component::LocalComponent) -> Result<String> {
    let entities = local_comps.llm.extract_entities(question)?;
    let matching_entities = local_comps.store.find_entities_by_names(&entities)?;
    let relations = local_comps
        .store
        .find_relation_by_entities(&matching_entities)?;
    let all_entity_ids = component::operation::graph_search(&matching_entities, &relations).await?;
    let all_entities = local_comps.store.find_entities_by_ids(&all_entity_ids)?;
    let chunks = local_comps
        .store
        .find_chunks_by_entity_ids(&all_entity_ids)?;
    let texts: Vec<String> = chunks.iter().map(|chunk| chunk.content.clone()).collect();
    let prompt = component::llm::build_complete_prompt(
        question,
        if all_entities.len() > 0 {
            &all_entities[0].name
        } else {
            ""
        },
        &vec![],
        &texts,
    );
    local_comps.llm.complete(&prompt)
}

#[tokio::test]
async fn test_query() {
    tracing_subscriber::fmt::init();
    let doc = component::operation::Document {
        name: "test".to_string(),
        content: "Beijing is the capital of China.".to_string(),
    };
    let mut local_comps = component::LocalComponent::default();
    insert(&doc, &mut local_comps).await.unwrap();
    let question = "What is the capital of China?";
    let answer = query(question, &mut local_comps).await.unwrap();
    println!("answer: {}", answer);
}

pub async fn query_local(
    prompt: &str,
    local_comps: &mut component::LocalComponent,
) -> Result<String> {
    let encoded = component::bert::encode_single_sentence(
        prompt,
        &mut local_comps.tokenizer,
        &local_comps.bert,
    )?;
    let memory = local_comps.cache.query(&encoded)?;
    Ok(memory.text)
}

pub async fn chat_local(
    question: &str,
    local_comps: &mut component::LocalComponent,
) -> Result<String> {
    let answer = query_local(question, local_comps).await?;
    add_local(vec![question.to_string()], local_comps).await?;
    let llm_answer = local_comps.llm.complete(question)?;
    Ok(format!("based on {}, answer is {}", answer, llm_answer))
}

pub async fn upload_local() {
    todo!("later then")
}
