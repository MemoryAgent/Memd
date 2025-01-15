use anyhow::Result;

use crate::{bert::encode_prompt, ServeMode};

pub fn chat_local(question: &str, local_state: &mut ServeMode) -> Result<String> {
    match local_state {
        ServeMode::LOCAL {
            tokenizer,
            bert,
            db,
        } => {
            let encoded = encode_prompt(question, tokenizer, bert)?;
            let mem = db.query(&encoded)?;
            db.add(&encoded, question);
            Ok(format!("you said {}", mem.text))
        }
        _ => todo!("this is unreachable..."),
    }
}
