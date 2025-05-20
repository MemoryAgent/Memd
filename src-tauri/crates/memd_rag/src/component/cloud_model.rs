//! use openai sdk to access deepseek model
//!

use anyhow::Result;
use openai_api_rust::{
    chat::{ChatApi, ChatBody},
    Auth, Message, OpenAI,
};

use super::llm::LLM;

const API_ENDPOINT: &str = "https://api.deepseek.com";

const MODEL_NAME: &str = "deepseek-chat";

#[derive(Debug)]
pub struct CloudLLM {
    openai: OpenAI,
}

impl CloudLLM {
    pub fn new(api_key: &str) -> CloudLLM {
        let auth = Auth::new(api_key);
        let openai = OpenAI::new(auth, API_ENDPOINT);
        Self { openai }
    }

    pub fn complete(&self, prompt: &str) -> Result<String> {
        let body = ChatBody {
            model: API_ENDPOINT.to_string(),
            messages: vec![Message {
                role: openai_api_rust::Role::User,
                content: prompt.to_string(),
            }],
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            max_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
        };

        let resp = self.openai.chat_completion_create(&body).unwrap();
        let choice = resp.choices;
        let message = choice[0].message.clone().unwrap().content;
        Ok(message)
    }
}

impl LLM for CloudLLM {
    fn llm_complete(&mut self, prompt: &str) -> Result<String> {
        self.complete(prompt)
    }
}
