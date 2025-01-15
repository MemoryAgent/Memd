pub struct Document {
    pub text: String,
}

pub fn str_reader(chunk_size: usize, text: &str) -> Vec<Document> {
    text.chars()
        .collect::<Vec<char>>()
        .chunks(chunk_size)
        .map(|c| Document {
            text: c.iter().collect::<String>(),
        })
        .collect()
}
