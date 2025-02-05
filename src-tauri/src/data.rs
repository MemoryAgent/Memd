#[derive(Debug, Clone)]
pub struct Document {

}

#[derive(Debug, Clone)]
pub struct Entity {

}

#[derive(Debug, Clone)]
pub struct Relation {
    subject: String,
    relation: String,
    object: String,
}

impl Relation {
    pub fn parse(s: &str) -> Self {
        let xs: Vec<String> = s.split(',').map(|x| x.to_string()).collect();
        Relation {
            subject: xs[0].clone(),
            relation: xs[1].clone(),
            object: xs[2].clone(),
        }
    }
}
