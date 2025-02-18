use candle_core::Device;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};

use anyhow::{Error as E, Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

pub fn build_model_and_tokenizer(
    model_id: Option<String>,
    revision: Option<String>,
) -> Result<(BertModel, Tokenizer)> {
    let device = Device::Cpu;
    let default_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
    let default_revision = "refs/pr/21".to_string();
    let (model_id, revision) = match (model_id.to_owned(), revision.to_owned()) {
        (Some(model_id), Some(revision)) => (model_id, revision),
        (Some(model_id), None) => (model_id, "main".to_string()),
        (None, Some(revision)) => (default_model, revision),
        (None, None) => (default_model, default_revision),
    };

    let repo = Repo::with_revision(model_id, RepoType::Model, revision);
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json")?;
        let tokenizer = api.get("tokenizer.json")?;
        let weights = api.get("model.safetensors")?;
        (config, tokenizer, weights)
    };
    let config = std::fs::read_to_string(config_filename)?;
    let mut config: Config = serde_json::from_str(&config)?;
    config.hidden_act = HiddenAct::GeluApproximate;

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
    let model = BertModel::load(vb, &config)?;
    Ok((model, tokenizer))
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_all()?.sqrt()?)?)
}

pub fn encode_single_sentence(
    s: &str,
    tokenizer: &mut Tokenizer,
    model: &BertModel,
) -> Result<Tensor> {
    let device = &model.device;
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    let tokens = tokenizer
        .encode(s, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    let token_type_ids = token_ids.zeros_like()?;
    let embedding = model
        .forward(&token_ids, &token_type_ids, None)
        .map_err(E::msg)?;
    let pooled_embedding = embedding.sum((0, 1))? / (tokens.len() as f64);
    let normalized_embedding = normalize_l2(&pooled_embedding?);
    normalized_embedding.map_err(E::msg)
}

pub fn encode_sentence(
    sentences: &Vec<String>,
    tokenizer: &mut Tokenizer,
    model: &BertModel,
) -> Result<Vec<Tensor>> {
    let device = &model.device;

    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }

    let tokens = tokenizer
        .encode_batch(sentences.to_vec(), true)
        .map_err(E::msg)?;
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), device)?)
        })
        .collect::<Result<Vec<_>>>()?;
    let attention_mask = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_attention_mask().to_vec();
            Ok(Tensor::new(tokens.as_slice(), device)?)
        })
        .collect::<Result<Vec<_>>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)?;
    let attention_mask = Tensor::stack(&attention_mask, 0)?;
    let token_type_ids = token_ids.zeros_like()?;

    let embeddings = model
        .forward(&token_ids, &token_type_ids, Some(&attention_mask))
        .map_err(E::msg)?;
    let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
    let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;

    let mut batch_embeddings = vec![];
    for i in 0..sentences.len() {
        batch_embeddings.push(embeddings.get(i)?);
    }
    Ok(batch_embeddings)
}

#[test]
fn test_encode_sentence() {
    let (bert, mut tokenizer) = build_model_and_tokenizer(None, None).unwrap();
    let sentences = vec![
            "CONTEXT One key target of the United Nations Millennium Development goals is to reduce the prevalence of underweight among children younger than 5 years by half between 1990 and 2015. OBJECTIVE To estimate trends in childhood underweight by geographic regions of the world. DESIGN, SETTING, AND PARTICIPANTS Time series study of prevalence of underweight, defined as weight 2 SDs below the mean weight for age of the National Center for Health Statistics and World Health Organization (WHO) reference population. National prevalence rates derived from the WHO Global Database on Child Growth and Malnutrition, which includes data on approximately 31 million children younger than 5 years who participated in 419 national nutritional surveys in 139 countries from 1965 through 2002. MAIN OUTCOME MEASURES Linear mixed-effects modeling was used to estimate prevalence rates and numbers of underweight children by region in 1990 and 2015 and to calculate the changes (ie, increase or decrease) to these values between 1990 and 2015. RESULTS Worldwide, underweight prevalence was projected to decline from 26.5% in 1990 to 17.6% in 2015, a change of -34% (95% confidence interval [CI], -43% to -23%). In developed countries, the prevalence was estimated to decrease from 1.6% to 0.9%, a change of -41% (95% CI, -92% to 343%). In developing regions, the prevalence was forecasted to decline from 30.2% to 19.3%, a change of -36% (95% CI, -45% to -26%). In Africa, the prevalence of underweight was forecasted to increase from 24.0% to 26.8%, a change of 12% (95% CI, 8%-16%). In Asia, the prevalence was estimated to decrease from 35.1% to 18.5%, a change of -47% (95% CI, -58% to -34%). Worldwide, the number of underweight children was projected to decline from 163.8 million in 1990 to 113.4 million in 2015, a change of -31% (95% CI, -40% to -20%). Numbers are projected to decrease in all subregions except the subregions of sub-Saharan, Eastern, Middle, and Western Africa, which are expected to experience substantial increases in the number of underweight children. CONCLUSIONS An overall improvement in the global situation is anticipated; however, neither the world as a whole, nor the developing regions, are expected to achieve the Millennium Development goals. This is largely due to the deteriorating situation in Africa where all subregions, except Northern Africa, are expected to fail to meet the goal.".to_string(),
 "BACKGROUND Homocysteine is a risk factor for coronary artery disease (CAD), although a causal relation remains to be proven. The importance of determining direct causality rests in the fact that plasma homocysteine can be safely and inexpensively reduced by 25% with folic acid. This reduction is maximally achieved by doses of 0.4 mg/d. High-dose folic acid (5 mg/d) improves endothelial function in CAD, although the mechanism is controversial. It has been proposed that improvement occurs through reduction in total (tHcy) or free (non-protein bound) homocysteine (fHcy). We investigated the effects of folic acid on endothelial function before a change in homocysteine in patients with CAD. METHODS AND RESULTS A randomized, placebo-controlled study of folic acid (5 mg/d) for 6 weeks was undertaken in 33 patients. Endothelial function, assessed by flow-mediated dilatation (FMD), was measured before, at 2 and 4 hours after the first dose of folic acid, and after 6 weeks of treatment. Plasma folate increased markedly by 1 hour (200 compared with 25.8 nmol/L; P<0.001). FMD improved at 2 hours (83 compared with 47 microm; P<0.001) and was largely complete by 4 hours (101 compared with 51 microm; P<0.001). tHcy did not significantly differ acutely (4-hour tHcy, 9.56 compared with 9.79 micromol/L; P=NS). fHcy did not differ at 3 hours but was slightly reduced at 4 hours (1.55 compared with 1.78 micromol/L; P=0.02). FMD improvement did not correlate with reductions in either fHcy or tHcy at any time. CONCLUSIONS These data suggest that folic acid improves endothelial function in CAD acutely by a mechanism largely independent of homocysteine.".to_string(),
 "OBJECTIVES To carry out a further survey of archived appendix samples to understand better the differences between existing estimates of the prevalence of subclinical infection with prions after the bovine spongiform encephalopathy epizootic and to see whether a broader birth cohort was affected, and to understand better the implications for the management of blood and blood products and for the handling of surgical instruments. DESIGN Irreversibly unlinked and anonymised large scale survey of archived appendix samples. SETTING Archived appendix samples from the pathology departments of 41 UK hospitals participating in the earlier survey, and additional hospitals in regions with lower levels of participation in that survey. SAMPLE 32,441 archived appendix samples fixed in formalin and embedded in paraffin and tested for the presence of abnormal prion protein (PrP). RESULTS Of the 32,441 appendix samples 16 were positive for abnormal PrP, indicating an overall prevalence of 493 per million population (95% confidence interval 282 to 801 per million). The prevalence in those born in 1941-60 (733 per million, 269 to 1596 per million) did not differ significantly from those born between 1961 and 1985 (412 per million, 198 to 758 per million) and was similar in both sexes and across the three broad geographical areas sampled. Genetic testing of the positive specimens for the genotype at PRNP codon 129 revealed a high proportion that were valine homozygous compared with the frequency in the normal population, and in stark contrast with confirmed clinical cases of vCJD, all of which were methionine homozygous at PRNP codon 129. CONCLUSIONS This study corroborates previous studies and suggests a high prevalence of infection with abnormal PrP, indicating vCJD carrier status in the population compared with the 177 vCJD cases to date. These findings have important implications for the management of blood and blood products and for the handling of surgical instruments.".to_string(),
 "Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated. It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites. We propose as an alternative explanation that variants much less common than the associated one may create synthetic associations by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele. Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings. Here, we use simple computer simulations to show the conditions under which such synthetic associations will arise and how they may be recognized. We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies. We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible for both hearing loss and sickle cell anemia create genome-wide significant synthetic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of blocks of associated variants. In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.".to_string(),
 "Nanotechnologies are emerging platforms that could be useful in measuring, understanding, and manipulating stem cells. Examples include magnetic nanoparticles and quantum dots for stem cell labeling and in vivo tracking; nanoparticles, carbon nanotubes, and polyplexes for the intracellular delivery of genes/oligonucleotides and protein/peptides; and engineered nanometer-scale scaffolds for stem cell differentiation and transplantation. This review examines the use of nanotechnologies for stem cell tracking, differentiation, and transplantation. We further discuss their utility and the potential concerns regarding their cytotoxicity.".to_string()
        ];
    let embeddings = encode_sentence(&sentences, &mut tokenizer, &bert).unwrap();
    for e in embeddings {
        println!("{:?}", e.shape())
    }
}
