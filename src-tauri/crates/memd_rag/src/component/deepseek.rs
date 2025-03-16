//! This module provides helper functions for deepseek series LLM
//!
//! A typical Deepseek R1 response is like:
//!
//! ```text
//! <think>
//! what it thinks ...
//! </think>
//!
//! answer
//!

pub(crate) const DEEPSEEK_R1_1B: &str =
    "lmstudio-community/DeepSeek-R1-Distill-Qwen-1.5B-GGUF~DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf";

/// Deepseek R1's answers contains two parts
///
/// <think>
/// what it thinks ...
/// </think>
///
/// answer
///
/// So it is necessary to write a function to parse these two parts
pub(crate) fn extract_answer(answer: &str) -> (&str, &str) {
    assert!(answer.starts_with("<think>\n"));
    let subview = &answer[8..];
    subview.split_once("</think>\n\n").unwrap()
}

#[test]
fn test_extract_answer() {
    let text = "<think>
I am thinking about ...
</think>

The answer is quite straight forward.
";

    let (think, answer) = extract_answer(text);
    assert_eq!(think, "I am thinking about ...\n");
    assert_eq!(answer, "The answer is quite straight forward.\n");
}
