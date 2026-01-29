use anyhow::Result;
use tokenizers::Tokenizer;

use crate::models::chat::{ContentPart, Message, MessageRole};

/// Encodes chat messages into a Qwen-style ChatML prompt, while also collecting
/// multimodal placeholders (image spans).
///
/// This is intentionally minimal: it is enough to drive the model and is compatible
/// with both CLI prompts and the OpenAI-like API request.
pub struct PromptEncoder {
    pub image_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,

    pub im_start_token_id: u32,
    pub im_end_token_id: u32,
}

/// A placeholder span for where image features should be injected into token embeddings.
#[derive(Debug, Clone)]
pub struct ImageSpan {
    /// Start index in the token sequence (inclusive).
    pub start: usize,
    /// End index in the token sequence (exclusive).
    pub end: usize,
}

impl PromptEncoder {
    pub fn from_tokenizer(
        tokenizer: &Tokenizer,
        image_token_id: u32,
        vision_start_token_id: u32,
        vision_end_token_id: u32,
    ) -> Result<Self> {
        let im_start_token_id = tokenizer
            .token_to_id("<|im_start|>")
            .ok_or_else(|| anyhow!("tokenizer missing <|im_start|>"))?;
        let im_end_token_id = tokenizer
            .token_to_id("<|im_end|>")
            .ok_or_else(|| anyhow!("tokenizer missing <|im_end|>"))?;
        Ok(Self {
            image_token_id,
            vision_start_token_id,
            vision_end_token_id,
            im_start_token_id,
            im_end_token_id,
        })
    }

    fn encode_text(tokenizer: &Tokenizer, s: &str) -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(s, false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec())
    }

    fn push_chat_header(&self, tokenizer: &Tokenizer, ids: &mut Vec<u32>, role: &str) -> Result<()> {
        ids.push(self.im_start_token_id);
        ids.extend(Self::encode_text(tokenizer, role)?);
        ids.extend(Self::encode_text(tokenizer, "\n")?);
        Ok(())
    }

    fn push_chat_footer(&self, tokenizer: &Tokenizer, ids: &mut Vec<u32>) -> Result<()> {
        ids.push(self.im_end_token_id);
        ids.extend(Self::encode_text(tokenizer, "\n")?);
        Ok(())
    }

    /// Encode messages into input_ids, returning also the image spans where the `image_token_id`
    /// placeholders were inserted.
    ///
    /// `image_token_counts` provides, for each image part encountered in order, how many
    /// placeholder tokens to insert (these should match the number of visual feature tokens).
    pub fn encode(
        &self,
        tokenizer: &Tokenizer,
        messages: &[Message],
        image_token_counts: &[usize],
    ) -> Result<(Vec<u32>, Vec<ImageSpan>)> {
        let mut ids: Vec<u32> = vec![];
        let mut spans: Vec<ImageSpan> = vec![];
        let mut img_idx = 0usize;

        for m in messages {
            let role_str = match m.role {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
            };

            self.push_chat_header(tokenizer, &mut ids, role_str)?;

            match &m.content {
                crate::models::chat::MessageContent::Text(s) => {
                    ids.extend(Self::encode_text(tokenizer, s)?);
                }
                crate::models::chat::MessageContent::Parts(parts) => {
                    for part in parts {
                        match part {
                            ContentPart::Text { text } => {
                                ids.extend(Self::encode_text(tokenizer, text)?);
                            }
                            ContentPart::ImageBase64 { .. } | ContentPart::ImageUrl { .. } => {
                                let n = *image_token_counts
                                    .get(img_idx)
                                    .ok_or_else(|| anyhow!("missing image_token_count for image #{img_idx}"))?;
                                img_idx += 1;

                                ids.push(self.vision_start_token_id);
                                let start = ids.len();
                                ids.extend(std::iter::repeat(self.image_token_id).take(n));
                                let end = ids.len();
                                ids.push(self.vision_end_token_id);
                                spans.push(ImageSpan { start, end });
                            }
                        }
                    }
                }
            }

            self.push_chat_footer(tokenizer, &mut ids)?;
        }

        // Add the start of an assistant message for the model to complete.
        self.push_chat_header(tokenizer, &mut ids, "assistant")?;

        Ok((ids, spans))
    }
}

