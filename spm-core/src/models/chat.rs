use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// The role of a message in a chat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageRole {
    /// System prompt.
    #[serde(alias = "system")]
    System,
    /// User prompt.
    #[serde(alias = "user")]
    User,
    /// Assistant response.
    #[serde(alias = "assistant")]
    Assistant,
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
            }
        )
    }
}

/// A chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message role.
    pub role: MessageRole,
    /// Messagae content.
    pub content: MessageContent,
}

/// Message content, compatible with both plain-text and multi-part (multimodal) inputs.
///
/// - Legacy format: `"content": "hello"`
/// - Multi-part format (OpenAI-style): `"content": [{"type":"text","text":"hello"}, ...]`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },

    /// Base64-encoded image payload (raw base64, without data URL prefix).
    #[serde(rename = "image_base64")]
    ImageBase64 {
        /// Optional mime type, e.g. "image/png" or "image/jpeg".
        #[serde(default)]
        media_type: Option<String>,
        /// Base64 data, without prefix.
        data: String,
    },

    /// OpenAI-style image URL part. This also supports data URLs such as
    /// "data:image/png;base64,....".
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

impl MessageContent {
    /// Returns a plain-text representation.
    /// Errors if the content contains non-text parts.
    pub fn to_text_strict(&self) -> Result<String> {
        match self {
            MessageContent::Text(s) => Ok(s.clone()),
            MessageContent::Parts(parts) => {
                let mut out = String::new();
                for part in parts {
                    match part {
                        ContentPart::Text { text } => out.push_str(text),
                        ContentPart::ImageBase64 { .. } | ContentPart::ImageUrl { .. } => {
                            bail!("non-text content (image) is not supported by this model")
                        }
                    }
                }
                Ok(out)
            }
        }
    }

    pub fn has_image(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Parts(parts) => parts.iter().any(|p| !matches!(p, ContentPart::Text { .. })),
        }
    }
}

impl Message {
    /// Create a system message.
    pub fn system(content: String) -> Self {
        Self {
            role: MessageRole::System,
            content: MessageContent::Text(content),
        }
    }

    /// Create a user message.
    pub fn user(content: String) -> Self {
        Self {
            role: MessageRole::User,
            content: MessageContent::Text(content),
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: String) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: MessageContent::Text(content),
        }
    }

    pub fn is_multimodal(&self) -> bool {
        self.content.has_image()
    }
}
