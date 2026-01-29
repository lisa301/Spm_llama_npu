//! This module contains spm protocol specific objects and constants.

/// spm protocol header magic value.
const PROTO_MAGIC: u32 = 0x104F4C7;

/// spm protocol message max size.
const MESSAGE_MAX_SIZE: u32 = 512 * 1024 * 1024;

mod message;

pub use message::*;
