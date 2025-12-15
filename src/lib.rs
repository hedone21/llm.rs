#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_dotprod))]

pub mod backend;
pub mod core;
pub mod profile;
