# Builder for prototypes/polytune-argmax: the official Rust image, plus the wasm32 target, wasm-pack and
# the wasm-bindgen CLI matching the crate's pinned wasm-bindgen (0.2.129). Nothing is installed on the
# host. Polytune needs Rust 1.89 or later; this project's host Rust is 1.88, hence the container.
# Pinned by digest as well as tag, so a rebuild gets the same toolchain the measurements were taken with.
FROM rust:1.99.1-bookworm@sha256:48b7f435b8fc6378ce43b44f15e3a47fd8919b2825f835c39bd4bd3da50af5b8
RUN rustup target add wasm32-unknown-unknown \
 && cargo install --locked wasm-pack@0.15.0 \
 && cargo install --locked wasm-bindgen-cli@0.2.129 \
 && rm -rf /usr/local/cargo/registry /usr/local/cargo/git
WORKDIR /work
