//! Embeds the engine's git revision as `TAFLZERO_GIT` (`git rev-parse --short=12 HEAD`,
//! `-dirty` when tracked files differ from it; in a crate from crates.io, outside git,
//! the commit `cargo package` recorded in `.cargo_vcs_info.json`; `unknown` without
//! either), so a run can record exactly which engine produced its data
//! (`taflzero --version`, UCI `id name`).

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The commit a published crate was packaged from (`"sha1": "<hash>"` in the
/// `.cargo_vcs_info.json` that `cargo package` writes), cut to 12 characters.
fn packaged_rev() -> Option<String> {
    let dir = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let info =
        std::fs::read_to_string(std::path::Path::new(&dir).join(".cargo_vcs_info.json")).ok()?;
    let at = info.find("\"sha1\"")?;
    let hash: String = info[at + 6..]
        .chars()
        .skip_while(|c| !c.is_ascii_hexdigit())
        .take_while(|c| c.is_ascii_hexdigit())
        .collect();
    (hash.len() >= 12).then(|| hash[..12].to_string())
}

fn main() {
    let rev = match git(&["rev-parse", "--short=12", "HEAD"]) {
        Some(hash) if !hash.is_empty() => {
            let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
                .is_some_and(|s| !s.is_empty());
            if dirty { format!("{hash}-dirty") } else { hash }
        }
        _ => packaged_rev().unwrap_or_else(|| "unknown".to_string()),
    };
    println!("cargo:rustc-env=TAFLZERO_GIT={rev}");

    // `dml`: the `directml` feature on Windows, the only platform ONNX Runtime has
    // DirectML for. Elsewhere the feature is a no-op and the engine never names it.
    println!("cargo::rustc-check-cfg=cfg(dml)");
    let windows = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "windows");
    if windows && std::env::var_os("CARGO_FEATURE_DIRECTML").is_some() {
        println!("cargo:rustc-cfg=dml");
    }

    // Rebuild when the sources change (dirty state) or HEAD moves (a commit, a checkout).
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=build.rs");
    if let Some(dir) = git(&["rev-parse", "--git-dir"]) {
        println!("cargo:rerun-if-changed={dir}/HEAD");
        println!("cargo:rerun-if-changed={dir}/index");
    }
    if let Some(head_ref) = git(&["symbolic-ref", "-q", "HEAD"])
        && let Some(common) = git(&["rev-parse", "--git-common-dir"])
    {
        println!("cargo:rerun-if-changed={common}/{head_ref}");
    }
}
