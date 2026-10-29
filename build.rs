//! Embeds the engine's git revision as `TAFLZERO_GIT` (`git rev-parse --short=12 HEAD`,
//! `-dirty` when tracked files differ from it, `unknown` outside git), so a run can
//! record exactly which engine produced its data (`taflzero --version`, UCI `id name`).

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let rev = match git(&["rev-parse", "--short=12", "HEAD"]) {
        Some(hash) if !hash.is_empty() => {
            let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
                .is_some_and(|s| !s.is_empty());
            if dirty { format!("{hash}-dirty") } else { hash }
        }
        _ => "unknown".to_string(),
    };
    println!("cargo:rustc-env=TAFLZERO_GIT={rev}");

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
