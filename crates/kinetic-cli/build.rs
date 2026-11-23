//! The version `kinetic --version` prints: the package version, without a
//! patch number of 0 (0.1, not 0.1.0), and after it what `KINETIC_BUILD` says
//! about the build, such as a nightly's date and commit.

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=KINETIC_BUILD");
    let version = env::var("CARGO_PKG_VERSION").expect("cargo sets CARGO_PKG_VERSION");
    let version = match version.rsplit_once('.') {
        Some((minor, "0")) => minor.to_owned(),
        _ => version,
    };
    let version = match env::var("KINETIC_BUILD") {
        Ok(build) if !build.trim().is_empty() => format!("{version} ({})", build.trim()),
        _ => version,
    };
    println!("cargo:rustc-env=KINETIC_VERSION={version}");
}
