//! Compiles `cuda/kernels.cu` into one fatbin that `src/kernels.rs` embeds.
//!
//! Without nvcc an empty image is written instead, so the crate still builds
//! and `Kernels::load` reports at runtime that the kernels were not built.
//! A failing nvcc, on the other hand, fails the build.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const SOURCE: &str = "cuda/kernels.cu";

/// Machine code for each architecture from Pascal on, plus PTX for
/// compute_90 and compute_120 so the driver can compile for GPUs newer than
/// this list (PTX only runs on devices of at least its compute capability).
///
/// Pascal (sm_60, sm_61) and Volta (sm_70) have no tensor-core instructions
/// that the kernels use; there the kernels that need them are never
/// launched. CUDA 13 no longer builds for them, see [`LAST_CUDA_FOR`].
/// `KINETIC_CUDA_ARCHS` replaces this list, see [`gencodes`].
const GENCODE: &[&str] = &[
    "arch=compute_60,code=sm_60",
    "arch=compute_61,code=sm_61",
    "arch=compute_70,code=sm_70",
    "arch=compute_75,code=sm_75",
    "arch=compute_80,code=sm_80",
    "arch=compute_86,code=sm_86",
    "arch=compute_89,code=sm_89",
    "arch=compute_90,code=[sm_90,compute_90]",
    "arch=compute_120,code=[sm_120,compute_120]",
];

/// The architectures of [`GENCODE`] that a CUDA release after this one no
/// longer builds for.
const LAST_CUDA_FOR: (u32, &[&str]) = (12, &["60", "61", "70"]);

fn main() {
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDAHOSTCXX");
    println!("cargo:rerun-if-env-changed=KINETIC_CUDA_ARCHS");

    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"))
        .join("kernels.fatbin");
    let Some(nvcc) = find_nvcc() else {
        println!("cargo:warning=nvcc not found: CUDA kernels not built");
        fs::write(&out, []).expect("write empty kernel image");
        return;
    };

    let mut cmd = Command::new(&nvcc);
    cmd.args(["-fatbin", "-O3", "-std=c++17", "-o"])
        .arg(&out)
        .arg(SOURCE);
    for gencode in gencodes(&nvcc, nvcc_release(&nvcc)) {
        cmd.args(["-gencode", &gencode]);
    }
    // nvcc 12 warns that sm_60 to sm_70 are deprecated; they still build.
    cmd.arg("-Wno-deprecated-gpu-targets");
    if cfg!(windows) {
        // nvcc checks the MSVC version against a list fixed at its release;
        // newer toolsets compile these kernels fine.
        cmd.arg("-allow-unsupported-compiler");
    }
    if let Some(ccbin) = find_msvc() {
        cmd.arg("-ccbin").arg(ccbin);
    }
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("running {}: {e}", nvcc.display()));
    assert!(status.success(), "nvcc failed to compile {SOURCE}");
}

/// The `-gencode` values for the CUDA `release` that nvcc is, if known:
/// supported entries from [`GENCODE`], or from
/// `KINETIC_CUDA_ARCHS`, a list of compute capabilities such as `60;61` or
/// `61,86` that builds machine code for exactly those, with PTX for the last
/// so newer GPUs can still run it. A smaller list builds faster and makes a
/// smaller binary.
fn gencodes(nvcc: &Path, release: Option<u32>) -> Vec<String> {
    let (last, old) = LAST_CUDA_FOR;
    let too_new = release.filter(|&r| r > last);
    let Some(list) = env::var("KINETIC_CUDA_ARCHS")
        .ok()
        .filter(|s| !s.trim().is_empty())
    else {
        if let Some(r) = too_new {
            println!(
                "cargo:warning=CUDA {r} builds no kernels for Pascal and Volta (sm_60 to sm_70); CUDA {last} does"
            );
        }
        // Earlier toolkits can lack newer targets even within the same major release.
        let targets = Command::new(nvcc)
            .arg("--list-gpu-code")
            .output()
            .expect("query nvcc GPU targets");
        assert!(targets.status.success(), "nvcc --list-gpu-code failed");
        let targets = String::from_utf8_lossy(&targets.stdout);
        let codes: Vec<_> = GENCODE
            .iter()
            .filter(|code| {
                let (arch, _) = code.split_once(',').expect("gencode has a code field");
                let capability = arch.trim_start_matches("arch=compute_");
                if too_new.is_some() && old.contains(&capability) {
                    return false;
                }
                let arch = format!("sm_{capability}");
                let supported = targets.lines().any(|line| line.trim() == arch);
                if !supported {
                    println!(
                        "cargo:warning=nvcc does not support {arch}; skipping that GPU target"
                    );
                }
                supported
            })
            .map(|g| g.to_string())
            .collect();
        assert!(
            !codes.is_empty(),
            "nvcc supports none of Kinetic's GPU targets"
        );
        return codes;
    };
    let archs: Vec<&str> = list
        .split([';', ',', ' '])
        .map(|a| {
            a.trim()
                .trim_start_matches("sm_")
                .trim_start_matches("compute_")
        })
        .filter(|a| !a.is_empty())
        .collect();
    assert!(
        !archs.is_empty() && archs.iter().all(|a| a.bytes().all(|b| b.is_ascii_digit())),
        "KINETIC_CUDA_ARCHS takes compute capabilities such as \"60;61\", got {list:?}"
    );
    if let (Some(r), Some(a)) = (too_new, archs.iter().find(|a| old.contains(a))) {
        panic!(
            "KINETIC_CUDA_ARCHS asks for sm_{a}, which CUDA {r} no longer builds; use CUDA {last}"
        );
    }
    let last = archs.len() - 1;
    archs
        .iter()
        .enumerate()
        .map(|(i, a)| match i == last {
            true => format!("arch=compute_{a},code=[sm_{a},compute_{a}]"),
            false => format!("arch=compute_{a},code=sm_{a}"),
        })
        .collect()
}

/// The major CUDA release of `nvcc`, from the `release 12.8` that
/// `nvcc --version` prints. If that can't be read (a wrapper script, say),
/// the build warns and assumes a release that still builds every GPU.
fn nvcc_release(nvcc: &Path) -> Option<u32> {
    let release = Command::new(nvcc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| {
            let text = String::from_utf8_lossy(&out.stdout).into_owned();
            let (_, rest) = text.split_once("release ")?;
            rest.split('.').next()?.trim().parse().ok()
        });
    if release.is_none() {
        let (last, _) = LAST_CUDA_FOR;
        println!(
            "cargo:warning=could not read the CUDA release from `{} --version`; building for Pascal and Volta too, which needs CUDA {last} or older (set KINETIC_CUDA_ARCHS otherwise)",
            nvcc.display()
        );
    }
    release
}

/// `$NVCC`, then `$CUDA_PATH/bin`, then `nvcc` on `PATH`.
fn find_nvcc() -> Option<PathBuf> {
    if let Some(path) = env::var_os("NVCC") {
        return Some(path.into());
    }
    let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    if let Some(root) = env::var_os("CUDA_PATH") {
        let path = Path::new(&root).join("bin").join(exe);
        if path.is_file() {
            return Some(path);
        }
    }
    let on_path = Command::new(exe).arg("--version").output();
    on_path
        .is_ok_and(|o| o.status.success())
        .then(|| exe.into())
}

/// On Windows nvcc needs `cl.exe`. If it is not on `PATH` (a plain shell
/// rather than a developer prompt), the newest MSVC toolset found by vswhere
/// is passed to nvcc instead.
fn find_msvc() -> Option<PathBuf> {
    if let Some(path) = env::var_os("CUDAHOSTCXX") {
        return Some(path.into());
    }
    if !cfg!(windows) || Command::new("cl.exe").output().is_ok() {
        return None;
    }
    let program_files = env::var_os("ProgramFiles(x86)")?;
    let vswhere = Path::new(&program_files).join(r"Microsoft Visual Studio\Installer\vswhere.exe");
    let out = Command::new(vswhere)
        .args(["-latest", "-products", "*", "-property", "installationPath"])
        .args([
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        ])
        .output()
        .ok()?;
    let install = String::from_utf8(out.stdout).ok()?;
    let toolsets = Path::new(install.trim()).join(r"VC\Tools\MSVC");
    let newest = fs::read_dir(toolsets)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .max()?;
    let bin = newest.join(r"bin\Hostx64\x64");
    bin.join("cl.exe").is_file().then_some(bin)
}
