# Versioning

Kinetic's version is the one in the workspace `Cargo.toml`; the command
line, the libraries and the Python package all carry it, and `kinetic
--version` prints it.

- **0.1.x** releases fix bugs. They add support where it was missing (a
  pre-tokenizer, a weight type on a device) but do not change how existing
  commands, flags or API fields behave.
- **0.2** brings new features, such as macOS, older NVIDIA GPUs and
  several GPUs for one model. While Kinetic is below 1.0, a minor version
  may also change behavior; the [changelog](changelog.md) lists every such
  change.
- **Every release before 1.0 is a beta.** Releases are measured and
  announced, and their notes say what is known not to work yet.
- **Nightly builds** are main as it was last night, built every night main
  changed and published as the prerelease
  [`nightly`](https://github.com/Veloresearch/kinetic/releases/tag/nightly),
  which replaces the one before. They are untested and not announced; use
  them to try a fix before the next release. They keep the version of the
  last release, and `kinetic --version` adds the night and the commit, such
  as `0.1 (nightly 2026-11-21 f3723b1)`. The install scripts take
  them with `--nightly` / `-Nightly`. Nightlies have no Python wheels and
  are not on PyPI.
- **File formats are versioned on their own.** A `.ktc` file starts with the
  magic `KTC1` and a format version, which changes only when the layout does,
  independent of Kinetic's version. See
  [FORMAT.md](https://github.com/Veloresearch/kinetic/blob/main/crates/kinetic-ktc/FORMAT.md).
- **Releases are git tags** named `vX.Y.Z`, such as `v0.1.0`. Each tag builds
  the release archives, the Python wheels and their `SHA256SUMS`.
