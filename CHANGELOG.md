# Changelog

All notable project changes will be documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project intends
to use semantic versioning after its first stable release.

## [Unreleased]

### Added

- Leakage-safe longitudinal data pipeline and strict validation.
- SD1.5 source conditioning with LoRA and DoRA adapters.
- Composite diffusion, identity, and age loss.
- Mixed-precision training, atomic checkpoints, and deterministic monitoring.
- Direct and DDIM-inverse face-aging inference.
- YAML configuration, CLI entry points, Docker support, and project metadata.
- Explicit age-delta MLP conditioning, relative-age supervision, and
  source/generated age-delta diagnostics.
- In-memory conditioning-isolation diagnostic for CLIP text versus age-delta control.
