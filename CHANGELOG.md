# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com); this project uses semantic
versioning.

## v0.1.4 — 2026-08-10

- Made the faster narrow comparator the recovered-engine default; RTX 4070 ABBA testing measured a
  1.51% gain over wide mode, with a 12.912 KH/s 20-iteration result and 12.908 KH/s live mean.
- Kept the safe 4096 default and auto-tune fallback below the 8 GB resource cliff.
- Added CUDA 13.3 and native MSVC build compatibility plus a native Windows x64 release asset.
- Expanded release fatbins to Ampere, Ada, and Hopper, and hardened releases with signed tags,
  protected tag names, immutable assets, checksums, and pre-publication smoke tests.

## v0.1.3 — 2026-06-28

Internal cleanup and documentation. No change in behavior or performance: the GPU output is identical
(parity-green) and the hashrate is unchanged from v0.1.2.

- Removed dead GPU kernels and the unused 32-bit suffix-array path; the source is smaller with no
  functional change.
- Rewrote the README in a plainer style and added this changelog.

## v0.1.2 — 2026-06-23

- Changed the default pool to the Dirtybird community pool.
- Verified the HiveOS/MMPOS package statistics on a local rig.
- The arm64 and HiveOS packages bundle their runtime libraries with an `$ORIGIN/lib` rpath, so they
  start on a bare rig.

## v0.1.1 — 2026-06-23

- arm64 source portability: guarded the x86 CPU intrinsics and generalized the build for multiple
  architectures.
- Added experimental arm64 (Jetson `sm_87`, GH200 `sm_90`) and HiveOS/MMPOS packages.

## v0.1.0 — 2026-06-23

- First release. CUDA AstroBWTv3 miner for DERO, recovered fast path, 0% fee.
- Linux x86-64 and Windows-via-WSL2 builds, verified at CPU-oracle parity on an RTX 4070.
