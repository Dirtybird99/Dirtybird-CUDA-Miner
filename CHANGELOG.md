# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com); this project uses semantic
versioning.

## v0.1.6 — 2026-08-11

- Added Blackwell / RTX 50-series (`sm_120`) support and raised the build toolkit to CUDA 12.9.
- `build.sh` now emits PTX for the newest architecture in `CUDA_ARCH`. It previously passed only
  `code=sm_N`, which is SASS with no virtual target, so the Linux and HiveOS binaries could not run
  on any GPU newer than the newest one compiled in — the driver found no kernel image and the miner
  died at startup. Confirmed with `cuobjdump --list-ptx` against the published v0.1.5 asset, which
  reported "No PTX file found"; the Windows binaries were unaffected because a bare entry in
  `CMAKE_CUDA_ARCHITECTURES` already emits both real and virtual.
- Stayed on the CUDA 12.x line deliberately. CUDA 13 is not published for Ubuntu 20.04, and the
  HiveOS artifact has to build in a focal container to hold its glibc 2.31 floor; 12.x also keeps
  minor-version compatibility so the driver requirement on existing rigs does not move.

## v0.1.5 — 2026-08-11

- Built the amd64 packages inside an Ubuntu 20.04 container, putting the glibc floor at 2.31.
  Built on 22.04, the binary and its bundled libssl/libcrypto/libstdc++ demanded GLIBC_2.32+ and a
  HiveOS rig refused to load them; the previous release notes recorded that floor as a limitation
  rather than a defect. The desktop and WSL packages come off the same nvcc link and inherit it.
- Replaced the hardcoded `libssl.so.3` bundling list with an ldd-driven one. Focal ships OpenSSL
  1.1.1, so the named list would have failed the very build that lowers the floor; the soname is
  now discovered, and each expected library is still asserted present.
- Read the SHA feature bit from CPUID instead of `__builtin_cpu_supports("sha")`, which GCC did not
  accept until 11. This also unblocks building from source on distros at the supported floor.
- Rewrote the HiveOS stats hook. It reported rig uptime rather than miner uptime, took the hashrate
  from whatever the last `KH/s` line happened to be — the per-GPU TOTAL row, correct only by
  coincidence — and re-read the whole log on every poll. It now anchors on the status line's labels,
  emits real per-GPU `hs[]` and `temp[]` arrays and a `ver` field, and reports zeros rather than
  `null` before the first status line. `mmp-stats.sh` shares the parser instead of duplicating it.
- Added `hiveos/test-h-stats.sh`, run in CI and against the packaged copy during release, and a
  release gate that executes the packaged tarball inside the 20.04 container through its
  `$ORIGIN/lib` rpath alone.

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
