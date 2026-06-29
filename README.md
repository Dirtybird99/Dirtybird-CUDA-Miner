# Dirtybird-CUDA-Miner

A CUDA implementation of AstroBWTv3 for DERO. It runs the recovered ("astronv-class") fast path at a
0% fee.

Status: v0.1.3 · NVIDIA GPUs (`sm_86` / `sm_89` / `sm_90`) · Linux and Windows-via-WSL2, with arm64
and HiveOS packages.

On an RTX 4070 Laptop it sustains about 12.7–13.0 KH/s. The closed astronv miner runs at about
12.53 KH/s on the same card and takes a 4.9% fee; this miner takes none, so the payout is higher.
Raw throughput is near parity and thermal-bound on a laptop. A CPU miner can do better on a strong
CPU; see the other Dirtybird repositories.

The miner builds the suffix array on the GPU and refines tied suffixes with an LCP-aware sorter for
short segments. `--auto-batch` measures the fastest safe batch size for a card and caches it. Every
build checks the GPU output against an independent CPU AstroBWT oracle and will not run until they
agree.

## Requirements

- An NVIDIA GPU with about 8 GB of memory. The default architecture is Ada (`sm_89`); Ampere
  (`sm_86`) and Hopper (`sm_90`) are supported.
- The CUDA toolkit (`nvcc`), `gcc`, and the OpenSSL development headers (`libssl-dev`).
- On Windows, run inside WSL2 with the CUDA-on-WSL driver.

## Build

```bash
bash build.sh
CUDA_ARCH=sm_86 bash build.sh     # override the GPU architecture
SKIP_SELFTEST=1 bash build.sh     # build without a GPU; skips the parity and speed gate
```

`build.sh` finds `nvcc` on `PATH` or through `$CUDA_HOME`, links against OpenSSL, and writes
`./bin/openastronv_v3`. Before deploying it checks the binary against the CPU oracle and a floor of
8 KH/s. CMake also works:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build -j
```

## Usage

```bash
./bin/openastronv_v3 -d <pool_host:port> -w <your_dero_wallet> --fast-gpu --auto-batch
```

| flag | meaning |
|------|---------|
| `-d host:port` | pool or daemon getwork endpoint |
| `-w WALLET` | DERO payout address |
| `--worker NAME` | worker name appended to the pool path |
| `--fast-gpu` | use the recovered GPU engine (recommended) |
| `--auto-batch` | measure and cache the best batch size for this GPU |
| `-b N` | force a batch size; overrides `--auto-batch` |
| `--color auto\|always\|never` | colored output (default auto) |
| `--status-interval N` | seconds between status lines (default 10) |
| `--quiet` | suppress periodic status lines |
| `--bench [N]` | offline throughput benchmark; no pool |
| `--verify-recovered-gpu-parity` | check GPU output against the CPU oracle |
| `-h`, `--help` | usage |

### Windows

`run-gpu-miner.bat` runs the WSL binary in an auto-restart loop. It defaults to the Dirtybird
community pool and a community wallet at 0% fee. Change the `-w` address to your own DERO wallet to
mine to yourself. If your WSL distribution is not the default, add `-d <YourDistro>` to the
`wsl.exe` call.

## Correctness

The GPU suffix array and hash are compared byte for byte against an independent CPU implementation
of AstroBWT:

```bash
./bin/openastronv_v3 --verify-recovered-gpu-parity
```

All vectors must report OK. This is the build gate; run it after any change.

## Performance (RTX 4070 Laptop GPU)

| metric | value |
|---|---|
| live hashrate | ~12.7–13.0 KH/s (`--fast-gpu`, batch 4096, 0% fee) |
| `--bench 40` (synthetic) | ~12.7 KH/s |
| astronv reference | ~12.53 KH/s at 4.9% fee |

On an 8 GB card `--auto-batch` settles on batch 4096, below the resource cliff where this engine
falls back to a slower path. The default stays below the cliff.

## Releases

Releases are on the [Releases page](../../releases). CI builds the binaries on GitHub runners, which
have no GPU, so verify on your own card before relying on a build:

```bash
./openastronv_v3 --verify-recovered-gpu-parity
```

| asset | platform | status |
|---|---|---|
| `…-linux-amd64-<tag>.tar.gz` | Linux x86-64 (`sm_89`) | verified on an RTX 4070 |
| `…-windows-wsl-<tag>.zip` | Windows via WSL2 | verified |
| `…-linux-arm64-<tag>.tar.gz` | NVIDIA arm64 (Jetson `sm_87`, GH200 `sm_90`) | build-only; not verified |
| `dirtybird-cuda-miner-<tag>.hiveos_mmpos.amd64.tar.gz` | HiveOS / MMPOS | stats verified |
| `SHA256SUMS.txt` and source archives | — | — |

The HiveOS package and the arm64 tarball carry their runtime libraries (`libssl`, `libcrypto`,
`libstdc++`) in a `lib/` directory with an `$ORIGIN/lib` rpath, so the binary starts on a bare rig.
CUDA is linked statically; the NVIDIA driver supplies `libcuda`. Binaries are built on Ubuntu 22.04
(glibc 2.35); older systems should build from source. macOS is not supported, because Apple machines
have no CUDA and no NVIDIA GPU.

To cut a release, push an annotated tag:

```bash
git tag -a vX.Y.Z -m "…" && git push origin vX.Y.Z
```

See [CHANGELOG.md](CHANGELOG.md) for the release history.

## License

MIT; see [LICENSE](LICENSE). The vendored libsais (Apache-2.0) is documented in
[THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md).
