# Dirtybird-CUDA-Miner

CUDA **AstroBWTv3** miner for **DERO** — a GPU implementation of the recovered ("astronv-class")
fast path, with a **0% dev fee**.

**Status:** v0.1.0 · NVIDIA GPU (CUDA, `sm_86` / `sm_89` / `sm_90`) · Linux + Windows-via-WSL2

- **Beats astronv on the same hardware.** On an RTX 4070 Laptop it sustains **~12.7–13.0 KH/s live**,
  vs the closed astronv miner's ~12.53 KH/s — and at **0% fee** vs astronv's 4.9%, so your net
  payout is higher still.
- **GPU suffix-array + exact-tiny refine** with an LCP-aware tiny-segment sorter.
- **Auto batch-size detection** (`--auto-batch`) that finds the fastest safe batch per GPU and caches it.
- **Correctness-gated:** every build verifies the GPU output against a CPU AstroBWT oracle
  (`--verify-recovered-gpu-parity`, byte-identical) before it will run.
- **astronv-style live readout:** banner, per-GPU stats (HR / power / clock / temp / efficiency), and
  a `Mined N (total)` miniblock counter.

> Honest note: raw GPU throughput is roughly at parity with astronv on this hardware
> (~12.5–13 KH/s, thermal-bound on a laptop); the decisive edge is the **0% fee** and the open,
> tunable source. A CPU miner (see the other Dirtybird repos) can exceed this on a strong CPU.

## Requirements

- NVIDIA GPU with ~8 GB+ VRAM (Ada `sm_89` default; Ampere `sm_86` / Hopper `sm_90` supported).
- CUDA toolkit (`nvcc`), `gcc`, and OpenSSL dev headers (`libssl-dev`).
- On Windows: run inside **WSL2** with the CUDA-on-WSL driver.

## Build

```bash
bash build.sh
# override GPU arch if needed:   CUDA_ARCH=sm_86 bash build.sh
# build without a GPU (skips the parity/speed gate):   SKIP_SELFTEST=1 bash build.sh
```

`build.sh` finds `nvcc` on `PATH` (or via `$CUDA_HOME`), links against OpenSSL, builds to
`./bin/openastronv_v3`, and **gates** the result on a CPU-oracle parity check plus a ≥8 KH/s speed
floor before deploying. A CMake path is also provided:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build -j
```

## Usage

```bash
./bin/openastronv_v3 -d <pool_host:port> -w <your_dero_wallet> --fast-gpu --auto-batch
```

| flag | meaning |
|------|---------|
| `-d host:port` | pool / daemon getwork endpoint |
| `-w WALLET` | DERO wallet address (your payout address) |
| `--worker NAME` | worker name appended to the pool path |
| `--fast-gpu` | use the fast recovered GPU engine (recommended) |
| `--auto-batch` | auto-detect the best batch size for this GPU (cached per GPU) |
| `-b N` | force a specific batch size (overrides `--auto-batch`) |
| `--color auto\|always\|never` | colored output (default: auto) |
| `--status-interval N` | seconds between status lines (default: 10) |
| `--quiet` | suppress periodic status lines |
| `--bench [N]` | offline GPU throughput benchmark (no pool) |
| `--verify-recovered-gpu-parity` | verify GPU output vs the CPU oracle |
| `-h`, `--help` | usage |

### Windows launcher

`run-gpu-miner.bat` is a minimal auto-restart loop that runs the WSL binary. It defaults to the
**Dirtybird community pool** and a community wallet at 0% fee — **change the `-w` address to your
own DERO wallet** to mine to yourself. If your WSL distro is not the default, add `-d <YourDistro>`
to the `wsl.exe` call.

## Correctness

The GPU suffix-array + hash output is checked byte-for-byte against an independent CPU AstroBWT
implementation:

```bash
./bin/openastronv_v3 --verify-recovered-gpu-parity   # all vectors must report OK
```

This is the build gate and should be run after any change.

## Performance (RTX 4070 Laptop GPU)

| metric | value |
|---|---|
| Live hashrate | ~12.7–13.0 KH/s (`--fast-gpu`, batch 4096, 0% fee) |
| `--bench 40` (synthetic) | ~12.7 KH/s |
| astronv reference | ~12.53 KH/s @ 4.9% fee |

`--auto-batch` lands on batch 4096 on an 8 GB card (the fast zone below the exact-tiny resource
cliff). Larger batches can trip a slow fallback on this engine; the auto-detect and the default
stay below it.

## License

MIT — see [LICENSE](LICENSE). Vendored third-party code (libsais, Apache-2.0) is documented in
[THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md).
