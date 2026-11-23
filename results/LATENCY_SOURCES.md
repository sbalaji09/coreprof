# Published latency numbers for a Groq-like dataflow chip (LoopCore)

Goal: replace the assumed per-stage costs in the LoopCore per-layer latency model with published numbers. LoopCore runs Ouro-1.4B: 24 layers, hidden size 2048, 16 heads × 128, MLP width 5632, decode at batch 1–16.

Research date: November 2026. Five parallel agents read the primary sources. Unless a row says otherwise, every quote below comes from text the agents extracted themselves (pdftotext on the PDFs, or the raw HTML).

## Summary

Published numbers exist for these quantities:
- clock
- vector width
- on-chip hop latency
- matrix-unit timing
- barrier cost
- chip-to-chip latency

No published number exists for four of the quantities the model needs most:
- vector-unit pipeline depth
- cross-lane reduction latency
- SRAM read latency on Groq
- per-layer batch-1 latency for an LLM on any dataflow chip

On a Groq-style chip at batch 1, matmul time is set by how fast weights move from SRAM into the matrix unit. For one Ouro layer in fp16 on one chip, that comes to roughly 6–11 µs. This range is my arithmetic from Groq's published install and load rates (see the per-stage table), not a measured number. It is far larger than any of the vector stages.

## Evidence labels

| Label | Meaning |
|---|---|
| **Indep** | Measured by an independent party |
| **V-meas** | Measured by the vendor |
| **Spec** | Design spec or architectural statement |
| **Est** | Estimated by the source itself |
| **Derived** | My own arithmetic on a quoted number. Not a published figure. |

## Sources

| Key | Source | URL / location | Year |
|---|---|---|---|
| A | Abts et al., "Think Fast: A Tensor Streaming Processor (TSP)…", ISCA | https://cdn.sanity.io/files/chol0sk5/production/16e7e13197dfe5f52d3b0ed62e29620662c38743.pdf (the groq.com/wp-content copy returns 404) | 2020 |
| B | Abts et al., "A Software-defined Tensor Streaming Multiprocessor…", ISCA | https://cdn.sanity.io/files/chol0sk5/production/81d0c45ed70a10f1804bbd82f2d0076af1186ff0.pdf | 2022 |
| C | Abts, Groq slides, Hot Chips 34 | https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf | 2022 |
| D | Abts, Groq slides, ISC ML HW Workshop | https://mlhardware.github.io/2020/groq.pdf | 2020 |
| E | Ahmed et al. (Groq), "Answer Fast: Accelerating BERT on the TSP" | https://arxiv.org/pdf/2206.11062 | 2022 |
| F | Bitar et al. (Groq), "SHIP: SRAM-based Huge Inference Pipelines", MLSys | https://proceedings.mlsys.org/paper_files/paper/2026/file/9c20f16b05f5e5e70fa07e2a4364b80e-Paper-Conference.pdf | 2026 |
| G | NVIDIA, "Inside NVIDIA Groq 3 LPX" | https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/ | Mar 2026 |
| H | Groq press release reporting the ArtificialAnalysis benchmark | https://groq.com/newsroom/groq-lpu-inference-engine-leads-in-first-independent-llm-benchmark | Feb 2024 |
| I | Tom's Hardware coverage of Hot Chips 2026, Groq 3 LPX (secondary source) | Aug 26 2026 | 2026 |
| J | Jia, Tillman, Maggioni, Scarpazza (Citadel), "Dissecting the Graphcore IPU…" | https://arxiv.org/pdf/1912.03413 | 2019 |
| K | Graphcore, Tile Vertex ISA IPU21 r1.3.1 | read from an archive.org copy because docs.graphcore.ai failed the TLS handshake | Dec 2022 |
| K2 | Graphcore IPU Programmer's Guide §2.3 | docs.graphcore.ai/projects/ipu-programmers-guide (archive.org copy) | — |
| K3 | Third-party IPU papers | https://arxiv.org/pdf/2304.08662 ; https://arxiv.org/pdf/2311.04417 | 2023 |
| L | Rocki et al., "Fast stencil-code computation on a wafer-scale processor", SC20 (CS-1 / WSE-1) | https://arxiv.org/abs/2010.03660 | 2020 |
| M | WaferLLM, OSDI'25 | https://arxiv.org/html/2502.04563v3 | 2025 |
| M2 | Cerebras SDK docs | https://sdk.cerebras.ai/computing-with-cerebras | — |
| AA | Cerebras press release, inference launch | cerebras.ai/press-release/cerebras-launches-the-worlds-fastest-ai-inference | Aug 2024 |
| AB | Cerebras blog, "Cerebras inference 3x faster" | cerebras.ai/blog/cerebras-inference-3x-faster | Oct 2024 |
| AC | Cerebras press release, Llama 405B | cerebras.ai/press-release/cerebras-inference-llama-405b | Nov 2024 |
| N | Brown & Arshad, Tenstorrent Blackhole | https://arxiv.org/pdf/2611.04740 | 2026 |
| O | corsix, "Tenstorrent Wormhole part 3" (reverse engineering) | https://corsix.org/content/tt-wh-part3 | Sep 2024 |
| P | tt-metal, BasicEthernetGuide.md (the vendor marks it outdated) | github.com/tenstorrent/tt-metal/blob/main/tech_reports/EthernetMultichip/BasicEthernetGuide.md | — |
| P2 | tt-metal PERF.md v0.65.0, and README at v0.53.1-rc7 | raw.githubusercontent.com/tenstorrent/tt-metal/v0.65.0/models/tt_transformers/PERF.md | — |
| P3 | tt-metal memory docs for kernel developers | docs.tenstorrent.com/tt-metal/latest/.../memory_for_kernel_developers.html | — |
| Q | SambaNova SN40L | https://arxiv.org/abs/2405.07518v2 | Nov 2024 |
| R | Jouppi et al., TPU v1, ISCA | https://arxiv.org/pdf/1704.04760 | 2017 |
| S | Jouppi et al., TPUv4i, ISCA (IEEE version is paywalled; course-hosted copy used) | https://www.cs.cmu.edu/~18742/papers/Jouppi2021.pdf | 2021 |
| T | Jouppi et al., TPU v4, ISCA | https://arxiv.org/pdf/2304.01433 | 2023 |
| BB | Meta MTIA, ISCA'25 (open access) | aisystemcodesign.github.io/papers/MTIA-ISCA25.pdf | 2025 |
| U | d-Matrix Corsair, Hot Chips 2025 slides | hc2025.hotchips.org/…/24_dMatrix_Bhoja_final.pdf | Aug 2025 |
| U2 | d-Matrix Corsair launch announcement | d-matrix.ai/announcements/d-matrix-launches-corsair-for-ai-inference-without-gpus-hbm/ | Nov 2024 |
| U3 | ServeTheHome write-up of the d-Matrix Hot Chips 2025 talk | servethehome.com | 2025 |
| EE | Etched announcement (original is 404; read via the gwern.net archive) | etched.com/announcing-etched | Jun 2024 |
| V | Jia et al., "Dissecting the NVIDIA Volta GPU…" | https://arxiv.org/pdf/1804.06826 | 2018 |
| V2 | Jia et al., Turing microbenchmarks | https://arxiv.org/pdf/1903.07486 | 2019 |
| W | Luo et al., "Benchmarking and Dissecting the Nvidia Hopper GPU…" | https://arxiv.org/pdf/2402.13499 | 2024 |
| X | "Dissecting the NVIDIA Hopper Architecture… Multiple Level Analysis" | https://arxiv.org/pdf/2501.12084 | 2025 |
| CC | Blackwell microbenchmarks | https://arxiv.org/pdf/2512.02189v3 | 2025 |
| Y | GPU synchronization microbenchmarks | https://arxiv.org/pdf/2004.05371 | 2020 |
| Z | Hazy Research, "No Bubbles" megakernel blog | https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles | May 2025 |
| MPK | Mirage MPK | https://arxiv.org/html/2512.22219v1 | 2025 |

## (1) Table

### Groq TSP / LPU

| Quantity | Value | System | Source + location | Evidence | Quote |
|---|---|---|---|---|---|
| Clock | 900 MHz | TSP v1 (14 nm) | A, abstract | Spec | "25×29 mm 14nm chip operating at a nominal clock frequency of 900 MHz" |
| Clock during the BERT runs | 900 MHz | TSP v1 | E §IV | V-meas | "The TSP was running at a 900 MHz clock" |
| Vector width | 320 elements (20 superlanes × 16) | TSP v1 | A §I; D | Spec | "each tile producing 16 elements of the 320-element maximum vector length"; "maxVL of 20x16=320 elements" |
| Vector width | 320 B | Groq 3 LPU (LP30) | G | Spec | "Compute and communication in the LPU are organized around 320-byte vectors as the unit of work." |
| Slice pipeline / stagger | 20 stages; 1 cycle per superlane | TSP v1 | A §I, §II-C | Spec | "Each functional slice implements a 20-stage vector pipeline that spans the tiles of each slice"; "The data for successive 16-element superlanes are lagging by 1 cycle" |
| Stream hop between slices | 1 cycle | TSP v1 | A §IV; C | Spec | "on each tick of the core clock, the TSP propagates stream values by one stream register hop in the direction of flow"; "Stream register = one-cycle hop" |
| Stations across the chip | 45 stations (Derived: crossing the chip ≈ 45 cycles ≈ 50 ns) | TSP v1 | C, SRF slide | Spec | "45 SRF stations across the superlane for on-chip data transfer" |
| Instruction timing model | T = N + d_func + δ(j,i) cycles; no d_func values published | TSP v1 | A §II, Eq. 4 | Spec | "d_func is the functional delay of the instruction … The transit delay, δ(j, i) is the distance (in cycles) between SRj and SRi" |
| Matrix unit (MXM) shape | 4 planes of 320×320 MACs | TSP v1 | A §III-D | Spec | "four (4) independent 320×320 … planes of multiply-accumulate (MACC) units" |
| MXM operation shape | int8 K = 320; fp16 K = 160 | TSP v1 | B §5.1; C | Spec | "[1× K]×[K × 320] sub-operations, where K=[160,320] i.e. the vector lengths of the hardware for FP16 and int8" |
| MXM time for one vector | 20-cycle load + 20-cycle execute, fully pipelined | TSP v1 | C, MXM slide | Spec | "Loads 320B x16 in 20 cycles / 20 cycle execution / Fully pipelined, N" |
| MXM throughput | one 320-wide int32 vector per cycle per plane | TSP v1 | A §III-D; E §II | Spec | "every cycle the MXM can generate a new int32 dot-product"; "producing a 320-value int32 vector each cycle" |
| MXM weight install | under 40 cycles for 409,600 weights | TSP v1 | A §I, §III-D | Spec | "the MEM can read and MXM can install 400K weights into all four 320×320 arrays in less than 40 cycles including SRAM and on-chip network transit delay" |
| Vector unit (VXM) | 16 ALUs per lane (4×4 mesh), 5,120 in total, chainable; no pipeline depth published | TSP v1 | A §III-C | Spec | "16 vector ALUs per lane"; "supports chaining together two or more vector ALUs within each lane" |
| VXM chain example | GELU = 13-ALU chain, 1 vector per cycle | TSP v1 | E §III-A | Spec (vendor design) | "We map GELU onto the VXM by building a pipelined chain of 13 ALUs. This pipelined chain produces a new result vector every clock cycle." |
| LayerNorm cost | 3·j·⌈k/320⌉·¼ + c cycles; c not given | TSP v1 | E §III-B, Eq. 10 | Spec (vendor design) | "we have a throughput of 4 physical vectors per cycle … a normalization step that requires a constant number of cycles (c)" |
| SRAM | 220 MiB in 88 slices; 1 vector read + 1 vector write per slice per cycle; no read latency given | TSP v1 | A §II-B; C | Spec | "220 MiBytes for all 88 slices"; "Can read and write one physical stream (vector) per cycle, from 2 banks" |
| SRAM bandwidth | 55 TiB/s SRAM, 20 TiB/s stream (computed at 1 GHz for illustration); 80 TB/s on the slides | TSP v1 | A Eq. 1–2; C | Spec | "80 TB/s of BW" |
| Chip-wide barrier | 35 cycles | TSP v1 | A §III-A2 | Spec | "A chip-wide barrier synchronization can be accomplished in 35 clock cycles" |
| Chip-to-chip link timing | min 209–211, mean 216.3–217.4, max 225–228 counter ticks; units and one-way vs round trip not stated | TSP v1, 7 links | B Table 2 | V-meas | "HAC latency characterization of seven (7) intra-node C2C links based on 100K iterations." e.g. "A 209 216.87 228 2.93" |
| Chip-to-chip hop | 722 ns | TSP v1, Dragonfly topology | B §5.3 | Spec / vendor | "an all-reduce across all TSPs has a pipelined network latency of 722 ns per hop ×3 hops = 2,166 ns" |
| Chip-to-chip hop | 300 ns | LPU v1 in production (SHIP) | F §4.1–4.2 | Vendor | "sub-µs hop latency: 300 ns per LPU-to-LPU hop, compared to the 1–10 µs typically observed in GPU collectives" |
| Network diameter | 3 hops up to 264 chips; 5 hops at 10,440 chips; 5 hops for a 136-LPU TP partition | TSP / LPU v1 | B §2.2; F §4.1 | Spec | "a diameter of three hops, with each hop taking only 300 ns" |
| End-to-end system latency | under 3 µs | 10,440-TSP system | B, abstract | Spec | "accessible in less than 3 microseconds of end-to-end system latency" |
| Chip-to-chip bandwidth | 235 GB/s aggregate over 11 ports | LPU v1 | F §7.1 | Spec | "provides 235 GB/s of aggregate chip-to-chip I/O bandwidth across 11 logical ports" |
| AllReduce bandwidth saturation | 50% at 32 KiB; 90% at 80 KiB | LPU v1 | F, Fig. 6 | V-meas | "50% bandwidth saturation at a tensor size of 32KiB, and 90% saturation at a tensor size of 80KiB" |
| BERT-base, batch 1 | 128.9 µs mean (p99 129.5); Derived ≈ 8.9 µs per layer | 1 TSP, int8, 12 layers, hidden 768, seq 128 (encoder, not decode). Includes PCIe; excludes tokenization. | E §IV-A | V-meas | "an average latency of 128.9 µs and a standard deviation of only 3.8 µs"; "83% of the end-to-end latency is from the encoder stack … divided equally between all the layers" |
| BERT-Large, batch 1 | 99% under 1,225 µs | 4 TSPs, includes PCIe | B §5.4 | V-meas | "99% of inferences return in under 1225 μs, with all of them returning by 1300μs" |
| ResNet50, batch 1 | 49 µs | 1 TSP | A §V | V-meas | "inference latency of only 49 μs for a single image sample" |
| Llama-2 70B decode | 241 tok/s; Derived ≈ 52 µs per layer (80 layers). Chips, precision and batch not stated. | GroqCloud API | H | Indep (ArtificialAnalysis, relayed by Groq) | "achieving throughput of 241 tokens per second"; "total time to receive 100 output tokens at 0.8 seconds" |
| Groq 3 LPU | 500 MB SRAM; 96 chip-to-chip links at 112 Gbps; 150 TB/s SRAM bandwidth; no clock or latency given | LP30 | G | Spec | "Each LPU connects through 96 C2C links running at 112 Gbps each" |
| Groq 3 chip-to-chip | 350 ns | LPX | I (press, no primary source found) | Spec, secondary | "350 ns of chip-to-chip latency" |
| Groq 3 LPX decode | 3,431 tok/s. Chip count and speculative decoding not stated. NVIDIA's on-stage figure was 10,996 tok/s. | Gemma 4 31B, 100K context, concurrency 1 | I | Indep (ArtificialAnalysis, via press) | "median of 50 sequential client requests at a concurrency of one" |

### Graphcore IPU

All GC2 numbers below were measured at 1.6 GHz on SDK 1.0.49.

| Quantity | Value | System | Source + location | Evidence | Quote |
|---|---|---|---|---|---|
| Clock | 1.6 GHz (test system) | GC2 | J §2.1 | Indep | "C2 board hosts 2 IPU processors running at 1.6 GHz" |
| Clock | 1.33 GHz (GC200); 1.85 GHz (Bow) | Mk2 / Bow | K3 (2304.08662 §2.1.1) | Spec (restated by a third party) | "run at 1.33 GHz on the GC200 and at 1.85 GHz on the BOW" |
| Tiles / SRAM | 1,216 tiles × 256 KiB (GC2); 1,472 tiles × 624 KB (GC200) | GC2, GC200 | J §1.1; K2 §2.3.1 | Spec | "1,216 processing elements called tiles … plus 256 KiB of local memory"; "each tile has 624 kilobytes of SRAM" |
| Local SRAM load | 6 cycles (3.75 ns), fixed | GC2 | J §3.1, Table 1.1 | Indep | "The latency experienced by a tile reading a value from its local memory is 6 clock cycles … does not depend on access patterns, stride, working set size" |
| SRAM latency seen by a worker | 1 execution cycle | IPU21 | K §2.9.1 | Spec | "workers experience only a single execution cycle memory latency" |
| Instruction latency | 1 cycle for most; f32div 3, f32sqrt 5, f32exp 3, f32tanh 5, f16v2exp 2 (worker cycles) | IPU21 | K §4.1.2.2, Table 4.3 | Spec | "The minimum latency for floating-point instructions is 1 cycle." |
| Instruction latency (a third party's wording) | 6 cycles | GC200 / Bow | K3 (2304.08662) | Third-party statement | "The majority of IPU instructions, including loading and storing from local tile memory, take exactly six cycles." |
| Threads | 6 worker contexts per tile, round-robin | GC2 / IPU21 | J §1.2; K §2.3 | Indep / Spec | "Each tile rotates among threads according to a static, round-robin schedule." |
| Matrix unit (AMP) throughput | 64 MAC/cycle/tile; latency described only qualitatively | GC2 / Mk2 | J §1.3; K2; K §2.10.11 | Spec | "can perform up to 64 multiply-accumulate operations per cycle" |
| Tile-to-tile, sync excluded | 37–59 ns (59–95 cycles) within a column; +1.25 ns (2 cycles) per island | GC2 | J §4.1.2 | Indep | "Data transfers within the same column take 37...59 ns (59...95 clock cycles)" |
| Tile-to-tile, worst pair across columns | 98 ns | GC2 | J §4.1.2 | Indep | "Latency is 98 ns (the highest) in transfers from Column 8 to 0" |
| Tile-to-tile, sync included | 133 ns idle; 165 ns under full load | GC2 | J Tables 4.1–4.2 | Indep | "The latency of an on-chip tile-to-tile exchange is 165 nanoseconds or lower and does not degrade under load" |
| Local copy (tile to itself) | 12 ns | GC2 | J §4.2.1 | Indep | "(12 nanoseconds) only for reference" |
| Broadcast 1 word from 1 tile | 94 ns to 1–16 tiles; 194 ns to the whole chip; 1.92 µs to 16 IPUs | GC2 | J Table 4.8 | Indep | "the entire IPU (a) 1,216 0.194 µs" |
| Gather 1 word from every tile to 1 tile | 835 ns | GC2 | J Table 4.10 | Indep | "from the entire IPU (a) 1,216 0.835" |
| Reduce (sum, one FP32 per tile, to one tile) | 1.97 µs over 1,216 tiles; 2.1 µs for 19,456 values on 1 IPU; 14.51 µs across 16 IPUs | GC2 | J Tables 4.16–4.17 | Indep | "Reducing 19,456 single-precision values on one IPU takes 2.1 microsecond." |
| All-to-all | 0.131 µs (2 tiles) to 0.552 µs (608 tiles) | GC2 | J Table 4.15 | Indep | "half IPU 608 … 0.552" |
| Inter-chip link (IPU-Link) | 0.524 µs direct; 0.633 µs on the same board; 0.779 µs indirect | GC2 | J Table 4.1 | Indep | "Off-board, direct IPU link (c) 0.524 µs" |
| Inter-chip, extra hops | +0.16 µs per hop (0.145–0.174); worst pair 1.76 µs | GC2, 16-IPU system | J §2.2 | Indep | "any addition hop costs on average 0.16 microseconds" |
| Inter-chip under load | 2.52–5.99 µs | GC2 | J Table 4.2 | Indep | "Off-board, indirect IPU link (d) 0.779 µs 5.989 µs 7.7×" |
| Host to IPU (PCIe) | at least 8.81 µs | GC2 | J §4.7.1 | Indep | "the latency to transfer data to an IPU is at least 8.81 µs" |

### Cerebras

| Quantity | Value | System | Source + location | Evidence | Quote |
|---|---|---|---|---|---|
| Fabric hop | 1 cycle (about 1 ns) | CS-1 (WSE-1) | L §I, §III | Spec (vendor authors) | "nanosecond per hop message latencies"; "The single cycle-per-hop latency of the interconnect allows us to implement the AllReduce operation in a cycle count only about 10% greater than the diameter of the system." |
| SRAM load-to-use | 1 cycle; 48 KB per core | CS-1 | L §II-A | Spec | "Local memory is 48 KB … The load-to-use latency is one cycle." |
| SIMD width / multiply-accumulate rate | 4-way fp16; 2 FMAC per core per cycle in mixed precision | CS-1 | L §II-A | Spec | "the throughput is two FMACs per core per cycle" |
| Router | 5 links per cycle; core receives 1 datum per cycle | CS-1 | L §II-A, §III | Spec | "The router can move data into and out of these five links, in parallel, on every cycle." |
| Scalar AllReduce | under 1.5 µs over about 380k cores | CS-1 | L §I | V-meas | "for scalars takes under 1.5 microseconds for a system of about 380,000 independent, interconnected processors" |
| Clock | 1.1 GHz max | WSE-2 | M §7 | Spec, as reported by an independent paper | "each with a Compute Engine (CE) operating at a maximum of 1.1 GHz" |
| Message size / channels | 32-bit wavelet per cycle; 24 colors | WSE | M2 | Vendor docs | "32-bit messages called wavelets in a single clock cycle" |
| GEMV | 1.2 µs (1×16K by 16K×16K); 2.03 µs (32K) | WSE-2 | M Table 6 | Indep | "MeshGEMV (WSE-2) Time (ms) \| 0.0012 \| 0.00203" |
| Llama3-8B decode | 2,700 / 2,502 / 2,243 tokens/s at 420², 540², 660² cores (units not stated); Derived ≈ 0.37 ms/token ≈ 11.6 µs per layer | WSE-2 | M Table 4 | Indep | "LLaMA3-8B: WaferLLM \| 2699.9 \| 2501.5 \| 2243.3" |
| Llama 3.1 8B / 70B decode | above 1,800 / 446 tok/s, 16-bit, via API; Derived ≈ 17 µs per layer for 8B | CS-3 | AA | Indep (ArtificialAnalysis, relayed) | "above 1,800 output tokens per second on Llama 3.1 8B and above 446 output tokens per second on Llama 3.1 70B" |
| Llama 3.1 70B decode | 2,100 tok/s **with speculative decoding** | CS-3 | AB | Vendor | "This release also implements speculative decoding" |
| Llama 405B | 969 tok/s; time to first token 240 ms; 128K context, 16-bit | CS-3 | AC | Vendor, citing ArtificialAnalysis | "generated 969 output tokens per second" |

### Tenstorrent

| Quantity | Value | System | Source + location | Evidence | Quote |
|---|---|---|---|---|---|
| NoC hop | 8.85–9.03 cycles | Blackhole p150 | N Table II | Indep | "NoC latency/hop \| 8.85–9.03 \| 9.00" |
| L1 load (RISC-V core) | 8.1 cycles | Blackhole p150 | N Table II | Indep | "RISC-V L1 load \| 8.098 \| 8.00" |
| Matrix-vector multiply (MVMUL) issue cost | about 1.06 cycles (issue cost, not pipeline latency) | Blackhole p150 | N Table II | Indep | "FPU MVMUL \| 1.061–1.067 \| 1.00" |
| Clock / cores | 1,350 MHz; 120 Tensix cores (12×10) | Blackhole p150 | N §IV | Indep (configuration) | "at a clock frequency of 1350 MHz" |
| L1 size | 1,464 KiB (Wormhole); 1,536 KiB (Blackhole) | both | N §II | Spec | — |
| NoC hop | 9 cycles; NoC read round trip 90 cycles over 10 tiles | Wormhole | O | Indep (reverse engineering) | "the tile-to-tile propagation delay is 9 clock cycles" |
| L1 latency (vendor wording) | "several cycles" | Tensix | P3 | Vendor docs | "a latency of several cycles" |
| Ethernet chip-to-chip | 530–620 ns one-way; about 1.1 µs round trip; about 650 ns per hop in an 8-chip ring; about 1 µs per hop at 1 KB | Wormhole | P | V-meas | "the latency to send a packet over the ethernet link is between 530 and 620 ns" |
| Decode at batch 1 (tokens/s per user) | Llama-3.2-1B: 87.8 (N150), 105.9 (N300). Llama-3.1-8B: 28.3 (N150), 33.6 (P150). bfp4 MLP / bfp8 attention weights. Derived: 1B on N150 ≈ 0.71 ms per layer. | Wormhole / Blackhole | P2 | V-meas | "\| Llama-3.2-1B \| N150 \| 79 \| 97 \| 87.8 \| 26 \|"; t/s/u = "1 / inter token latency" |

### SambaNova SN40L

| Quantity | Value | Source + location | Evidence | Quote |
|---|---|---|---|---|
| Clock | under 2 GHz | Q, §IV table | Spec | "Clock Frequency \| < 2 GHz" |
| Compute / SRAM | 1,040 compute units (PCU), 1,040 memory units (PMU); 520 MiB on-chip SRAM; 638 BF16 TFLOPS | Q §I | Spec | "520 MiB of on-chip PMU SRAM" |
| On-chip network hop latency | not given (credit-based flow control at every hop) | Q §IV-C | Spec | "subject to credit-based flow control at every hop" |
| Decode speedup from hardware orchestration | 1.4–8× | Q §VI | V-meas | "Autoregressive decoding inference benchmarks achieve a noticeable speedup of 1.4× to 8×." |
| Decode (tokens/s per user) | 1,042 (8B), 457 (70B), 129 (405B); 16 sockets, BF16, 8K sequence length; **speculative decoding on 70B and 405B**. Derived: 8B ≈ 30 µs per layer. | Q Table IV | V-meas | "Speculative decoding [55] is employed on 70B and 405B models." |

### Google TPU, Meta MTIA, d-Matrix, Etched

| Quantity | Value | System | Source + location | Evidence | Quote |
|---|---|---|---|---|---|
| Clock | 700 MHz | TPU v1 | R Table 2 | Spec | — |
| Systolic array | 256×256 MACs; B input rows take B pipelined cycles; one 256-wide partial sum per cycle | TPU v1 | R §2 | Spec | "taking B pipelined cycles to complete"; "one 256-element partial sum per clock cycle" |
| Weight tile load | 256 cycles (double-buffered) | TPU v1 | R §2 | Spec | "to hide the 256 cycles it takes to shift a tile in" |
| Systolic fill latency | not quantified | TPU v1 | R §2 | Spec | "for performance, it does worry about the latency of the unit" |
| Instruction CPI | 10–20 cycles | TPU v1 | R §2 | Spec | "typically 10 to 20" |
| Production latency (MLP0) | P99 7.0 ms at batch 200, includes host | TPU v1 | R Table 4 | V-meas | "include the server host time" |
| Clock | 1,050 MHz | TPUv4i, TPU v4 | S Table 1; T Table 4 | Spec | "Clock Rate 1050 MHz" |
| Matrix unit (MXU) critical path | ¼ of the baseline (ratio only) | TPUv4i | S §3 | Spec | "cuts the critical path through the systolic array to ¼ the latency of the baseline approach" |
| Model-switch target | under 100 µs | TPUv4i | S §2 | Design target | "(e.g., <100 us)" |
| Vector unit / memory | 4 × 128×128 MXU per core; vector unit 128 lanes × 16 ALUs; 16 MiB VMEM; 128 MiB CMEM | TPU v4 | T §2 | Spec | "a Vector Processing Unit (VPU) with 128 lanes (16 ALUs per lane)" |
| Inter-chip link (ICI) | 6 links at 50 GB/s; no latency given | TPU v4 | T Table 4 | Spec | — |
| Clock | 1.35 GHz (MTIA 2i); 800 MHz (MTIA 1) | MTIA | BB Table 2 | Spec | "Frequency 1.35GHz 800MHz" |
| Compute array / vector width | 8×8 processing elements; 64 B vector extension | MTIA 2i | BB §3.1–3.2 | Spec | "RISC-V vector extension (64B wide)" |
| SRAM | 384 KB per processing element at 1 TB/s; 256 MB shared at 2.7 TB/s; no latency given | MTIA 2i | BB Table 2 | Spec | — |
| Job launch | under 1 µs; job replacement under 0.5 µs | MTIA 2i | BB §3.3 | Spec | "launching jobs in under 1 µs and replacing jobs in less than 0.5 µs" |
| LLM decode | Llama2-7B misses the 60 ms/token target because of LPDDR bandwidth | MTIA 2i | BB §3.6 | V-meas | "the decode stage fails to generate each additional token within the latency requirement of 60ms" |
| Clock | 800 MHz at 275 W; 1.2 GHz at 550 W | Corsair | U3 | Vendor, reported by press | "275W @ 800MHz" |
| In-memory matmul size | 64×64 (MXINT8); 64×128 (MXINT4); 4-wide VLIW vector unit | Corsair | U, slides 13 and 15 | Spec | "Supports 64x64 Matrix Ops in MXINT8" |
| Chip-to-chip | 115 ns die-to-die; 400 ns through a PCIe bridge; 650 ns through a PCIe switch; 2 µs scale-out NIC | Corsair | U, slide 19 | Spec | "115ns D2D, Bridge PCIe 400ns, Switches PCIe 650ns" |
| Per-token latency | 1 ms/token (Llama3 8B); 2 ms/token (Llama3 70B); batch not stated | Corsair | U2 | Vendor projection | "60,000 tokens/second at 1 ms/token for Llama3 8B in a single server" |
| Throughput | over 500,000 tok/s per 8-chip server; FP8; no latency given | Etched Sohu | EE | Vendor claim | "One 8xSohu server can serve over 500,000 Llama 70B tokens per second." |

### GPU cross-check

All rows were measured by independent parties, except where the evidence column says otherwise.

| Quantity | Value | System | Source + location | Evidence | Quote |
|---|---|---|---|---|---|
| Clock | 1,380 MHz (V100); 1,410 MHz (A100); 1,755 MHz (H800) | — | V Table 3.1; W Table III | Spec, from the paper tables | — |
| L1 / shared memory / L2 hit | 28 / 19 / ~193 cycles | V100 | V Table 3.1 | Indep | "Hit latency 28"; "No-conflict latency 19" |
| Global memory | 375 cycles (L2 miss + TLB hit); 1,029 cycles (L2 miss + TLB miss) | V100 | V Fig. 3.2 | Indep | "the 375-cycle latency reflects an L2 cache miss latency with a TLB hit" |
| L1 / shared / L2 / global | 37.9 / 29.0 / 261.5 / 466.3 cycles | A100 | W Table IV | Indep | "Global 541.5 466.3 478.8" |
| L1 / shared / L2 / global (re-measured, different method) | 33.0 / 29.0 / 202.8–408 / 566 (A100); 32.0 / 29.0 / 264.5–502 / 656 (H800) | A100 / H800 | X Table 3 | Indep | The two papers disagree on global latency because they use different P-chase methods. |
| L2 near hit / far hit | 208 / 357 cycles (A100); 258 / 414 cycles (H800) | A100 / H800 | X Table 4 | Indep | — |
| FP32 instruction latency | 4 cycles (fp16: 6) | V100 | V Table 4.1 | Indep | "most integer and single-precision instructions have a latency of 4 cycles" |
| mma.m16n8k16 fp16 | 24.6 (A100) / 24.1 (H800) cycles | — | W Table VII | Indep | — |
| wgmma | 32 / 64 / 128 cycles for N = 64 / 128 / 256 | Hopper | CC Table V; W §IV-C | Indep | — |
| tcgen05.mma | 11.0–11.4 cycles; pipeline depth not measured | B200 | CC Table V | Indep | "pipeline depth is not directly measured" |
| SM-to-SM over the DSM network | 180–181 cycles | H800 | W §VI; X §7.1 | Indep | "SM-to-SM network latency is 180 cycles" |
| SM-to-SM through global memory | 1,110 cycles (store + load) | H800 | X §7.1 | Indep | — |
| Warp shuffle / `__syncthreads` | 22 / 22 cycles | V100 | Y Table II | Indep | — |
| Warp reduction, 32 doubles | 164 cycles | V100 | Y Table V | Indep | — |
| Grid sync | 1.43–2.21 µs at 1 block per SM | V100 | Y Fig. 5 | Indep | — |
| Kernel launch | 2.1 µs on a CUDA stream; 1.3 µs with CUDA graphs | H100 | Z | Indep (blog) | "running on a CUDA stream incurs a launch cost of about 2.1 microseconds" |
| Kernel launch, warm | about 1.08 µs | V100/P100 DGX-1 | Y Table I | Indep | — |
| Async barrier (mbarrier) | about 60 ns | B200 | Z | Indep (blog) | "requiring about 60 nanoseconds each time" |
| Llama-3.2-1B forward pass | under 1 ms (H100); under 680 µs (B200). bf16, 16 layers, batch 1, no speculative decoding. Derived ≈ 37 µs per layer on B200. | H100, B200 | Z | Indep (blog) | "under 680 microseconds per forward pass" |
| B200 forward-pass breakdown (600 µs total) | 250 µs moving activations; 200 µs RMSNorm + matrix-vector multiply; 40 µs warp sync; 80 µs setup | B200 | Z | Indep (blog) | "250 microseconds are spent storing activations, awaiting consistency, and loading them" |
| Qwen3-8B decode, batch 1 | 14.5 → 12.5 ms/token with MPK | A100 | MPK §6.3 | Indep (read via a fetch summary only, not checked in the PDF) | "reduces per-token decoding latency from 14.5 ms to 12.5 ms" |

## (2) Per-stage latency on a Groq-like chip

Setting: 900 MHz, one Ouro layer, fp16, batch 1, one chip.

Everything in this section is **Derived**. It combines the Groq primitives above with Ouro's shapes, using these assumptions:
- **Clock and vector count.** 1 cycle = 1.11 ns. A 2048-wide fp16 vector is 4,096 B, which is 13 vectors of 320 B.
- **Install capacity in fp16 (my assumption).** In fp16 a weight install holds half of 409,600 weights (K = 160, from B), which is 204,800 weights. The source gives the 409,600 figure for int8 only.
- **Install cost.** The source's "<40 cycles" (A) is a latency, not a throughput. The 20-cycle load (C) is the best floor available. I therefore use 20–40 cycles per install.
- **Batch size.** The matrix unit streams N input rows at one per cycle (C; R: "B pipelined cycles"). Going from batch 1 to batch 16 adds at most about 16 cycles per install, which is small against 20–40.
- **Weight counts per layer.** QKV 12.58M, o_proj 4.19M, gate/up 23.07M, down 11.53M, for 51.4M in total. This assumes full multi-head attention (no grouped KV heads).

| # | Stage | Range | Built from |
|---|---|---|---|
| 1 | RMSNorm | at least 75 cycles + c + cross-lane reduction (at least 0.08 µs; no upper bound) | transit ≤ 45 cycles (C), stagger 20 cycles (A), 3 passes × 13 vectors at 4 vectors/cycle ≈ 10 cycles (E). The constant c and the 320-to-1 reduction are unpublished. Proxies: CS-1 AllReduce ≈ 1.1× network diameter (L); IPU software reduction 1.97 µs (J). |
| 2 | QKV matmul | 62 installs × 20–40 = 1,240–2,480 cycles (1.4–2.8 µs) | A, B, C |
| 3 | Rotary + KV-cache write | about 75–100 cycles (about 0.1 µs) + d_func | 26 vectors at one per cycle, plus transit and stagger (A, C). SRAM writes 1 vector per cycle per slice (C). d_func is unpublished; proxy 3–6 cycles per op (K, V). |
| 4 | QK^T (context length L) | about (L/100) × 20–40 + 40 cycles. At L = 1024: 240–440 cycles (0.27–0.49 µs). | Assumes the K cache is installed into the matrix unit the same way as weights. No source describes how the TSP runs attention. |
| 5 | Softmax | at least 16L/160 vectors + 2 reductions + exp chain. At L = 1024: at least about 170 cycles + 2 reductions. | exp ≈ 3 cycles (K, proxy). Reduction latency unknown. |
| 6 | AV | same as stage 4 | same as stage 4 |
| 7 | o_proj + residual | 21 × 20–40 + about 35 = 455–875 cycles (0.5–1.0 µs) | A, C |
| 8 | RMSNorm | same as stage 1 | same as stage 1 |
| 9 | gate/up | 113 × 20–40 = 2,260–4,520 cycles (2.5–5.0 µs) | A, C |
| 10 | SiLU × mul | about 18 vectors + chain depth + 65 ≈ 100 cycles (about 0.1 µs) | GELU-style chain at one vector per cycle (E) |
| 11 | down + residual | 57 × 20–40 = 1,140–2,280 cycles (1.3–2.5 µs) | A, C |

**Per layer:**
- The four matmuls total about 5.7–11.3 µs.
- The vector and attention stages add at least about 1 µs at L = 1024.
- int8 weights halve the matmul time.
- Storage check: a TSP v1 has 220 MiB of SRAM. One layer in fp16 is about 103 MB, so one chip holds about 2 layers.

**Tensor parallelism:** if a layer is split across chips, add at least 2 all-reduces per layer. Each costs 300 ns per hop (F) to 722 ns per hop (B). Other chip-to-chip latencies for comparison:

| System | Chip-to-chip latency | Source |
|---|---|---|
| d-Matrix | 115 ns die-to-die | U |
| Graphcore IPU-Link | 524 ns | J |
| Tenstorrent Ethernet | 530–620 ns | P |

**Cross-checks.** All per-layer figures below are Derived from published per-token or end-to-end numbers.

| System | Model / setting | Per layer |
|---|---|---|
| Groq TSP | BERT-base, int8, 128-token encoder (not decode) | 8.9 µs |
| WSE-2 | Llama3-8B | 11.6 µs |
| CS-3 | Llama 3.1 8B, via API | 17 µs |
| SN40L | Llama 3.1 8B | 30 µs |
| B200 | Llama-1B | 37 µs |
| GroqCloud | Llama-2 70B, via API | 52 µs |

## (3) Not found, with the closest proxy

| Quantity | Closest proxy |
|---|---|
| Groq VXM pipeline depth / per-op d_func | IPU21: 1 cycle typical, exp 3 cycles, tanh 5 cycles (K). GPU: 4 cycles fp32, 6 cycles fp16 (V). |
| Cross-lane or cross-vector reduction latency on any dataflow chip | CS-1 AllReduce ≈ 1.1× network diameter (L); Groq LayerNorm formula with unknown c (E); IPU reduce 1.97 µs (J); V100 warp reduction 164 cycles (Y) |
| Groq SRAM read latency in cycles | IPU 6 cycles (J); CS-1 1 cycle (L); Blackhole 8.1 cycles (N) |
| Matrix-unit time to first output / systolic fill and drain | Groq "20-cycle load + 20-cycle execute" (C) is the only figure. TPU v1 describes it qualitatively. TPUv4i gives only a ¼ ratio. |
| Groq weight-install throughput (as opposed to latency) | The "<40 cycles" latency (A) |
| Groq chip-to-chip counter units, and whether one-way or round trip | 300 ns per hop (F) |
| Standalone BSP sync latency on the IPU | Inference only: 133 ns (sync included) minus 37–98 ns (sync excluded) suggests about 35–95 ns (J) |
| Per-layer batch-1 LLM decode latency from any vendor | Derived per-layer figures above. Groq SHIP normalizes its time-per-output-token results. |
| Whether any tokens/s figure includes host time or sampling | Not stated anywhere. Only Groq's BERT result (E) and TPU v1 (R) state that host or PCIe time is included. |
| Groq 3 clock, MXM shape, on-chip latencies | Only the 320 B vector (G) and 350 ns chip-to-chip from press coverage (I) |
| Graphcore Mk2/Bow latencies; WSE-2/3 hop cycles; SN40L hop latency; TPU ICI latency; Tenstorrent matrix and vector pipeline latency | The older-generation or other-vendor figures above |

## Access notes

- The groq.com/wp-content PDFs return 404; working copies are on cdn.sanity.io.
- Groq's Hot Chips 32 (2020) slides were not found; the ISC 2020 deck (D) was used instead.
- No primary slides from Groq's Hot Chips 2026 talk were found; those numbers come from press coverage.
- docs.graphcore.ai failed the TLS handshake. Graphcore documents were read from archive.org snapshots.
- The MTIA v1 ISCA'23 PDF was blocked by ACM, and no open copy was found.
- The d-Matrix BusinessWire release returned 403.
- The etched.com announcement page returns 404; it was read via the gwern.net archive.
- No published IPU transformer latency was found:
  - arXiv 2404.10730 has been withdrawn.
  - arXiv 2310.04607 was searched but not read in full.
