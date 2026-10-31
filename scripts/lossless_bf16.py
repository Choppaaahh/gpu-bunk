#!/usr/bin/env python3
"""lossless_bf16.py -- lossless bf16/fp16 weight codec + exact XOR-delta family store + wake-time decode bench.

WHAT. A bf16 value's high byte (sign + top 7 exponent bits) has ~2.7 bits of entropy, its low byte ~8, so entropy-coding
the high byte and keeping the low byte raw removes ~1/3 of the bytes with no change to a single bit. A fine-tune stored as
`base XOR d` with d entropy-coded costs 1.6-10 bits/param instead of 16. This script is the codec (static order-0 rANS,
12-bit tables, 8 interleaved lanes, 2^18-symbol chunks, a C core compiled at run time), the container format, the
round-trip proof over real checkpoints, and the host-RAM wake-time decode benchmark. Version 2 adds two arms, both off by
default: dedup of byte-identical tensors (D) and a 9-bit exponent symbol (E). The pass bars and the test goldens were
written down before the code.

usage:
  lossless_bf16.py --self-test                      goldens G1-G9 + the v2 goldens (builds the C core in ~/.cache)
  lossless_bf16.py --roundtrip MODELS_DIR --out RESULT.json [--keep-dir DIR --keep-std NAME --keep-xd NAME] [--threads N]
        encode + decode + verify EVERY *.safetensors file of every model dir; deltas vs the refs for the index sets.
        WRITES RESULT.json (resumes a partial one: models marked done are skipped) and containers for --keep-* names
        into --keep-dir. Reads MODELS_DIR only.
  lossless_bf16.py --bench KEEP_DIR --out BENCH.json --base-dir MODELS_DIR --ref RESULT.json [--threads-grid 1,2,..] [--runs 5]
        wake-time decode throughput from RAM-resident containers into a pre-faulted buffer, sha256-verified after
        every timed run; plus a multi-threaded memcpy ceiling. WRITES BENCH.json only.
  lossless_bf16.py --summarize RESULT.json [--bench-json BENCH.json]
        the report tables under the pass bars (SIZE / DELTA / SPEED bands). Reads only; exit 1 if not EXACT.
  lossless_bf16.py --roundtrip-v2 MODELS_DIR --evalset EVALSET.json --out RESULT.json [--only MODEL] [--threads N]
        v2 arms INC / D / E / DE on every file (+ XD-* deltas for fine-tunes whose base dir is present). Checks every input
        file's sha256 against EVALSET.json first and replays the recorded v1 (INC) container bytes. WRITES RESULT.json.
  lossless_bf16.py --summarize-v2 RESULT.json        the v2 pass bars over a --roundtrip-v2 RESULT.json. Reads only.

Output contract:
  success = exit 0 + a final `RESULT: ...` line (self-test: `SELF-TEST PASS n/n`).
  any golden red / any round-trip mismatch / any lane-state error = exit 1 + `FAIL ...` / `MISMATCH ...` lines.
  bad usage = exit 2. The C core failing to build = exit 3 (`BUILD-FAILED`), never a silent pure-Python fallback.

Container (format v1, little-endian): b"BBL3283\\x01" + u64 manifest_len + manifest JSON + blob (+65,600 B pad). A file
segment is RAW bytes or a tensor; a tensor is two byte streams (hi, lo) each RAW | RANS | CONST, combined by op
PLAIN | XOR | SUB | XORCAST | SUBCAST against an optional source (the base) file. A RANS chunk = 8 u32 final states +
8 u32 lane byte counts + the lanes' 16-bit words in decode order. Decode checks every lane ends at state 2^16 with its
pointer at its lane end (a free integrity check: a flipped bit almost never survives it).

Top pitfalls (OBSERVED while building this):
 1. An earlier entropy probe globbed every *.safetensors in a dir with name-overwrite: morningtea006/affine-dog holds a stray Qwen3-8B-shaped
    4-shard set beside its 4B index set, and the probe's "6.89 b/p" compared 10 q/k_norm tensors of the 8B set. Pair tensors
    through model.safetensors.index.json, never a glob.
 2. With CPU decode the PCIe link carries DECODED bytes: the codec saves host RAM, not transfer time. Fewer PCIe bytes
    needs GPU-side decode (not built here).
 3. Our data disk was a 172 MB/s HDD: every extra pass over 117 GB cost ~11 min. One read per file.
 4. Model dir names contain "__" (Qwen__Qwen3-4B): a container named <model>__<file> splits on the LAST "__" (the first
    bench launch died on KeyError 'Qwen', 2026-10-30).
 5. gcc -O3 compiles the rANS renorm select `x = nd ? (x << 16) | w : x` into an unpredictable BRANCH (~17% taken):
    1.10 GB/s/core on our 16-core Zen 5 host. An arithmetic select gave 2.00, an x86 cmov (inline asm, portable C elsewhere) 2.53, split
    decode tables + an odd-byte fast path 2.70 (dev probes, one 3.99 GB Qwen3-4B file, 2026-10-30). Read the asm.
"""
import argparse
import ctypes
import hashlib
import heapq
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

# ------------------------------------------------------------------------------------------------ the C core
C_SRC = r"""
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

#define SCALE 12
#define MTOT (1u << SCALE)
#define LANES 8
#define RL (1u << 16)
#define CH (1u << 18)
#define SUBB 16384
#define CMAX (64u + 2u * CH + 64u)

enum { OP_PLAIN = 0, OP_XOR = 1, OP_SUB = 2, OP_XORCAST = 3, OP_SUBCAST = 4 };
enum { M_RAW = 0, M_RANS = 1, M_CONST = 2 };
enum { M_RANS9 = 3, M_PACK7 = 4 };     /* v2 E: hi slot = 9-bit symbol v >> 7 (512-symbol rANS), lo slot = v & 0x7F packed */
enum { K_RAWCOPY = 0, K_TENSOR = 1 };
#define T9W (MTOT + MTOT / 2)           /* a 9-bit decode table: MTOT u32 (freq | bias << 16) + MTOT u16 symbols */

static inline uint16_t f32_to_f16_rne(uint32_t x) {
    uint32_t sign = (x >> 16) & 0x8000u, ax = x & 0x7FFFFFFFu;
    if (ax >= 0x7F800000u) {
        if (ax == 0x7F800000u) return (uint16_t)(sign | 0x7C00u);
        return (uint16_t)(sign | 0x7E00u | ((ax >> 13) & 0x3FFu));
    }
    if (ax >= 0x477FF000u) return (uint16_t)(sign | 0x7C00u);
    if (ax < 0x38800000u) {
        if (ax < 0x33000000u) return (uint16_t)sign;
        uint32_t e = ax >> 23, m = (ax & 0x7FFFFFu) | 0x800000u, sh = 126u - e;
        uint32_t q = m >> sh, rem = m & ((1u << sh) - 1u), half = 1u << (sh - 1u);
        if (rem > half || (rem == half && (q & 1u))) q++;
        return (uint16_t)(sign | q);
    }
    uint32_t h = (ax >> 13) - (112u << 10), rem = ax & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return (uint16_t)(sign | h);
}
static inline uint16_t cast_bf16_f16(uint16_t b) { return f32_to_f16_rne((uint32_t)b << 16); }

static inline uint16_t ld16(const uint8_t* p, uint64_t i) { return (uint16_t)(p[2 * i] | (p[2 * i + 1] << 8)); }

static inline uint16_t form(int op, uint16_t a, uint16_t b) {
    switch (op) {
    case OP_XOR: return (uint16_t)(a ^ b);
    case OP_SUB: return (uint16_t)(a - b);
    case OP_XORCAST: return (uint16_t)(a ^ cast_bf16_f16(b));
    case OP_SUBCAST: return (uint16_t)(a - cast_bf16_f16(b));
    default: return a;
    }
}

/* ---------------- tiny thread pool: run fn(i, ctx) for i in [0, n) over nt threads ---------------- */
typedef void (*job_fn)(uint64_t, void*);
typedef struct { job_fn fn; void* ctx; uint64_t n; volatile uint64_t next; } pool_t;
static void* pool_worker(void* a) {
    pool_t* p = (pool_t*)a;
    for (;;) {
        uint64_t i = __atomic_fetch_add(&p->next, 1, __ATOMIC_RELAXED);
        if (i >= p->n) break;
        p->fn(i, p->ctx);
    }
    return 0;
}
static void run_pool(job_fn fn, void* ctx, uint64_t n, int nt) {
    pool_t p = {fn, ctx, n, 0};
    if (nt <= 1 || n <= 1) { pool_worker(&p); return; }
    if (nt > 256) nt = 256;
    pthread_t th[256];
    int started = 0;
    for (int t = 0; t < nt - 1; t++) if (pthread_create(&th[t], 0, pool_worker, &p) == 0) started++; else break;
    pool_worker(&p);
    for (int t = 0; t < started; t++) pthread_join(th[t], 0);
}

/* ---------------- histogram (joint 16-bit) of form(op, a, b) ---------------- */
typedef struct { const uint8_t* a; const uint8_t* b; uint64_t n; int op; uint64_t* out; uint64_t per;
                 pthread_mutex_t mu; int err; } hist_ctx;
static void hist_job(uint64_t i, void* c_) {
    hist_ctx* c = (hist_ctx*)c_;
    uint64_t s = i * c->per, e = s + c->per; if (e > c->n) e = c->n;
    uint32_t* h = (uint32_t*)calloc(65536, sizeof(uint32_t));   /* per job <= 2^24 elements: a u32 bin cannot wrap */
    if (!h) { __atomic_store_n(&c->err, 1, __ATOMIC_RELAXED); return; }
    if (c->op == OP_PLAIN) for (uint64_t k = s; k < e; k++) h[ld16(c->a, k)]++;
    else for (uint64_t k = s; k < e; k++) h[form(c->op, ld16(c->a, k), ld16(c->b, k))]++;
    pthread_mutex_lock(&c->mu);
    for (int k = 0; k < 65536; k++) c->out[k] += h[k];
    pthread_mutex_unlock(&c->mu);
    free(h);
}
int bbl_hist(const uint8_t* a, const uint8_t* b, uint64_t n, int op, uint64_t* out65536, int nt) {
    hist_ctx c; c.a = a; c.b = b; c.n = n; c.op = op; c.out = out65536; c.per = 1u << 24; c.err = 0;
    pthread_mutex_init(&c.mu, 0);
    memset(out65536, 0, 65536 * sizeof(uint64_t));
    run_pool(hist_job, &c, (n + c.per - 1) / c.per, nt);
    pthread_mutex_destroy(&c.mu);
    return c.err ? -1 : 0;
}

/* ---------------- byte stream extraction: out[i] = byte sel of form(op, a[i], b[i]) ---------------- */
typedef struct { const uint8_t* a; const uint8_t* b; uint64_t n; int op, sel; uint8_t* out; uint64_t per; } form_ctx;
static void form_job(uint64_t i, void* c_) {
    form_ctx* c = (form_ctx*)c_;
    uint64_t s = i * c->per, e = s + c->per; if (e > c->n) e = c->n;
    for (uint64_t k = s; k < e; k++) {
        uint16_t v = c->op == OP_PLAIN ? ld16(c->a, k) : form(c->op, ld16(c->a, k), ld16(c->b, k));
        c->out[k] = (uint8_t)(c->sel ? (v >> 8) : (v & 0xFF));
    }
}
int bbl_form_bytes(const uint8_t* a, const uint8_t* b, uint64_t n, int op, int sel, uint8_t* out, int nt) {
    form_ctx c = {a, b, n, op, sel, out, 1u << 22};
    run_pool(form_job, &c, (n + c.per - 1) / c.per, nt);
    return 0;
}

/* ---------------- rANS encode: chunk ci of the stream into out + ci*CMAX, its length into clen[ci] ---------------- */
typedef struct { const uint8_t* a; const uint8_t* b; uint64_t n; int op, sel; const uint16_t* freq; uint32_t cum[257];
                 uint8_t* out; uint64_t* clen; int err; } enc_ctx;
static void enc_job(uint64_t ci, void* c_) {
    enc_ctx* c = (enc_ctx*)c_;
    uint64_t s0 = ci * (uint64_t)CH, s1 = s0 + CH; if (s1 > c->n) s1 = c->n;
    uint64_t m = s1 - s0;
    uint8_t* dst = c->out + ci * (uint64_t)CMAX;
    uint16_t* tmp = (uint16_t*)malloc((m / LANES + 2) * sizeof(uint16_t));
    if (!tmp) { __atomic_store_n(&c->err, 2, __ATOMIC_RELAXED); return; }
    uint32_t st[LANES], lb[LANES];
    uint8_t* w = dst + 64;
    for (int j = 0; j < LANES; j++) {
        uint64_t mj = (m > (uint64_t)j) ? (m - j + LANES - 1) / LANES : 0;
        uint32_t x = RL; uint64_t k = 0;
        for (int64_t t = (int64_t)mj - 1; t >= 0; t--) {
            uint64_t idx = s0 + j + (uint64_t)t * LANES;
            uint16_t v = c->op == OP_PLAIN ? ld16(c->a, idx) : form(c->op, ld16(c->a, idx), ld16(c->b, idx));
            uint8_t sym = (uint8_t)(c->sel ? (v >> 8) : (v & 0xFF));
            uint32_t f = c->freq[sym];
            if (f == 0) { __atomic_store_n(&c->err, 1, __ATOMIC_RELAXED); free(tmp); return; }
            uint32_t xmax = f << (32 - SCALE);
            if (x >= xmax) { tmp[k++] = (uint16_t)x; x >>= 16; }
            x = ((x / f) << SCALE) + (x % f) + c->cum[sym];
        }
        st[j] = x; lb[j] = (uint32_t)(2 * k);
        for (uint64_t q = 0; q < k; q++) { uint16_t v = tmp[k - 1 - q]; w[0] = (uint8_t)v; w[1] = (uint8_t)(v >> 8); w += 2; }
    }
    memcpy(dst, st, sizeof st); memcpy(dst + 32, lb, sizeof lb);
    c->clen[ci] = (uint64_t)(w - dst);
    free(tmp);
}
int bbl_encode(const uint8_t* a, const uint8_t* b, uint64_t n, int op, int sel, const uint16_t* freq,
               uint8_t* out, uint64_t* clen, int nt) {
    enc_ctx c; c.a = a; c.b = b; c.n = n; c.op = op; c.sel = sel; c.freq = freq; c.out = out; c.clen = clen; c.err = 0;
    uint32_t s = 0;
    for (int k = 0; k < 256; k++) { c.cum[k] = s; s += freq[k]; }
    c.cum[256] = s;
    if (s != MTOT) return -3;
    for (int k = 0; k < 256; k++) if (freq[k] >= MTOT) return -4;
    run_pool(enc_job, &c, (n + CH - 1) / CH, nt);
    return -c.err;
}

/* ---------------- v2 E: 9-bit symbol encode (512 alphabet) + 7-bit mantissa packer ---------------- */
typedef struct { const uint8_t* a; const uint8_t* b; uint64_t n; int op; const uint16_t* freq; uint32_t cum[513];
                 uint8_t* out; uint64_t* clen; int err; } enc9_ctx;
static void enc9_job(uint64_t ci, void* c_) {
    enc9_ctx* c = (enc9_ctx*)c_;
    uint64_t s0 = ci * (uint64_t)CH, s1 = s0 + CH; if (s1 > c->n) s1 = c->n;
    uint64_t m = s1 - s0;
    uint8_t* dst = c->out + ci * (uint64_t)CMAX;
    uint16_t* tmp = (uint16_t*)malloc((m / LANES + 2) * sizeof(uint16_t));
    if (!tmp) { __atomic_store_n(&c->err, 2, __ATOMIC_RELAXED); return; }
    uint32_t st[LANES], lb[LANES];
    uint8_t* w = dst + 64;
    for (int j = 0; j < LANES; j++) {
        uint64_t mj = (m > (uint64_t)j) ? (m - j + LANES - 1) / LANES : 0;
        uint32_t x = RL; uint64_t k = 0;
        for (int64_t t = (int64_t)mj - 1; t >= 0; t--) {
            uint64_t idx = s0 + j + (uint64_t)t * LANES;
            uint16_t v = c->op == OP_PLAIN ? ld16(c->a, idx) : form(c->op, ld16(c->a, idx), ld16(c->b, idx));
            uint32_t sym = (uint32_t)(v >> 7), f = c->freq[sym];
            if (f == 0) { __atomic_store_n(&c->err, 1, __ATOMIC_RELAXED); free(tmp); return; }
            uint32_t xm9 = f << (32 - SCALE);
            if (x >= xm9) { tmp[k++] = (uint16_t)x; x >>= 16; }
            x = ((x / f) << SCALE) + (x % f) + c->cum[sym];
        }
        st[j] = x; lb[j] = (uint32_t)(2 * k);
        for (uint64_t q = 0; q < k; q++) { uint16_t v = tmp[k - 1 - q]; w[0] = (uint8_t)v; w[1] = (uint8_t)(v >> 8); w += 2; }
    }
    memcpy(dst, st, sizeof st); memcpy(dst + 32, lb, sizeof lb);
    c->clen[ci] = (uint64_t)(w - dst);
    free(tmp);
}
int bbl_encode9(const uint8_t* a, const uint8_t* b, uint64_t n, int op, const uint16_t* freq512,
                uint8_t* out, uint64_t* clen, int nt) {
    enc9_ctx c; c.a = a; c.b = b; c.n = n; c.op = op; c.freq = freq512; c.out = out; c.clen = clen; c.err = 0;
    uint32_t s = 0;
    for (int k = 0; k < 512; k++) { c.cum[k] = s; s += freq512[k]; }
    c.cum[512] = s;
    if (s != MTOT) return -3;
    for (int k = 0; k < 512; k++) if (freq512[k] >= MTOT) return -4;
    run_pool(enc9_job, &c, (n + CH - 1) / CH, nt);
    return -c.err;
}
/* mantissa m7 = v & 0x7F of form(op, a, b), 8 per group into 7 bytes little-endian (bits 7j..7j+6 of the group word);
   out holds 7 * ceil(n / 8) bytes; the last group's missing elements are 0 */
typedef struct { const uint8_t* a; const uint8_t* b; uint64_t n; int op; uint8_t* out; uint64_t per; } pk_ctx;
static void pk_job(uint64_t i, void* c_) {
    pk_ctx* c = (pk_ctx*)c_;
    uint64_t g0 = i * c->per, g1 = g0 + c->per, ng = (c->n + 7) / 8; if (g1 > ng) g1 = ng;
    for (uint64_t g = g0; g < g1; g++) {
        uint64_t wd = 0;
        for (uint64_t j = 0; j < 8; j++) {
            uint64_t idx = 8 * g + j;
            if (idx >= c->n) break;
            uint16_t v = c->op == OP_PLAIN ? ld16(c->a, idx) : form(c->op, ld16(c->a, idx), ld16(c->b, idx));
            wd |= (uint64_t)(v & 0x7Fu) << (7 * j);
        }
        uint8_t* q = c->out + 7 * g;
        for (int bq = 0; bq < 7; bq++) q[bq] = (uint8_t)(wd >> (8 * bq));
    }
}
int bbl_pack7(const uint8_t* a, const uint8_t* b, uint64_t n, int op, uint8_t* out, int nt) {
    pk_ctx c = {a, b, n, op, out, 1u << 19};
    uint64_t ng = (n + 7) / 8;
    run_pool(pk_job, &c, (ng + c.per - 1) / c.per, nt);
    return 0;
}

/* ---------------- decode ---------------- */
/* task = 8 x u64: out_off, src_off, n, hi_off, lo_off, tabs (hi | lo<<32), flags, reserved
   flags: kind | hi_mode<<8 | lo_mode<<16 | hi_c<<24 | lo_c<<32 | op<<40 | src_idx<<48                        */
typedef struct { const uint8_t* blob; uint64_t blob_len; const uint64_t* tasks; const uint32_t* T; uint8_t* out;
                 const uint8_t* const* srcs; volatile uint64_t err; const uint32_t* T9; uint64_t ntab9; } dec_ctx;

static inline __attribute__((always_inline)) int rans_dec_core(uint32_t* x, const uint8_t** p, const uint32_t* T, uint8_t* o,
                                                              uint64_t k, const int OS) {
    const uint8_t* S = (const uint8_t*)(T + MTOT);
    /* k symbols, starting at a lane-0 boundary; nfull_lanes unused (k%LANES handled at the end) */
    uint64_t g = k / LANES, r = k % LANES;
    uint32_t x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4], x5 = x[5], x6 = x[6], x7 = x[7];
    const uint8_t *p0 = p[0], *p1 = p[1], *p2 = p[2], *p3 = p[3], *p4 = p[4], *p5 = p[5], *p6 = p[6], *p7 = p[7];
#if defined(__x86_64__)
#define RENORM(X, P) { uint16_t w_; memcpy(&w_, P, 2); uint32_t y_ = (X << 16) | w_; uint64_t two_ = 2, z_ = 0; \
    __asm__("cmpl %[rl], %[x]\n\tcmovbl %[y], %[x]\n\tcmovaeq %[z], %[t]" : [x] "+r"(X), [t] "+r"(two_) : [y] "r"(y_), \
            [z] "r"(z_), [rl] "i"(RL) : "cc"); P += two_; }
#else
#define RENORM(X, P) { uint16_t w_; memcpy(&w_, P, 2); uint32_t nd_ = X < RL; \
    X = (X << (nd_ << 4)) | ((uint32_t)w_ & (0u - nd_)); P += nd_ << 1; }
#endif
#define STEP(J) { uint32_t sl = x##J & (MTOT - 1); uint32_t e = T[sl]; o[(J) * OS] = S[sl]; \
                  x##J = (e & 0xFFFFu) * (x##J >> SCALE) + (e >> 16); RENORM(x##J, p##J) }
    for (uint64_t i = 0; i < g; i++) { STEP(0) STEP(1) STEP(2) STEP(3) STEP(4) STEP(5) STEP(6) STEP(7) o += LANES * OS; }
    if (r > 0) STEP(0)
    if (r > 1) STEP(1)
    if (r > 2) STEP(2)
    if (r > 3) STEP(3)
    if (r > 4) STEP(4)
    if (r > 5) STEP(5)
    if (r > 6) STEP(6)
#undef STEP
    x[0] = x0; x[1] = x1; x[2] = x2; x[3] = x3; x[4] = x4; x[5] = x5; x[6] = x6; x[7] = x7;
    p[0] = p0; p[1] = p1; p[2] = p2; p[3] = p3; p[4] = p4; p[5] = p5; p[6] = p6; p[7] = p7;
    return 0;
}
static int rans_dec_chunk_part(uint32_t* x, const uint8_t** p, const uint32_t* T, uint8_t* o, uint64_t k, int unused) {
    (void)unused; return rans_dec_core(x, p, T, o, k, 1);
}
static int rans_dec_into_odd(uint32_t* x, const uint8_t** p, const uint32_t* T, uint8_t* o, uint64_t k) {
    return rans_dec_core(x, p, T, o, k, 2);
}

/* v2 E: k 9-bit symbols into o (u16), starting at a lane-0 boundary. Same lane layout as the 8-bit core. */
static void rans9_dec(uint32_t* x, const uint8_t** p, const uint32_t* T9, uint16_t* o, uint64_t k) {
    const uint16_t* S9 = (const uint16_t*)(T9 + MTOT);
    uint32_t y0 = x[0], y1 = x[1], y2 = x[2], y3 = x[3], y4 = x[4], y5 = x[5], y6 = x[6], y7 = x[7];
    const uint8_t *q0 = p[0], *q1 = p[1], *q2 = p[2], *q3 = p[3], *q4 = p[4], *q5 = p[5], *q6 = p[6], *q7 = p[7];
#define S9STEP(J, OUT) { uint32_t sl9 = y##J & (MTOT - 1); uint32_t e9 = T9[sl9]; OUT = S9[sl9]; \
                         y##J = (e9 & 0xFFFFu) * (y##J >> SCALE) + (e9 >> 16); RENORM(y##J, q##J) }
    uint64_t g = k / LANES, r = k % LANES;
    for (uint64_t i = 0; i < g; i++, o += LANES) {
        S9STEP(0, o[0]) S9STEP(1, o[1]) S9STEP(2, o[2]) S9STEP(3, o[3]) S9STEP(4, o[4]) S9STEP(5, o[5]) S9STEP(6, o[6]) S9STEP(7, o[7])
    }
    uint16_t* t = o;
    switch (r) {             /* lanes 0..r-1 carry one more symbol (the encoder's per-lane counts) */
    case 7: S9STEP(6, t[6]) /* fall through */
    case 6: S9STEP(5, t[5]) /* fall through */
    case 5: S9STEP(4, t[4]) /* fall through */
    case 4: S9STEP(3, t[3]) /* fall through */
    case 3: S9STEP(2, t[2]) /* fall through */
    case 2: S9STEP(1, t[1]) /* fall through */
    case 1: S9STEP(0, t[0]) /* fall through */
    default: break;
    }
#undef S9STEP
    x[0] = y0; x[1] = y1; x[2] = y2; x[3] = y3; x[4] = y4; x[5] = y5; x[6] = y6; x[7] = y7;
    p[0] = q0; p[1] = q1; p[2] = q2; p[3] = q3; p[4] = q4; p[5] = q5; p[6] = q6; p[7] = q7;
}
/* v2 E: unpack k mantissas (k from a group boundary) from pk; then v = (s9 << 7) | m7 into hb/lb */
static void unpack7_join(const uint8_t* pk, const uint16_t* s9, uint8_t* hb, uint8_t* lb, uint64_t k) {
    for (uint64_t g = 0; 8 * g < k; g++) {
        uint64_t wd = 0; memcpy(&wd, pk + 7 * g, 8);          /* reads 1 byte past the group: inside the blob pad */
        for (uint64_t j = 0; j < 8 && 8 * g + j < k; j++) {
            uint64_t i = 8 * g + j;
            uint16_t v = (uint16_t)((s9[i] << 7) | ((wd >> (7 * j)) & 0x7Fu));
            hb[i] = (uint8_t)(v >> 8); lb[i] = (uint8_t)v;
        }
    }
}

typedef struct { uint32_t x[LANES]; const uint8_t* p[LANES]; const uint8_t* end[LANES]; } rstate;
static int rstate_init(rstate* s, const uint8_t* chunk, const uint8_t* blob_end) {
    uint32_t lb[LANES];
    memcpy(s->x, chunk, 32); memcpy(lb, chunk + 32, 32);
    const uint8_t* q = chunk + 64;
    for (int j = 0; j < LANES; j++) {
        if (s->x[j] < RL) return 1;
        s->p[j] = q; q += lb[j]; s->end[j] = q;
        if (q > blob_end) return 1;
    }
    return 0;
}
static int rstate_done(const rstate* s) {
    for (int j = 0; j < LANES; j++) if (s->x[j] != RL || s->p[j] != s->end[j]) return 1;
    return 0;
}

static void dec_job(uint64_t ti, void* c_) {
    dec_ctx* c = (dec_ctx*)c_;
    const uint64_t* t = c->tasks + 8 * ti;
    uint64_t out_off = t[0], src_off = t[1], n = t[2], hi_off = t[3], lo_off = t[4], tabs = t[5], fl = t[6];
    int kind = (int)(fl & 0xFF), hm = (int)((fl >> 8) & 0xFF), lm = (int)((fl >> 16) & 0xFF);
    uint8_t hc = (uint8_t)((fl >> 24) & 0xFF), lc = (uint8_t)((fl >> 32) & 0xFF);
    int op = (int)((fl >> 40) & 0xFF), si = (int)((fl >> 48) & 0xFF);
    uint8_t* o = c->out + out_off;
    const uint8_t* bend = c->blob + c->blob_len;
    if (kind == K_RAWCOPY) {
        if (hi_off + n > c->blob_len) { __atomic_fetch_add(&c->err, 1, __ATOMIC_RELAXED); return; }
        memcpy(o, c->blob + hi_off, n); return;
    }
    const uint8_t* src = (op != OP_PLAIN) ? c->srcs[si] + src_off : 0;
    if (hm == M_CONST && lm == M_CONST && hc == 0 && lc == 0 && (op == OP_XOR || op == OP_SUB)) { memcpy(o, src, 2 * n); return; }
    if (hm == M_CONST && lm == M_CONST && op == OP_PLAIN) {
        for (uint64_t i = 0; i < n; i++) { o[2 * i] = lc; o[2 * i + 1] = hc; }
        return;
    }
    const uint32_t* Th = c->T + (uint64_t)(uint32_t)(tabs & 0xFFFFFFFFu) * (MTOT + MTOT / 4);
    const uint32_t* Tl = c->T + (uint64_t)(uint32_t)(tabs >> 32) * (MTOT + MTOT / 4);
    rstate sh, sl;
    const uint32_t* T9h = 0;
    if (hm == M_RANS9 || lm == M_PACK7) {     /* v2 E: the two slots come as a pair, with a valid 9-bit table */
        uint64_t t9i = (uint64_t)(uint32_t)(tabs & 0xFFFFFFFFu);
        if (hm != M_RANS9 || lm != M_PACK7 || !c->T9 || t9i >= c->ntab9 || lo_off + 7 * ((n + 7) / 8) > c->blob_len
            || rstate_init(&sh, c->blob + hi_off, bend)) { __atomic_fetch_add(&c->err, 1, __ATOMIC_RELAXED); return; }
        T9h = c->T9 + t9i * T9W;
    }
    if (hm == M_RANS && rstate_init(&sh, c->blob + hi_off, bend)) { __atomic_fetch_add(&c->err, 1, __ATOMIC_RELAXED); return; }
    if (lm == M_RANS && rstate_init(&sl, c->blob + lo_off, bend)) { __atomic_fetch_add(&c->err, 1, __ATOMIC_RELAXED); return; }
    if ((hm == M_RAW && hi_off + n > c->blob_len) || (lm == M_RAW && lo_off + n > c->blob_len)) {
        __atomic_fetch_add(&c->err, 1, __ATOMIC_RELAXED); return; }
    uint8_t hb[SUBB], lb[SUBB];
    uint16_t s9b[SUBB];
    if (hm == M_CONST) memset(hb, hc, SUBB);
    if (lm == M_CONST) memset(lb, lc, SUBB);
    for (uint64_t pos = 0; pos < n; pos += SUBB) {
        uint64_t k = n - pos < SUBB ? n - pos : SUBB;
        const uint8_t* H; const uint8_t* L;
        if (hm == M_RANS9) {                   /* fills hb/lb; the combine below then reads H = hb, L = lb (every op) */
            rans9_dec(sh.x, sh.p, T9h, s9b, k);
            unpack7_join(c->blob + lo_off + 7 * (pos / 8), s9b, hb, lb, k);
        }
        if (op == OP_PLAIN && hm == M_RANS && lm == M_RAW) {
            uint8_t* oo = o + 2 * pos; const uint8_t* ll = c->blob + lo_off + pos;
            for (uint64_t i = 0; i < k; i++) { oo[2 * i] = ll[i]; oo[2 * i + 1] = 0; }
            rans_dec_into_odd(sh.x, sh.p, Th, oo + 1, k);
            continue;
        }
        if (hm == M_RANS) { rans_dec_chunk_part(sh.x, sh.p, Th, hb, k, 0); H = hb; }
        else if (hm == M_RAW) H = c->blob + hi_off + pos; else H = hb;
        if (lm == M_RANS) { rans_dec_chunk_part(sl.x, sl.p, Tl, lb, k, 0); L = lb; }
        else if (lm == M_RAW) L = c->blob + lo_off + pos; else L = lb;
        uint8_t* oo = o + 2 * pos;
        if (op == OP_PLAIN) {
            for (uint64_t i = 0; i < k; i++) { oo[2 * i] = L[i]; oo[2 * i + 1] = H[i]; }
        } else {
            const uint8_t* ss = src + 2 * pos;
            if (op == OP_XOR) {
                for (uint64_t i = 0; i < k; i++) { oo[2 * i] = ss[2 * i] ^ L[i]; oo[2 * i + 1] = ss[2 * i + 1] ^ H[i]; }
            } else {
                for (uint64_t i = 0; i < k; i++) {
                    uint16_t b = (uint16_t)(ss[2 * i] | (ss[2 * i + 1] << 8)), d = (uint16_t)(L[i] | (H[i] << 8)), v;
                    if (op == OP_SUB) v = (uint16_t)(b + d);
                    else if (op == OP_XORCAST) v = (uint16_t)(cast_bf16_f16(b) ^ d);
                    else v = (uint16_t)(cast_bf16_f16(b) + d);
                    oo[2 * i] = (uint8_t)v; oo[2 * i + 1] = (uint8_t)(v >> 8);
                }
            }
        }
    }
    if ((hm == M_RANS && rstate_done(&sh)) || (lm == M_RANS && rstate_done(&sl))) __atomic_fetch_add(&c->err, 1, __ATOMIC_RELAXED);
    if (hm == M_RANS9 && rstate_done(&sh)) __atomic_fetch_add(&c->err, 1, __ATOMIC_RELAXED);
}

typedef struct { const uint16_t* freqs; uint32_t* T; } tab_ctx;
static void tab_job(uint64_t ti, void* c_) {
    tab_ctx* c = (tab_ctx*)c_;
    const uint16_t* f = c->freqs + 256 * ti; uint32_t* T = c->T + (MTOT + MTOT / 4) * ti;
    uint8_t* S = (uint8_t*)(T + MTOT);
    uint32_t cum = 0;
    for (uint32_t s = 0; s < 256; s++) {
        for (uint32_t k = 0; k < f[s]; k++) { T[cum + k] = (uint32_t)f[s] | (k << 16); S[cum + k] = (uint8_t)s; }
        cum += f[s];
    }
}
typedef struct { const uint16_t* freqs9; uint32_t* T9; } tab9_ctx;
static void tab9_job(uint64_t ti, void* c_) {
    tab9_ctx* c = (tab9_ctx*)c_;
    const uint16_t* f = c->freqs9 + 512 * ti; uint32_t* T = c->T9 + T9W * ti;
    uint16_t* S = (uint16_t*)(T + MTOT);
    uint32_t cum = 0;
    for (uint32_t s = 0; s < 512; s++) {
        for (uint32_t k = 0; k < f[s]; k++) { T[cum + k] = (uint32_t)f[s] | (k << 16); S[cum + k] = (uint16_t)s; }
        cum += f[s];
    }
}
/* returns lane/bounds error count (0 = clean); tables built inside (they are part of the wake). v2: + 9-bit tables */
int64_t bbl_decode2(const uint8_t* blob, uint64_t blob_len, const uint64_t* tasks, uint64_t ntasks,
                    const uint16_t* freqs, uint64_t ntab, const uint16_t* freqs9, uint64_t ntab9,
                    uint8_t* out, const uint8_t* const* srcs, int nt) {
    uint32_t* T = (uint32_t*)malloc((ntab ? ntab : 1) * (MTOT + MTOT / 4) * sizeof(uint32_t));
    if (!T) return -1;
    for (uint64_t i = 0; i < ntab; i++) {
        uint32_t s = 0; for (int k = 0; k < 256; k++) { s += freqs[256 * i + k]; if (freqs[256 * i + k] >= MTOT) { free(T); return -2; } }
        if (s != MTOT) { free(T); return -2; }
    }
    uint32_t* T9 = 0;
    if (ntab9) {
        for (uint64_t i = 0; i < ntab9; i++) {
            uint32_t s = 0;
            for (int k = 0; k < 512; k++) { s += freqs9[512 * i + k]; if (freqs9[512 * i + k] >= MTOT) { free(T); return -2; } }
            if (s != MTOT) { free(T); return -2; }
        }
        T9 = (uint32_t*)malloc(ntab9 * T9W * sizeof(uint32_t));
        if (!T9) { free(T); return -1; }
        tab9_ctx t9c = {freqs9, T9};
        run_pool(tab9_job, &t9c, ntab9, nt);
    }
    tab_ctx tc = {freqs, T};
    run_pool(tab_job, &tc, ntab, nt);
    dec_ctx c; c.blob = blob; c.blob_len = blob_len; c.tasks = tasks; c.T = T; c.out = out; c.srcs = srcs; c.err = 0;
    c.T9 = T9; c.ntab9 = ntab9;
    run_pool(dec_job, &c, ntasks, nt);
    free(T); free(T9);
    return (int64_t)c.err;
}
int64_t bbl_decode(const uint8_t* blob, uint64_t blob_len, const uint64_t* tasks, uint64_t ntasks,
                   const uint16_t* freqs, uint64_t ntab, uint8_t* out, const uint8_t* const* srcs, int nt) {
    return bbl_decode2(blob, blob_len, tasks, ntasks, freqs, ntab, 0, 0, out, srcs, nt);
}

/* ---------------- memcpy ceiling ---------------- */
typedef struct { uint8_t* d; const uint8_t* s; uint64_t n; } cp_ctx;
static void cp_job(uint64_t i, void* c_) {
    cp_ctx* c = (cp_ctx*)c_; uint64_t s = i * (uint64_t)(2 * CH), e = s + 2 * CH; if (e > c->n) e = c->n;
    memcpy(c->d + s, c->s + s, e - s);
}
int bbl_memcpy_mt(uint8_t* d, const uint8_t* s, uint64_t n, int nt) {
    cp_ctx c = {d, s, n}; run_pool(cp_job, &c, (n + 2 * CH - 1) / (2 * CH), nt); return 0;
}
/* first differing byte index, or -1 */
int64_t bbl_first_diff(const uint8_t* a, const uint8_t* b, uint64_t n) {
    uint64_t step = 1u << 20;
    for (uint64_t s = 0; s < n; s += step) {
        uint64_t k = n - s < step ? n - s : step;
        if (memcmp(a + s, b + s, k)) { for (uint64_t i = s; i < s + k; i++) if (a[i] != b[i]) return (int64_t)i; }
    }
    return -1;
}
uint16_t bbl_cast(uint16_t b) { return cast_bf16_f16(b); }
int bbl_consts(uint64_t* o) { o[0] = SCALE; o[1] = LANES; o[2] = CH; o[3] = CMAX; o[4] = SUBB; return 0; }
"""

SCALE, M_TOT, LANES, CH = 12, 4096, 8, 1 << 18
CMAX = 64 + 2 * CH + 64
OPS = {"plain": 0, "xor": 1, "sub": 2, "xorcast": 3, "subcast": 4}
MODES = {"raw": 0, "rans": 1, "const": 2, "rans9": 3, "pack7": 4}   # rans9 / pack7 = v2 E (9-bit symbol + 7-bit mantissa)
DEDUP_MIN_BYTES = 8192        # v2 D: tensors of >= 8,192 B are dedup candidates
CODE_IF_BITS_LE = 7.5          # a stream is rANS-coded iff its estimated cost is <= 7.5 bits/symbol (rule A1)
MIN_CODED_ELEMS = 4096         # tensors below this stay RAW (rule A1)
MAGIC = b"BBL3283\x01"
PAD = 2 * (CH // LANES) + 64      # a corrupted lane reads at most 2 B/symbol past its start: never outside the buffer
U8P = ctypes.POINTER(ctypes.c_uint8)
REFS = ("Qwen__Qwen3-4B", "Qwen__Qwen3-4B-Base")


class BuildError(RuntimeError):
    pass


_LIB = None


def build_lib():
    """Compile C_SRC once per source hash into ~/.cache/lossless_bf16/ and load it."""
    global _LIB
    if _LIB is not None:
        return _LIB
    h = hashlib.sha256(C_SRC.encode()).hexdigest()[:16]
    d = os.path.join(os.path.expanduser("~"), ".cache", "lossless_bf16")
    os.makedirs(d, exist_ok=True)
    so = os.path.join(d, "bbl_%s.so" % h)
    if not os.path.exists(so):
        fd, csrc = tempfile.mkstemp(suffix=".c", dir=d)
        os.write(fd, C_SRC.encode()); os.close(fd)
        tmp_so = so + ".tmp%d" % os.getpid()
        last = None
        for flags in (["-O3", "-march=native"], ["-O3"]):
            r = subprocess.run(["gcc", *flags, "-fPIC", "-shared", "-pthread", "-o", tmp_so, csrc],
                               capture_output=True, text=True)
            if r.returncode == 0:
                os.replace(tmp_so, so); last = None; break
            last = r.stderr[-2000:]
        os.unlink(csrc)
        if last is not None:
            raise BuildError(last)
    L = ctypes.CDLL(so)
    vp, u64, i32 = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int
    L.bbl_hist.argtypes = [vp, vp, u64, i32, vp, i32]
    L.bbl_hist.restype = i32
    L.bbl_form_bytes.argtypes = [vp, vp, u64, i32, i32, vp, i32]
    L.bbl_encode.argtypes = [vp, vp, u64, i32, i32, vp, vp, vp, i32]
    L.bbl_encode.restype = i32
    L.bbl_decode.argtypes = [vp, u64, vp, u64, vp, u64, vp, vp, i32]
    L.bbl_decode.restype = ctypes.c_int64
    L.bbl_decode2.argtypes = [vp, u64, vp, u64, vp, u64, vp, u64, vp, vp, i32]
    L.bbl_decode2.restype = ctypes.c_int64
    L.bbl_encode9.argtypes = [vp, vp, u64, i32, vp, vp, vp, i32]
    L.bbl_encode9.restype = i32
    L.bbl_pack7.argtypes = [vp, vp, u64, i32, vp, i32]
    L.bbl_memcpy_mt.argtypes = [vp, vp, u64, i32]
    L.bbl_first_diff.argtypes = [vp, vp, u64]
    L.bbl_first_diff.restype = ctypes.c_int64
    L.bbl_cast.argtypes = [ctypes.c_uint16]
    L.bbl_cast.restype = ctypes.c_uint16
    L.bbl_consts.argtypes = [vp]
    consts = np.zeros(5, np.uint64)
    L.bbl_consts(consts.ctypes.data)
    if tuple(int(x) for x in consts[:4]) != (SCALE, LANES, CH, CMAX):
        raise BuildError("C/Python constants disagree: %s" % consts.tolist())
    _LIB = L
    return L


def ptr(a):
    return None if a is None else a.ctypes.data


# ------------------------------------------------------------------------------------------------ entropy + tables
def H(c):
    c = np.asarray(c, dtype=np.float64)
    c = c[c > 0]
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def norm_freq(counts, M=M_TOT):
    """Integer frequencies summing to M, every present symbol >= 1, minimising cross-entropy greedily
    (floor, then +1 units by largest marginal gain / -1 units by smallest marginal loss; ties -> lowest symbol)."""
    c = np.asarray(counts, dtype=np.int64)
    n = int(c.sum())
    pres = [int(i) for i in np.nonzero(c)[0]]
    f = np.zeros(len(c), np.int64)
    if not pres:
        return f
    if len(pres) == 1:
        f[pres[0]] = M
        return f
    for i in pres:
        f[i] = max(1, (int(c[i]) * M) // n)
    s = int(f.sum())
    if s < M:
        h = [(-(c[i] * math.log2((f[i] + 1) / f[i])), i) for i in pres]
        heapq.heapify(h)
        while s < M:
            _, i = heapq.heappop(h)
            f[i] += 1; s += 1
            heapq.heappush(h, (-(c[i] * math.log2((f[i] + 1) / f[i])), i))
    while s > M:
        h = [(c[i] * math.log2(f[i] / (f[i] - 1)), i) for i in pres if f[i] > 1]
        _, i = min(h)
        f[i] -= 1; s -= 1
    return f


def ce_bits(counts, f, M=M_TOT):
    c = np.asarray(counts, np.float64)
    m = c > 0
    if not m.any():
        return 0.0
    return float(-(c[m] * np.log2(f[m] / M)).sum())


def est_stream_bytes(counts, n):
    """estimated stored bytes of one stream under the A1 rule (rANS table counted)."""
    m, _, _, b = plan_stream(counts, n)
    return b + (512 if m == "rans" else 0)


def prefault(n):
    a = np.empty(n, np.uint8)
    a.fill(1)
    return a


def plan_stream(counts, n):
    """-> (mode, freq|None, const, est_bytes). rule A1."""
    pres = np.nonzero(counts)[0]
    if n == 0:
        return "raw", None, 0, 0
    if n < MIN_CODED_ELEMS:
        return "raw", None, 0, n
    if len(pres) == 1:
        return "const", None, int(pres[0]), 0
    f = norm_freq(counts)
    bits = ce_bits(counts, f)
    if bits / n <= CODE_IF_BITS_LE:
        return "rans", f, 0, bits / 8
    return "raw", None, 0, n


# ------------------------------------------------------------------------------------------------ container writer
class Writer:
    """Accumulates blob parts + a manifest. Offsets are blob-relative."""

    def __init__(self):
        self.parts, self.size = [], 0
        self.tables = []          # blob offsets of 512 B freq tables
        self.tables9 = []         # v2 E: blob offsets of 1,024 B 9-bit freq tables
        self.files = []

    def add(self, b):
        b = np.ascontiguousarray(b, dtype=np.uint8).ravel()
        off = self.size
        if b.size:
            self.parts.append(b)
            self.size += b.size
        return off

    def add_table(self, f):
        self.tables.append(self.add(np.asarray(f, np.uint16).view(np.uint8)))
        return len(self.tables) - 1

    def add_table9(self, f):
        self.tables9.append(self.add(np.asarray(f, np.uint16).view(np.uint8)))
        return len(self.tables9) - 1

    def finish(self):
        d = {"v": 1, "tables": self.tables, "files": self.files}
        if self.tables9 or any(sg["k"] == "d" for F in self.files for sg in F["segs"]):
            d = {"v": 2, "tables": self.tables, "tables9": self.tables9, "files": self.files}   # v1 bytes unchanged otherwise
        man = json.dumps(d, separators=(",", ":")).encode()
        blob = np.concatenate(self.parts + [np.zeros(PAD, np.uint8)]) if self.parts else np.zeros(PAD, np.uint8)
        return man, blob


def encode_stream(L, w, a8, b8, n, op, sel, counts, nt):
    mode, f, cval, _ = plan_stream(counts, n)
    st = {"m": mode}
    if mode == "const":
        st["c"] = cval
        st["bytes"] = 0
    elif mode == "raw":
        buf = np.empty(n, np.uint8)
        if n:
            L.bbl_form_bytes(ptr(a8), ptr(b8), n, OPS[op], sel, ptr(buf), nt)
        st["off"] = w.add(buf)
        st["bytes"] = n
    else:
        nch = (n + CH - 1) // CH
        tmp = np.empty(nch * CMAX, np.uint8)
        clen = np.zeros(nch, np.uint64)
        fr = np.ascontiguousarray(f, np.uint16)
        rc = L.bbl_encode(ptr(a8), ptr(b8), n, OPS[op], sel, ptr(fr), ptr(tmp), ptr(clen), nt)
        if rc != 0:
            raise RuntimeError("bbl_encode rc=%d" % rc)
        st["t"] = w.add_table(fr)
        offs = []
        for ci in range(nch):
            offs.append(w.add(tmp[ci * CMAX: ci * CMAX + int(clen[ci])]))
        st["ch"] = offs
        st["bytes"] = int(clen.sum()) + 512
    return st


def est9_bytes(counts9, n):
    """v2 E estimate: s9 cross-entropy / 8 + 1,024 B table + 7 * ceil(n / 8); None if E is not eligible."""
    if n < MIN_CODED_ELEMS or np.count_nonzero(counts9) < 2:
        return None, None
    f9 = norm_freq(counts9)
    return ce_bits(counts9, f9) / 8 + 1024 + 7 * ((n + 7) // 8), f9


def encode_stream9(L, w, a8, b8, n, op, f9, nt):
    """v2 E: (hi slot, lo slot) = (512-symbol rANS of v >> 7, v & 0x7F packed 8 per 7 bytes)."""
    nch = (n + CH - 1) // CH
    tmp = np.empty(nch * CMAX, np.uint8)
    clen = np.zeros(nch, np.uint64)
    fr = np.ascontiguousarray(f9, np.uint16)
    rc = L.bbl_encode9(ptr(a8), ptr(b8), n, OPS[op], ptr(fr), ptr(tmp), ptr(clen), nt)
    if rc != 0:
        raise RuntimeError("bbl_encode9 rc=%d" % rc)
    hi = {"m": "rans9", "t9": w.add_table9(fr)}
    hi["ch"] = [w.add(tmp[ci * CMAX: ci * CMAX + int(clen[ci])]) for ci in range(nch)]
    hi["bytes"] = int(clen.sum()) + 1024
    pk = np.zeros(7 * ((n + 7) // 8), np.uint8)
    L.bbl_pack7(ptr(a8), ptr(b8), n, OPS[op], ptr(pk), nt)
    lo = {"m": "pack7", "off": w.add(pk), "bytes": int(pk.size)}
    return hi, lo


def encode_tensor(L, w, a8, b8, n, op, nt, joint=None, exp9=False, force9=False):
    """Encode one 16-bit tensor (a8: 2n bytes). Returns (segment dict, joint histogram).
    exp9 (v2 E): use the 9-bit split iff its estimate is strictly below HB's; force9 (self-test only) whenever eligible."""
    if joint is None:
        joint = np.zeros(65536, np.uint64)
        if n and L.bbl_hist(ptr(a8), ptr(b8), n, OPS[op], ptr(joint), nt) != 0:
            raise RuntimeError("bbl_hist allocation failure")
    J = joint.reshape(256, 256)
    hi, lo = J.sum(1), J.sum(0)
    if exp9 or force9:
        e9, f9 = est9_bytes(joint.reshape(512, 128).sum(1), n)
        if e9 is not None and (force9 or e9 < est_stream_bytes(hi, n) + est_stream_bytes(lo, n)):
            h9, l9 = encode_stream9(L, w, a8, b8, n, op, f9, nt)
            return {"k": "t", "n": n, "op": op, "hi": h9, "lo": l9}, joint
    seg = {"k": "t", "n": n, "op": op,
           "hi": encode_stream(L, w, a8, b8, n, op, 1, hi, nt),
           "lo": encode_stream(L, w, a8, b8, n, op, 0, lo, nt)}
    return seg, joint


def tasks_for(manifest, out_base=None, src_index=None):
    """manifest -> (tasks uint64[N,8], freq uint16[ntab,256] offsets list, total_out). Files are laid out back to back."""
    rows = []
    out = 0
    for fi, F in enumerate(manifest["files"]):
        base = out if out_base is None else out_base[fi]
        for seg in F["segs"]:
            o = base + seg["o"]
            if seg["k"] == "r":
                for s in range(0, seg["n"], 2 * CH):
                    k = min(2 * CH, seg["n"] - s)
                    rows.append((o + s, 0, k, seg["off"] + s, 0, 0, 0, 0))
                continue
            if seg["k"] == "d":            # v2 D: restored by decode()'s copy pass after the parallel decode
                continue
            n, op = seg["n"], OPS[seg["op"]]
            hi, lo = seg["hi"], seg["lo"]
            si, soff = (seg["src"] if seg.get("src") else (0, 0))
            if src_index is not None and seg.get("src"):
                si = src_index[si]
            ht = (hi["t9"] if hi["m"] == "rans9" else hi.get("t", 0)) & 0xFFFFFFFF
            lt = lo.get("t", 0) & 0xFFFFFFFF
            fl_base = (1 | (MODES[hi["m"]] << 8) | (MODES[lo["m"]] << 16) | (hi.get("c", 0) << 24)
                       | (lo.get("c", 0) << 32) | (op << 40) | (si << 48))
            for ci, s in enumerate(range(0, max(n, 1), CH)):
                k = min(CH, n - s)
                if k <= 0:
                    break
                hoff = hi["ch"][ci] if hi["m"] in ("rans", "rans9") else (hi.get("off", 0) + s)
                if lo["m"] == "pack7":
                    loff = lo["off"] + 7 * (s // 8)       # s is a multiple of CH, so of 8
                else:
                    loff = lo["ch"][ci] if lo["m"] == "rans" else (lo.get("off", 0) + s)
                rows.append((o + 2 * s, soff + 2 * s, k, hoff, loff, ht | (lt << 32), fl_base, 0))
        out = base + F["size"]
    t = np.array(rows, dtype=np.uint64).reshape(-1, 8) if rows else np.zeros((0, 8), np.uint64)
    return t, out


def freqs_from(manifest, blob):
    offs = manifest["tables"]
    if not offs:
        return np.zeros((1, 256), np.uint16), 0
    fr = np.stack([blob[o:o + 512].view(np.uint16) for o in offs])
    return np.ascontiguousarray(fr), len(offs)


def freqs9_from(manifest, blob):
    offs = manifest.get("tables9", [])
    if not offs:
        return np.zeros((1, 512), np.uint16), 0
    fr = np.stack([blob[o:o + 1024].view(np.uint16) for o in offs])
    return np.ascontiguousarray(fr), len(offs)


def dup_pass(L, manifest, out, dup_srcs, nt):
    """v2 D: restore every DUP segment by copying its referenced bytes. A reference to a file of this container reads
    this container's decoded output; any other reads dup_srcs[(file, offset)] (the referenced file's decoded bytes).
    A missing source raises ValueError: never a silent output."""
    bases, b = {}, 0
    for F in manifest["files"]:
        bases[F["name"]] = b
        b += F["size"]
    for F in manifest["files"]:
        for seg in F["segs"]:
            if seg["k"] != "d":
                continue
            rf, ro = seg["ref"]
            n = seg["n"]
            if rf in bases:
                src = out[bases[rf] + ro: bases[rf] + ro + n]
                if src.size != n:                                       # QA C0: a bad ref must not over-read
                    raise ValueError("DUP ref past its file: %s @%d (%d B) for %s" % (rf, ro, n, seg.get("name")))
            else:
                src = (dup_srcs or {}).get((rf, ro))
                if src is None or src.size != n:
                    raise ValueError("DUP source missing: %s @%d (%d B) for %s" % (rf, ro, n, seg.get("name")))
            dst = out[bases[F["name"]] + seg["o"]: bases[F["name"]] + seg["o"] + n]
            L.bbl_memcpy_mt(ptr(dst), ptr(np.ascontiguousarray(src)), n, nt)
    return out


def decode(L, manifest, blob, srcs, nt, out=None, dup_srcs=None):
    """Decode every file of a container back to back. srcs: list of uint8 arrays (base files) or []. -> (out, err).
    v2: 9-bit tables (manifest v2) and the DUP copy pass (dup_srcs: {(file, offset): decoded bytes} for references
    into files outside this container)."""
    tasks, total = tasks_for(manifest)
    fr, ntab = freqs_from(manifest, blob)
    fr9, ntab9 = freqs9_from(manifest, blob)
    if out is None:
        out = np.empty(total, np.uint8)
    keep = [s for s in srcs]
    sp = (ctypes.c_void_p * max(1, len(keep)))(*[s.ctypes.data for s in keep]) if keep else (ctypes.c_void_p * 1)(None)
    err = L.bbl_decode2(ptr(blob), blob.size - PAD, ptr(tasks), tasks.shape[0], ptr(fr), ntab, ptr(fr9), ntab9,
                        ptr(out), sp, nt)
    dup_pass(L, manifest, out, dup_srcs, nt)
    return out, int(err)


def save_container(path, man, blob):
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<Q", len(man)) + man)
        blob.tofile(f)


def load_container(path):
    with open(path, "rb") as f:
        head = f.read(16)
        if head[:8] != MAGIC:
            raise ValueError("not a BBL3283 container: %s" % path)
        n = struct.unpack("<Q", head[8:16])[0]
        man = json.loads(f.read(n))
        blob = np.fromfile(f, dtype=np.uint8)
    if blob.size < PAD:
        raise ValueError("truncated container: %s" % path)
    return man, blob


def check_container(man, blob):
    """Refuse a container whose segments point outside the blob (truncation) before any decode."""
    lim = blob.size - PAD
    if man.get("v", 1) not in (1, 2):
        return "unknown container version %r" % man.get("v")
    for F in man["files"]:
        for seg in F["segs"]:
            if seg["k"] == "r":
                if seg["off"] + seg["n"] > lim:
                    return "raw segment past blob end"
                continue
            if seg["k"] == "d":
                r = seg.get("ref")
                if not (isinstance(r, list) and len(r) == 2 and isinstance(r[0], str) and isinstance(r[1], int)
                        and r[1] >= 0 and seg["n"] >= DEDUP_MIN_BYTES):
                    return "bad DUP segment"
                continue
            if seg["k"] != "t":
                return "unknown segment kind %r" % seg["k"]
            if (seg["hi"]["m"] == "rans9") != (seg["lo"]["m"] == "pack7"):
                return "rans9 / pack7 must come as a pair"
            for s in ("hi", "lo"):
                st = seg[s]
                if st["m"] == "raw" and st["off"] + seg["n"] > lim:
                    return "raw stream past blob end"
                if st["m"] in ("rans", "rans9") and any(o + 64 > lim for o in st["ch"]):
                    return "rans chunk past blob end"
                if st["m"] == "pack7" and st["off"] + 7 * ((seg["n"] + 7) // 8) > lim:
                    return "pack7 stream past blob end"
                if st["m"] == "rans9" and not 0 <= st["t9"] < len(man.get("tables9", [])):
                    return "rans9 table index out of range"
    for o in man["tables"]:
        if o + 512 > lim:
            return "table past blob end"
    for o in man.get("tables9", []):
        if o + 1024 > lim:
            return "9-bit table past blob end"
    return None


# ------------------------------------------------------------------------------------------------ safetensors
def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return 8 + n, h


def st_entries(path):
    """-> (data_base, [(name, dtype, shape, abs_off, nbytes)] sorted by offset)."""
    base, h = st_header(path)
    h.pop("__metadata__", None)
    ents = []
    for k, v in h.items():
        a, b = v["data_offsets"]
        ents.append((k, v["dtype"], tuple(v["shape"]), base + a, b - a))
    ents.sort(key=lambda e: e[3])
    return base, ents


SIXTEEN = ("BF16", "F16", "I16", "U16")


def dedup_hash(b):
    """v2 D candidate key (a module-level hook so the self-test can force collisions, GD3)."""
    return hashlib.sha256(memoryview(np.ascontiguousarray(b))).digest()


def plan_dedup(L, files, prefer=None):
    """v2 D plan for ONE checkpoint. files = [(file_name, file_bytes_u8)] in the checkpoint's order (index set sorted).
    Tensors of >= DEDUP_MIN_BYTES are grouped by byte identity (a dedup_hash match is confirmed by memcmp; a hash match without
    byte equality never dedups). In each group of >= 2 the ROOT is chosen among the members in the group's EARLIEST file:
    the first (by offset) for which prefer(name) is true, else the first by offset (prefer=None = the frozen v2 rule, which
    is purely positional). Every other member becomes {(file, abs_off): (ref_file, ref_abs_off, ref_name, name, nbytes)}.
    Invariant (QA C6): a DUP never references a file later than its own, so decoding files in order always has the
    root decoded first; a root is never itself a DUP."""
    groups, order = {}, []
    for fname, fb in files:
        for (k, dt, shp, off, nb) in st_entries_bytes(fb)[1]:
            if nb < DEDUP_MIN_BYTES:
                continue
            a8 = fb[off:off + nb]
            key = (nb, dedup_hash(a8))
            hit = None
            for gi in groups.get(key, ()):
                rb = order[gi][0][3]
                if int(L.bbl_first_diff(ptr(np.asarray(a8)), ptr(np.asarray(rb)), nb)) == -1:
                    hit = gi
                    break
            if hit is None:
                groups.setdefault(key, []).append(len(order))
                order.append([(fname, off, k, a8)])
            else:
                order[hit].append((fname, off, k, a8))
    plan = {}
    for mem in order:
        if len(mem) < 2:
            continue
        first_file = mem[0][0]
        cand = [m for m in mem if m[0] == first_file]
        root = next((m for m in cand if prefer is not None and prefer(m[2])), cand[0])
        for m in mem:
            if m is not root:
                plan[(m[0], m[1])] = (root[0], root[1], root[2], m[2], len(m[3]))
    return plan


def encode_file(L, w, fbytes, path_name, srcmap, nt, stats_cb=None, force_op=None, exp9=False, dedup=None):
    """Append one safetensors file to writer w. srcmap(name, dtype, shape) -> (src_idx, src_off_in_src, op, src_u8) or None.
    v2: exp9 = the E arm's per-tensor choice; dedup = a plan_dedup() plan (a planned tensor becomes a DUP segment).
    Returns the file dict (also appended to w.files)."""
    base, ents = st_entries_bytes(fbytes)
    segs = []
    segs.append({"k": "r", "o": 0, "n": base, "off": w.add(fbytes[:base])})
    pos = base
    for (k, dt, shp, off, nb) in ents:
        if off > pos:
            segs.append({"k": "r", "o": pos, "n": off - pos, "off": w.add(fbytes[pos:off])})
        a8 = fbytes[off:off + nb]
        dp = dedup.get((path_name, off)) if dedup else None
        if dp is not None:
            segs.append({"k": "d", "o": off, "n": nb, "name": k, "ref": [dp[0], int(dp[1])], "ref_name": dp[2]})
            if stats_cb:
                stats_cb(k, dt, nb // 2, segs[-1], None)
            pos = off + nb
            continue
        if dt in SIXTEEN and nb % 2 == 0:
            n = nb // 2
            src = srcmap(k, dt, shp) if srcmap else None
            if src is None:
                seg, joint = encode_tensor(L, w, a8, None, n, "plain", nt, exp9=exp9)
            else:
                si, soff, op, sbytes = src
                seg, joint = encode_tensor(L, w, a8, sbytes, n, op, nt, exp9=exp9)
                seg["src"] = [si, soff]
            seg["o"] = off
            seg["name"] = k
            if stats_cb:
                stats_cb(k, dt, n, seg, joint)
            segs.append(seg)
        else:
            segs.append({"k": "r", "o": off, "n": nb, "off": w.add(a8), "name": k})
        pos = off + nb
    if pos < fbytes.size:
        segs.append({"k": "r", "o": pos, "n": fbytes.size - pos, "off": w.add(fbytes[pos:])})
    F = {"name": path_name, "size": int(fbytes.size), "segs": segs}
    w.files.append(F)
    return F


def st_entries_bytes(fbytes):
    n = int(np.frombuffer(fbytes[:8].tobytes(), "<u8")[0])
    h = json.loads(fbytes[8:8 + n].tobytes())
    h.pop("__metadata__", None)
    ents = []
    for k, v in h.items():
        a, b = v["data_offsets"]
        ents.append((k, v["dtype"], tuple(v["shape"]), 8 + n + a, b - a))
    ents.sort(key=lambda e: e[3])
    return 8 + n, ents


def sha(a):
    h = hashlib.sha256()
    mv = memoryview(a)
    for s in range(0, len(mv), 1 << 28):
        h.update(mv[s:s + (1 << 28)])
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ self-test (goldens)
def _bf16(x):
    return (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _single(L, arr16, nt=4):
    """round trip one 16-bit tensor through the container path; -> (ok, seg)"""
    w = Writer()
    a8 = np.ascontiguousarray(arr16.astype(np.uint16)).view(np.uint8)
    seg, _ = encode_tensor(L, w, a8, None, arr16.size, "plain", nt)
    seg["o"] = 0
    w.files.append({"name": "t", "size": int(a8.size), "segs": [seg]})
    man, blob = w.finish()
    man = json.loads(man)
    out, err = decode(L, man, blob, [], nt)
    return (err == 0 and out.tobytes() == a8.tobytes()), seg, (man, blob)


def _delta(L, fine16, base16, op, nt=4):
    w = Writer()
    a8 = np.ascontiguousarray(fine16.astype(np.uint16)).view(np.uint8)
    b8 = np.ascontiguousarray(base16.astype(np.uint16)).view(np.uint8)
    seg, _ = encode_tensor(L, w, a8, b8, fine16.size, op, nt)
    seg["o"] = 0
    seg["src"] = [0, 0]
    w.files.append({"name": "t", "size": int(a8.size), "segs": [seg]})
    man, blob = w.finish()
    man = json.loads(man)
    out, err = decode(L, man, blob, [b8], nt)
    return (err == 0 and out.tobytes() == a8.tobytes()), seg


def payload(seg):
    return seg["hi"]["bytes"] + seg["lo"]["bytes"]


def self_test():
    try:
        L = build_lib()
    except (BuildError, OSError) as e:
        print("BUILD-FAILED", str(e)[-500:])
        return 3
    rng = np.random.default_rng(3283)
    res = []

    def chk(name, cond, detail=""):
        res.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  " + str(detail)))

    # G9 normalisation (hand-computed)
    chk("G9a {3000,1000} -> {3072,1024}", norm_freq([3000, 1000]).tolist() == [3072, 1024], norm_freq([3000, 1000]))
    chk("G9b {1,999999} -> {1,4095}", norm_freq([1, 999999]).tolist() == [1, 4095], norm_freq([1, 999999]))
    chk("G9c {1,1,1} -> {1366,1365,1365}", norm_freq([1, 1, 1]).tolist() == [1366, 1365, 1365], norm_freq([1, 1, 1]))
    # G1 every 16-bit pattern x3, shuffled
    allp = np.tile(np.arange(65536, dtype=np.uint16), 3)
    rng.shuffle(allp)
    ok, seg, _ = _single(L, allp)
    chk("G1 all 65536 patterns x3 exact, both RAW, payload 2n", ok and seg["hi"]["m"] == "raw" and seg["lo"]["m"] == "raw"
        and payload(seg) == 2 * allp.size, (ok, seg["hi"]["m"], seg["lo"]["m"], payload(seg)))
    # G2 constant
    ok, seg, _ = _single(L, np.full(1_000_000, 0x3F80, np.uint16))
    chk("G2 constant 0x3F80: both CONST, 0 payload, exact", ok and seg["hi"]["m"] == "const" and seg["lo"]["m"] == "const"
        and payload(seg) == 0, (ok, seg["hi"], seg["lo"]))
    # G3 fair two-symbol hi
    n = 1 << 20
    hi = np.where(rng.integers(0, 2, n) == 1, 0xBF, 0x3F).astype(np.uint16)
    arr = (hi << 8) | rng.integers(0, 256, n).astype(np.uint16)
    ok, seg, _ = _single(L, arr)
    hb = seg["hi"]["bytes"] - 512
    chk("G3 fair hi {0x3F,0xBF}: hi rANS within 131072+-512 B, lo RAW, exact",
        ok and seg["hi"]["m"] == "rans" and abs(hb - 131072) <= 512 and seg["lo"]["m"] == "raw", (ok, seg["hi"]["m"], hb))
    # G4 lengths, Gaussian bf16
    g_ok = True
    for nn in (0, 1, 7, 65535, 65536, 65537, 262143, 262144, 262145, 1_000_003):
        arr = _bf16(rng.normal(0, 0.02, nn))
        ok, seg, _ = _single(L, arr)
        g_ok &= ok
        if not ok:
            print("   G4 failed at n=%d" % nn)
    chk("G4 lengths 0..1,000,003 Gaussian bf16 exact", g_ok)
    # efficiency on the biggest G4 case (sanity: coded hi within 1% of its order-0 bound + overhead)
    arr = _bf16(rng.normal(0, 0.02, 1_000_003))
    ok, seg, _ = _single(L, arr)
    bound = H(np.bincount(arr >> 8, minlength=256)) * arr.size / 8
    chk("G4b Gaussian hi rANS <= 1.01 x order-0 bound + 64 B/chunk + 512 B table",
        ok and seg["hi"]["bytes"] <= 1.01 * bound + 64 * 4 + 512, (seg["hi"]["bytes"], round(bound)))
    # G5 XOR delta: identical; one bit flipped
    base = _bf16(rng.normal(0, 0.02, n))
    ok, seg = _delta(L, base.copy(), base, "xor")
    chk("G5a XD fine==base: IDENT (both CONST 0), 0 payload, exact", ok and payload(seg) == 0
        and seg["hi"]["m"] == seg["lo"]["m"] == "const", (ok, seg["hi"], seg["lo"]))
    fine = base.copy()
    fine[12345] ^= np.uint16(1 << 3)
    ok, seg = _delta(L, fine, base, "xor")
    chk("G5b XD one bit flipped: exact, payload <= 2048 B", ok and payload(seg) <= 2048, (ok, payload(seg)))
    # G6 fp16 bridge with specials
    specials = np.array([0x7FC1, 0xFFC0, 0x7F80, 0xFF80, 0x0001, 0x807F, 0x0000, 0x8000, 0x477F, 0x4780, 0x7F7F,
                         0x3380, 0x3300, 0x337F], np.uint16)
    b6 = np.concatenate([specials, _bf16(rng.normal(0, 3e4, 20000)), _bf16(rng.normal(0, 1e-6, 20000))])
    f6 = np.array([L.bbl_cast(int(x)) for x in b6], np.uint16)
    ok, seg = _delta(L, f6, b6, "xorcast")
    chk("G6a XORCAST fine==cast(base) incl NaN/Inf/subnormal/>65504: IDENT, exact", ok and payload(seg) == 0, (ok, payload(seg)))
    f6b = f6.copy(); f6b[100] ^= np.uint16(0x0155)
    ok, seg = _delta(L, f6b, b6, "xorcast")
    chk("G6b XORCAST one element changed: exact", ok, seg)
    # cast spot values (hand): 1.0 bf16 0x3F80 -> f16 0x3C00; 65504 = bf16 0x477F? (0x477F = 65280) -> 0x7BF8; 2^-24 -> 0x0001
    chk("G6c cast goldens 0x3F80->0x3C00, 0x477F->0x7BF8, 0x3380->0x0001, 0x3300->0x0000, 0x7F80->0x7C00",
        [L.bbl_cast(x) for x in (0x3F80, 0x477F, 0x3380, 0x3300, 0x7F80)] == [0x3C00, 0x7BF8, 0x0001, 0x0000, 0x7C00],
        [hex(L.bbl_cast(x)) for x in (0x3F80, 0x477F, 0x3380, 0x3300, 0x7F80)])
    # SUB challenger round trip
    ok, seg = _delta(L, fine, base, "sub")
    chk("C1 SUB delta one bit flipped: exact", ok, seg)
    # G7 red paths: a flipped byte in a coded chunk; a truncated container
    arr = _bf16(rng.normal(0, 0.02, 300000))
    ok, seg, (man, blob) = _single(L, arr)
    c0 = seg["hi"]["ch"][0]
    bad = blob.copy(); bad[c0 + 64 + 100] ^= 0x5A
    out, err = decode(L, man, bad, [], 4)
    a8 = arr.astype(np.uint16).view(np.uint8)
    red = err > 0 or out.tobytes() != a8.tobytes()
    chk("G7a flipped byte in a coded chunk -> lane-state error or MISMATCH (never silent PASS)", ok and red, (err,))
    chk("G7a' ... and the lane-state check itself fired (err>0)", err > 0, err)
    trunc = blob[:blob.size // 2].copy()
    chk("G7b truncated container refused before decode", check_container(man, trunc) is not None)
    # G8 mini safetensors file round trip (BF16, F16, F32, metadata)
    t_bf = _bf16(rng.normal(0, 1, 15)).view(np.uint8)
    t_f16 = rng.normal(0, 1, 7).astype(np.float16).view(np.uint8)
    t_f32 = rng.normal(0, 1, 4).astype(np.float32).view(np.uint8)
    hdr = {"__metadata__": {"format": "pt"}, "a": {"dtype": "BF16", "shape": [3, 5], "data_offsets": [0, 30]},
           "b": {"dtype": "F16", "shape": [7], "data_offsets": [30, 44]},
           "c": {"dtype": "F32", "shape": [2, 2], "data_offsets": [44, 60]}}
    hb_ = json.dumps(hdr).encode()
    hb_ += b" " * (-len(hb_) % 8)
    fb = np.frombuffer(struct.pack("<Q", len(hb_)) + hb_ + t_bf.tobytes() + t_f16.tobytes() + t_f32.tobytes(), np.uint8).copy()
    w = Writer()
    F = encode_file(L, w, fb, "mini.safetensors", None, 2)
    man, blob = w.finish()
    man = json.loads(man)
    out, err = decode(L, man, blob, [], 2)
    kinds = [s["k"] for s in F["segs"]]
    chk("G8 mini safetensors sha256 exact, F32 RAW", err == 0 and sha(out) == sha(fb) and kinds.count("r") == 2, (err, kinds))
    # container file save/load round trip
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.bbl")
        save_container(p, json.dumps(man).encode(), blob)
        man2, blob2 = load_container(p)
        out2, err2 = decode(L, man2, blob2, [], 2)
        chk("G8b container save/load round trip exact", err2 == 0 and sha(out2) == sha(fb), err2)
    # ---------------- v2 goldens ----------------
    r84 = np.random.default_rng(3284)

    def single9(arr16, force=False, nt=4):
        w = Writer()
        a8 = np.ascontiguousarray(arr16.astype(np.uint16)).view(np.uint8)
        seg, _ = encode_tensor(L, w, a8, None, arr16.size, "plain", nt, exp9=True, force9=force)
        seg["o"] = 0
        w.files.append({"name": "t", "size": int(a8.size), "segs": [seg]})
        man, blob = w.finish()
        man = json.loads(man)
        out, err = decode(L, man, blob, [], nt)
        return (err == 0 and out.tobytes() == a8.tobytes()), seg, (man, blob, a8)

    n = 1 << 20
    m7 = r84.integers(0, 128, n).astype(np.uint16)
    s9 = np.where(r84.integers(0, 2, n) == 1, 0x081, 0x07E).astype(np.uint16)
    ok, seg, _ = single9((s9 << 7) | m7)
    chk("GE1 E chosen: rans9 + pack7 of 917,504 B, s9 payload 131,072 +- 512 B, exact",
        ok and seg["hi"]["m"] == "rans9" and seg["lo"]["m"] == "pack7" and seg["lo"]["bytes"] == 917504
        and abs(seg["hi"]["bytes"] - 1024 - 131072) <= 512, (ok, seg["hi"]["m"], seg["hi"].get("bytes"), seg["lo"].get("bytes")))
    s9b = np.where(r84.integers(0, 2, n) == 1, 0x07F, 0x07E).astype(np.uint16)
    ok, seg, _ = single9((s9b << 7) | r84.integers(0, 128, n).astype(np.uint16))
    chk("GE2 E refused when it loses: HB hi CONST + lo RAW, payload exactly 1,048,576 B, exact",
        ok and seg["hi"]["m"] == "const" and seg["lo"]["m"] == "raw" and payload(seg) == 1048576,
        (ok, seg["hi"]["m"], seg["lo"]["m"], payload(seg)))
    ok, seg, _ = single9(allp, force=True)
    chk("GE3 all 65,536 patterns x3, E forced: exact (NaN payloads, +-Inf, +-0, subnormals)", ok and seg["hi"]["m"] == "rans9",
        (ok, seg["hi"]["m"]))
    g_ok = True
    for nn in (4096, 4097, 4103, 65535, 65536, 65537, 262143, 262144, 262145, 1_000_003):
        ok, seg, _ = single9(_bf16(r84.normal(0, 0.02, nn)), force=True)
        g_ok &= ok and seg["hi"]["m"] == "rans9"
        if not ok:
            print("   GE4 failed at n=%d" % nn)
    chk("GE4 lengths 4,096..1,000,003 Gaussian bf16, E forced: exact", g_ok)
    ok, seg, (man, blob, a8) = single9(_bf16(r84.normal(0, 0.02, 300000)), force=True)
    bad = blob.copy(); bad[seg["hi"]["ch"][0] + 64 + 100] ^= 0x5A
    out, err = decode(L, man, bad, [], 4)
    chk("GE5a flipped byte in a rans9 chunk -> lane-state error (err > 0)", ok and err > 0, err)
    bad = blob.copy(); bad[seg["lo"]["off"] + 1000] ^= 0x21
    out, err = decode(L, man, bad, [], 4)
    chk("GE5b flipped byte in the pack7 region -> sha256 MISMATCH (the sha is its check)", ok and sha(out) != sha(a8), err)
    base = _bf16(r84.normal(0, 0.02, n))
    fine = base.copy(); fine[12345] ^= np.uint16(1 << 9)
    w = Writer()
    fa, ba = np.ascontiguousarray(fine).view(np.uint8), np.ascontiguousarray(base).view(np.uint8)
    seg, _ = encode_tensor(L, w, fa, ba, n, "xor", 4, exp9=True, force9=True)
    seg["o"] = 0; seg["src"] = [0, 0]
    w.files.append({"name": "t", "size": int(fa.size), "segs": [seg]})
    man, blob = w.finish(); man = json.loads(man)
    out, err = decode(L, man, blob, [ba], 4)
    chk("GE6 XOR delta (exponent bit flipped), E forced: exact", err == 0 and seg["hi"]["m"] == "rans9" and out.tobytes() == fa.tobytes(),
        (err, seg["hi"]["m"]))

    def mk_st(tensors):
        """[(name, dtype, shape, bytes)] -> safetensors file bytes (u8 array)."""
        hdr, off, body = {}, 0, b""
        for (nm, dt, shp, by) in tensors:
            hdr[nm] = {"dtype": dt, "shape": list(shp), "data_offsets": [off, off + len(by)]}
            off += len(by); body += by
        hb_ = json.dumps(hdr).encode()
        hb_ += b" " * (-len(hb_) % 8)
        return np.frombuffer(struct.pack("<Q", len(hb_)) + hb_ + body, np.uint8).copy()

    emb = _bf16(r84.normal(0, 0.02, 64 * 128)).tobytes()
    nrm = _bf16(r84.normal(1, 0.1, 8192)).tobytes()

    def dd_round(fb, name="mini.safetensors"):
        plan = plan_dedup(L, [(name, fb)])
        w = Writer()
        F = encode_file(L, w, fb, name, None, 2, dedup=plan)
        man, blob = w.finish(); man = json.loads(man)
        out, err = decode(L, man, blob, [], 2)
        return err == 0 and check_container(man, blob) is None and sha(out) == sha(fb), [sg for sg in F["segs"] if sg["k"] == "d"], w

    fb1 = mk_st([("model.embed_tokens.weight", "BF16", (64, 128), emb), ("lm_head.weight", "BF16", (64, 128), emb),
                 ("model.norm.weight", "BF16", (8192,), nrm)])
    ok, dups, _ = dd_round(fb1)
    e_off = st_entries_bytes(fb1)[1][0][3]
    chk("GD1 within-file DUP: exactly 1 (lm_head -> embed's offset), exact", ok and len(dups) == 1
        and dups[0]["name"] == "lm_head.weight" and dups[0]["ref"] == ["mini.safetensors", e_off], (ok, dups))
    near = bytearray(emb); near[2 * 4000] ^= 0x01
    fb2 = mk_st([("model.embed_tokens.weight", "BF16", (64, 128), emb), ("lm_head.weight", "BF16", (64, 128), bytes(near)),
                 ("model.norm.weight", "BF16", (8192,), nrm)])
    ok, dups, _ = dd_round(fb2)
    chk("GD2 near-duplicate (one bit): 0 DUP, exact", ok and len(dups) == 0, (ok, dups))
    real_hash = globals()["dedup_hash"]
    try:
        globals()["dedup_hash"] = lambda b: b"collide"
        fb3 = mk_st([("a", "BF16", (64, 128), emb), ("b", "BF16", (64, 128), _bf16(r84.normal(0, 0.02, 8192)).tobytes()),
                     ("c", "BF16", (8192,), nrm)])
        ok3, dups3, _ = dd_round(fb3)
        ok1, dups1, _ = dd_round(fb1)
    finally:
        globals()["dedup_hash"] = real_hash
    chk("GD3 forced hash collision: distinct tensors 0 DUP + exact; GD1's tensors still exactly 1 DUP",
        ok3 and len(dups3) == 0 and ok1 and len(dups1) == 1, (ok3, dups3, ok1, len(dups1)))
    # GD5 (post-freeze challenger rule): lm_head FIRST in the file (Pythagoras' layout); prefer embed as the root
    fb5 = mk_st([("lm_head.weight", "BF16", (64, 128), emb), ("model.embed_tokens.weight", "BF16", (64, 128), emb)])
    e5 = {nm: off for (nm, dt, shp, off, nb) in st_entries_bytes(fb5)[1]}
    p_pos = plan_dedup(L, [("m.safetensors", fb5)])
    p_tw = plan_dedup(L, [("m.safetensors", fb5)], prefer=lambda k: k == "model.embed_tokens.weight")
    w = Writer()
    encode_file(L, w, fb5, "m.safetensors", None, 2, dedup=p_tw)
    man, blob = w.finish(); man = json.loads(man)
    out, err = decode(L, man, blob, [], 2)
    chk("GD5 root preference: positional DUP = embed -> lm_head; preferred DUP = lm_head -> embed; exact",
        list(p_pos.values())[0][3] == "model.embed_tokens.weight" and list(p_tw.values())[0][3] == "lm_head.weight"
        and list(p_tw.values())[0][1] == e5["model.embed_tokens.weight"] and err == 0 and sha(out) == sha(fb5),
        (p_pos, p_tw, err))
    # GD4: cross-file DUP through the real entry point (the function --roundtrip-v2 calls)
    with tempfile.TemporaryDirectory() as td:
        md = os.path.join(td, "models", "T__tiny")
        os.makedirs(md)
        other = _bf16(r84.normal(0, 0.05, 32 * 128)).tobytes()
        fa_ = mk_st([("model.embed_tokens.weight", "BF16", (64, 128), emb)])
        fb_ = mk_st([("lm_head.weight", "BF16", (64, 128), emb), ("model.x.weight", "BF16", (32, 128), other)])
        fa_.tofile(os.path.join(md, "model-00001-of-00002.safetensors"))
        fb_.tofile(os.path.join(md, "model-00002-of-00002.safetensors"))
        json.dump({"weight_map": {"model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                                  "lm_head.weight": "model-00002-of-00002.safetensors",
                                  "model.x.weight": "model-00002-of-00002.safetensors"}},
                  open(os.path.join(md, "model.safetensors.index.json"), "w"))
        oj = os.path.join(td, "rt.json")
        with open(os.devnull, "w") as dn:
            so = sys.stdout
            sys.stdout = dn
            try:
                rc4 = roundtrip_v2(os.path.join(td, "models"), None, oj, 2)
            finally:
                sys.stdout = so
        R4 = json.load(open(oj))["models"]["T__tiny"]
        allok = rc4 == 0 and all(r["ok"] for arm in R4["files"].values() for r in arm.values())
        dl = R4["dups"]
        cross = (len(dl) == 1 and dl[0]["file"] == "model-00002-of-00002.safetensors"
                 and dl[0]["ref_file"] == "model-00001-of-00002.safetensors")
        plan = plan_dedup(L, [("model-00001-of-00002.safetensors", fa_), ("model-00002-of-00002.safetensors", fb_)])
        w = Writer()
        encode_file(L, w, fb_, "model-00002-of-00002.safetensors", None, 2, dedup=plan)
        man, blob = w.finish(); man = json.loads(man)
        ro = dl[0]["ref_off"] if dl else 0
        zero_src = {("model-00001-of-00002.safetensors", ro): np.zeros(len(emb), np.uint8)}
        out, err = decode(L, man, blob, [], 2, dup_srcs=zero_src)
        zero_red = sha(out) != sha(fb_)
        try:
            decode(L, man, blob, [], 2, dup_srcs={})
            absent_red = False
        except ValueError:
            absent_red = True
    chk("GD4 cross-file DUP via roundtrip_v2: every arm EXACT, 1 DUP (shard 2 -> shard 1); zeroed ref -> MISMATCH; absent ref -> raises",
        allok and cross and zero_red and absent_red, (rc4, allok, dl, zero_red, absent_red))
    npass = sum(res)
    print("SELF-TEST %s %d/%d" % ("PASS" if npass == len(res) else "FAIL", npass, len(res)))
    return 0 if npass == len(res) else 1


# ------------------------------------------------------------------------------------------------ round trip over real checkpoints
def model_files(md):
    fs = sorted(f for f in os.listdir(md) if f.endswith(".safetensors"))
    idx = os.path.join(md, "model.safetensors.index.json")
    if os.path.exists(idx):
        iset = sorted(set(json.load(open(idx))["weight_map"].values()))
    else:
        iset = fs
    return fs, iset


def mm(path):
    return np.memmap(path, dtype=np.uint8, mode="r")


def ref_index(md):
    """name -> (file, dtype, shape, abs_off, nbytes) over the index set."""
    _, iset = model_files(md)
    out = {}
    for f in iset:
        _, ents = st_entries(os.path.join(md, f))
        for (k, dt, shp, off, nb) in ents:
            out[k] = (f, dt, shp, off, nb)
    return out


def stream_summary(seg):
    return {"n": seg["n"], "op": seg["op"], "hi_m": seg["hi"]["m"], "lo_m": seg["lo"]["m"],
            "hi_b": seg["hi"]["bytes"], "lo_b": seg["lo"]["bytes"]}


def roundtrip(models_dir, out_json, keep_dir, keep, nt):
    L = build_lib()
    names = sorted(d for d in os.listdir(models_dir) if any(f.endswith(".safetensors") for f in os.listdir(os.path.join(models_dir, d))))
    order = [n for n in REFS if n in names] + [n for n in names if n not in REFS]
    refs = {r: ref_index(os.path.join(models_dir, r)) for r in REFS if r in names}
    refmm = {(r, f): mm(os.path.join(models_dir, r, f)) for r in refs for f in model_files(os.path.join(models_dir, r))[1]}
    result = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "threads": nt, "models": {},
              "consts": {"scale": SCALE, "lanes": LANES, "chunk": CH, "code_if_bits_le": CODE_IF_BITS_LE}}
    if os.path.exists(out_json):
        result = json.load(open(out_json))
    mismatches = 0
    for name in order:
        if name in result["models"] and result["models"][name].get("done"):
            print("skip (done)", name, flush=True)
            continue
        md = os.path.join(models_dir, name)
        fs, iset = model_files(md)
        R = {"files": {}, "index_set": iset, "tensors": {}, "delta": {}}
        t0 = time.time()
        # ---- pass A: standalone, every file
        for f in fs:
            p = os.path.join(md, f)
            fb = mm(p)
            tA = time.time()
            h0 = sha(fb)
            tens = R["tensors"]

            def cb(k, dt, n, seg, joint, f=f):
                Jr = joint.reshape(256, 256)
                tens[f + "::" + k] = dict(stream_summary(seg), dtype=dt,
                                          H_hi=H(Jr.sum(1)), H_lo=H(Jr.sum(0)), H16=H(joint),
                                          H_t9=H(joint.reshape(512, 128).sum(1)),
                                          hi_hist=Jr.sum(1).astype(np.int64).tolist())
            w = Writer()
            encode_file(L, w, fb, f, None, nt, stats_cb=cb)
            man, blob = w.finish()
            tE = time.time()
            mand = json.loads(man)
            bad = check_container(mand, blob)
            out, err = decode(L, mand, blob, [], nt)
            tD = time.time()
            h1 = sha(out)
            fd = int(L.bbl_first_diff(ptr(out), ptr(np.asarray(fb)), fb.size)) if h1 != h0 else -1
            ok = (h1 == h0 and err == 0 and bad is None and fd == -1)
            if not ok:
                mismatches += 1
                print("MISMATCH standalone %s/%s err=%d bad=%s first_diff=%d" % (name, f, err, bad, fd), flush=True)
            R["files"][f] = {"size": int(fb.size), "sha256": h0, "decoded_sha256": h1, "ok": ok, "lane_err": err,
                             "container_bytes": int(blob.size - PAD + len(man) + 16), "manifest_bytes": len(man),
                             "in_index": f in iset, "enc_s": round(tE - tA, 2), "dec_s": round(tD - tE, 3)}
            print("%-40s %-36s %s saving %.4f enc %.0fs dec %.2fs" % (name[:40], f, "OK" if ok else "MISMATCH",
                  1 - R["files"][f]["container_bytes"] / fb.size, tE - tA, tD - tE), flush=True)
            if keep_dir and name in keep.get("std", ()) and f in iset:
                save_container(os.path.join(keep_dir, "%s__%s.std.bbl" % (name, f)), man, blob)
            del out, blob, w, fb
        # ---- pass B: deltas (non-refs, index set only), XD primary + SUB challenger, base choice by estimate
        if name not in REFS and refs:
            mine = {}
            for f in iset:
                for (k, dt, shp, off, nb) in st_entries(os.path.join(md, f))[1]:
                    mine[k] = (f, dt, shp, off, nb)
            est, fmm = {}, {}
            for r, ri in refs.items():
                tot = {"xor": 0.0, "sub": 0.0}
                per = {}
                for k, (f, dt, shp, off, nb) in mine.items():
                    e = ri.get(k)
                    n = nb // 2
                    fbm = fmm.setdefault(f, mm(os.path.join(md, f)))[off:off + nb]
                    if e is None or e[2] != shp or not (e[1] == dt or (dt == "F16" and e[1] == "BF16")) or dt not in SIXTEEN:
                        st = R["tensors"].get(f + "::" + k)
                        sb = (st["hi_b"] + st["lo_b"]) if st else nb
                        tot["xor"] += sb; tot["sub"] += sb
                        continue
                    sbm = refmm[(r, e[0])][e[3]:e[3] + e[4]]
                    cast = (dt == "F16" and e[1] == "BF16")
                    for opn in ("xor", "sub"):
                        op = (opn + "cast") if cast else opn
                        j = np.zeros(65536, np.uint64)
                        L.bbl_hist(ptr(np.asarray(fbm)), ptr(np.asarray(sbm)), n, OPS[op], ptr(j), nt)
                        Jr = j.reshape(256, 256)
                        b = est_stream_bytes(Jr.sum(1), n) + est_stream_bytes(Jr.sum(0), n)
                        tot[opn] += b
                        per.setdefault(k, {})[opn] = {"H_hi": H(Jr.sum(1)), "H_lo": H(Jr.sum(0)), "H16": H(j),
                                                      "same": int(j[0]), "n": n, "est_bytes": b, "op": op}
                est[r] = {"tot": tot, "per": per}
            base = min(est, key=lambda r: (est[r]["tot"]["xor"], REFS.index(r)))
            R["delta"]["est_tot"] = {r: est[r]["tot"] for r in est}
            R["delta"]["base"] = base
            R["delta"]["per_tensor_bounds"] = est[base]["per"]
            bfiles = model_files(os.path.join(models_dir, base))[1]
            src_list = [refmm[(base, bf)] for bf in bfiles]
            bidx = {bf: i for i, bf in enumerate(bfiles)}
            ri = refs[base]
            for opn in ("xor", "sub"):
                R["delta"][opn] = {}
                for f in iset:
                    p = os.path.join(md, f)
                    fb = mm(p)

                    def srcmap(k, dt, shp, opn=opn):
                        e = ri.get(k)
                        if e is None or e[2] != shp or dt not in SIXTEEN:
                            return None
                        if e[1] == dt:
                            op = opn
                        elif dt == "F16" and e[1] == "BF16":
                            op = opn + "cast"
                        else:
                            return None
                        return (bidx[e[0]], e[3], op, np.asarray(refmm[(base, e[0])][e[3]:e[3] + e[4]]))
                    tens = {}

                    def cb(k, dt, n, seg, joint, tens=tens):
                        tens[k] = dict(stream_summary(seg), ident=bool(seg["hi"]["m"] == seg["lo"]["m"] == "const"
                                                                      and seg["hi"].get("c") == 0 and seg["lo"].get("c") == 0
                                                                      and seg["op"] != "plain"))
                    w = Writer()
                    tA = time.time()
                    encode_file(L, w, fb, f, srcmap, nt, stats_cb=cb)
                    man, blob = w.finish()
                    mand = json.loads(man)
                    bad = check_container(mand, blob)
                    out, err = decode(L, mand, blob, src_list, nt)
                    h1 = sha(out)
                    ok = (h1 == R["files"][f]["sha256"] and err == 0 and bad is None)
                    if not ok:
                        mismatches += 1
                        print("MISMATCH delta-%s %s/%s err=%d" % (opn, name, f, err), flush=True)
                    R["delta"][opn][f] = {"ok": ok, "lane_err": err, "decoded_sha256": h1,
                                          "container_bytes": int(blob.size - PAD + len(man) + 16),
                                          "manifest_bytes": len(man), "tensors": tens, "secs": round(time.time() - tA, 1)}
                    print("%-40s %-36s delta-%s vs %s %s %.3f b/p" % (name[:40], f, opn, base[6:], "OK" if ok else "MISMATCH",
                          8 * R["delta"][opn][f]["container_bytes"] / (fb.size / 2)), flush=True)
                    if keep_dir and name in keep.get("xd", ()) and opn == "xor":
                        save_container(os.path.join(keep_dir, "%s__%s.xd.bbl" % (name, f)), man, blob)
                    del out, blob, w
        R["secs"] = round(time.time() - t0, 1)
        R["done"] = True
        result["models"][name] = R
        json.dump(result, open(out_json + ".tmp", "w"))
        os.replace(out_json + ".tmp", out_json)
    result["mismatches"] = mismatches
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(result, open(out_json + ".tmp", "w"))
    os.replace(out_json + ".tmp", out_json)
    allok = all(F["ok"] for M in result["models"].values() for F in M["files"].values()) and all(
        D["ok"] for M in result["models"].values() for o in ("xor", "sub") for D in M["delta"].get(o, {}).values())
    nfiles = sum(len(M["files"]) for M in result["models"].values())
    print("RESULT: roundtrip %s files=%d models=%d mismatches=%d" % ("EXACT" if allok else "MISMATCH", nfiles,
                                                                       len(result["models"]), mismatches))
    return 0 if allok else 1


# ------------------------------------------------------------------------------------------------ codec v2 round trip
ARMS_V2 = (("INC", False, False), ("D", True, False), ("E", False, True), ("DE", True, True))   # (arm, dedup, exp9)


def _run_arm(L, fmm, files, shas, plan, e9, nt, srcmap=None, src_list=(), keep_path=None, label=""):
    """Encode + decode + sha-verify each file of ONE checkpoint group under one arm, in group order. Cross-file DUP sources
    are the DECODED bytes of the referenced file from this same arm, kept only for the referenced tensors."""
    keep_dec, rows = {}, {}
    need = {(v[0], v[1], v[4]) for (kf, ko), v in plan.items() if v[0] != kf} if plan else set()
    for f in files:
        st = {"n_rans9": 0, "n_dup": 0, "dup_bytes": 0, "bound_e": 0.0, "bound_hb": 0.0, "coded16_bytes": 0, "dtypes": {}}

        def cb(k, dt, n, seg, joint, st=st):
            st["dtypes"][dt] = st["dtypes"].get(dt, 0) + 1
            if seg["k"] == "d":
                st["n_dup"] += 1; st["dup_bytes"] += seg["n"]
                return
            if seg["hi"]["m"] == "rans9":
                st["n_rans9"] += 1
            if joint is not None and n >= MIN_CODED_ELEMS:
                hh, h9 = H(joint.reshape(256, 256).sum(1)), H(joint.reshape(512, 128).sum(1))
                st["bound_hb"] += n * (hh + 8) / 8
                st["bound_e"] += n * min(hh + 8, h9 + 7) / 8
                st["coded16_bytes"] += 2 * n
        w = Writer()
        t0 = time.time()
        encode_file(L, w, fmm[f], f, srcmap, nt, stats_cb=cb, exp9=e9, dedup=plan)
        man, blob = w.finish()
        mand = json.loads(man)
        bad = check_container(mand, blob)
        t1 = time.time()
        try:
            out, err = decode(L, mand, blob, list(src_list), nt, dup_srcs=keep_dec)
            h1 = sha(out)
        except Exception as ex:                                     # QA C5: record, never abort the model
            out, err, h1, bad = None, -1, None, (bad or "%s: %s" % (type(ex).__name__, ex))
        ok = (h1 == shas[f] and err == 0 and bad is None)
        if out is not None:
            for (rf, ro, nb) in need:
                if rf == f:
                    keep_dec[(rf, ro)] = out[ro:ro + nb].copy()
        rows[f] = dict(st, ok=ok, lane_err=int(err), bad=bad, decoded_sha256=h1,
                       container_bytes=int(blob.size - PAD + len(man) + 16), manifest_bytes=len(man),
                       enc_s=round(t1 - t0, 2), dec_s=round(time.time() - t1, 2))
        if keep_path:
            save_container(keep_path % f, man, blob)
        print("%-44s %-36s %-6s %s %.4f b/p%s" % (label[:44], f, "", "OK" if ok else "MISMATCH",
              8 * rows[f]["container_bytes"] / (fmm[f].size / 2), "" if ok else "  err=%s bad=%s" % (err, bad)), flush=True)
        del out, blob, w
    return rows


def roundtrip_v2(models_dir, evalset_json, out_json, nt, only=None, keep_dir=None, keep_e=(), xd_prefer_twin=False):
    """Arms INC / D / E / DE standalone on every file and
    XD-INC / XD-D / XD-E / XD-DE on each fine-tune's index set (base = the v1 run's recorded base, from the eval-set manifest).
    The input files are checked against the manifest's sha256 first (EVALSET-FAIL = exit 1, nothing else runs).
    WRITES out_json (resumable: models marked done are skipped) and, for models in keep_e, the E containers into keep_dir.
    xd_prefer_twin (POST-FREEZE named challenger, off by default): ALSO run XD-D / XD-DE with a dedup root that prefers the copy
    which has a same-name base tensor (so the cheap XD copy is kept and the twinless one becomes the DUP), stored as "xd_pt"."""
    L = build_lib()
    ev = json.load(open(evalset_json)) if evalset_json else None
    exp_f = {(r["model"], r["file"]): r for r in ev["files"]} if ev else {}
    exp_x = {(r["model"], r["file"]): r for r in ev["xd_files"]} if ev else {}
    names = sorted(d for d in os.listdir(models_dir) if any(f.endswith(".safetensors") for f in os.listdir(os.path.join(models_dir, d))))
    if only:
        names = [n for n in names if n in only]
    order = [n for n in REFS if n in names] + [n for n in names if n not in REFS]
    refs = {r: ref_index(os.path.join(models_dir, r)) for r in REFS if os.path.isdir(os.path.join(models_dir, r))}
    refmm = {(r, f): mm(os.path.join(models_dir, r, f)) for r in refs for f in model_files(os.path.join(models_dir, r))[1]}
    result = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "threads": nt, "models": {},
              "evalset": evalset_json,
              "script_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()}
    if os.path.exists(out_json):
        result = json.load(open(out_json))
    fails = 0
    for name in order:
        if name in result["models"] and result["models"][name].get("done"):
            print("skip (done)", name, flush=True)
            continue
        md = os.path.join(models_dir, name)
        fs, iset = model_files(md)
        extras = [f for f in fs if f not in iset]
        fmm = {f: mm(os.path.join(md, f)) for f in fs}
        shas = {}
        for f in fs:
            shas[f] = sha(fmm[f])
            e = exp_f.get((name, f))
            if ev is not None and (e is None or e["sha256"] != shas[f] or e["size"] != fmm[f].size):
                print("EVALSET-FAIL %s/%s sha %s" % (name, f, shas[f]), flush=True)
                result["evalset_fail"] = [name, f]
                json.dump(result, open(out_json + ".tmp", "w"))
                os.replace(out_json + ".tmp", out_json)
                return 1
        groups = [g for g in (iset, extras) if g]
        plans = [plan_dedup(L, [(f, fmm[f]) for f in g]) for g in groups]
        R = {"index_set": iset, "extras": extras, "sha256": shas, "sizes": {f: int(fmm[f].size) for f in fs},
             "files": {}, "xd": {}, "dups": []}
        for gi, plan in enumerate(plans):
            for (kf, ko), v in sorted(plan.items()):
                R["dups"].append({"scope": "index" if gi == 0 else "extra", "file": kf, "off": int(ko), "name": v[3],
                                  "nbytes": int(v[4]), "ref_file": v[0], "ref_off": int(v[1]), "ref_name": v[2]})
        t0 = time.time()
        for (arm, dd, e9) in ARMS_V2:
            R["files"][arm] = {}
            for gi, g in enumerate(groups):
                kp = os.path.join(keep_dir, "%s__%%s.%s.bbl" % (name, arm.lower())) if (keep_dir and name in keep_e and arm == "E") else None
                R["files"][arm].update(_run_arm(L, fmm, g, shas, plans[gi] if dd else None, e9, nt, keep_path=kp,
                                                label="%s %s" % (name, arm)))
        for f in fs:
            e = exp_f.get((name, f))
            R["files"]["INC"][f]["replay_expected"] = e["inc_container_bytes"] if e else None
        bx = [exp_x[(name, f)]["base"] for f in iset if (name, f) in exp_x]
        if name not in REFS and bx and bx[0] in refs:
            base = bx[0]
            ri = refs[base]
            bfiles = model_files(os.path.join(models_dir, base))[1]
            src_list = [refmm[(base, bf)] for bf in bfiles]
            bidx = {bf: i for i, bf in enumerate(bfiles)}

            def srcmap(k, dt, shp, ri=ri, base=base, bidx=bidx):
                e = ri.get(k)
                if e is None or e[2] != shp or dt not in SIXTEEN:
                    return None
                if e[1] == dt:
                    op = "xor"
                elif dt == "F16" and e[1] == "BF16":
                    op = "xorcast"
                else:
                    return None
                return (bidx[e[0]], e[3], op, np.asarray(refmm[(base, e[0])][e[3]:e[3] + e[4]]))
            R["xd_base"] = base
            for (arm, dd, e9) in ARMS_V2:
                R["xd"][arm] = _run_arm(L, fmm, iset, shas, plans[0] if dd else None, e9, nt, srcmap=srcmap,
                                        src_list=src_list, label="%s XD-%s" % (name, arm))
            for f in iset:
                R["xd"]["INC"][f]["replay_expected"] = exp_x[(name, f)]["inc_xd_container_bytes"] if (name, f) in exp_x else None
            if xd_prefer_twin:
                plan_pt = plan_dedup(L, [(f, fmm[f]) for f in iset], prefer=lambda k, ri=ri: k in ri)
                R["dups_pt"] = [{"file": kf, "off": int(ko), "name": v[3], "ref_file": v[0], "ref_off": int(v[1]), "ref_name": v[2]}
                                for (kf, ko), v in sorted(plan_pt.items())]
                R["xd_pt"] = {arm: _run_arm(L, fmm, iset, shas, plan_pt, e9, nt, srcmap=srcmap, src_list=src_list,
                                            label="%s XD-%s-PT" % (name, arm)) for (arm, dd, e9) in ARMS_V2 if dd}
        R["secs"] = round(time.time() - t0, 1)
        R["done"] = True
        fails += sum(1 for part in ("files", "xd", "xd_pt") for arm in R.get(part, {}).values() for r in arm.values() if not r["ok"])
        result["models"][name] = R
        json.dump(result, open(out_json + ".tmp", "w"))
        os.replace(out_json + ".tmp", out_json)
    allok = all(r["ok"] for M in result["models"].values() for part in ("files", "xd", "xd_pt")
                for arm in M.get(part, {}).values() for r in arm.values())
    rep = [(M_, part, f, r["container_bytes"], r["replay_expected"]) for M_, M in result["models"].items()
           for part in ("files", "xd") for f, r in M.get(part, {}).get("INC", {}).items() if r.get("replay_expected") is not None]
    rbad = [x for x in rep if x[3] != x[4]]
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result["replay"] = {"n": len(rep), "mismatch": rbad}
    json.dump(result, open(out_json + ".tmp", "w"))
    os.replace(out_json + ".tmp", out_json)
    nrows = sum(len(arm) for M in result["models"].values() for part in ("files", "xd", "xd_pt") for arm in M.get(part, {}).values())
    print("RESULT: roundtrip_v2 %s rows=%d models=%d replay=%d/%d%s" % ("EXACT" if allok else "MISMATCH", nrows,
          len(result["models"]), len(rep) - len(rbad), len(rep), "" if not rbad else " REPLAY-FAIL %s" % rbad[:3]))
    return 0 if (allok and not rbad) else 1


def fp8_bound(md):
    """order-0 byte-entropy bound of the F8_E4M3 tensors (n >= 4,096) of one checkpoint's index set; every other
    byte counted raw. -> dict (both readings of the r8 formula). Reads md only."""
    _, iset = model_files(md)
    raw = coded8 = n8 = 0
    bf16_bound = bf16_n = 0.0
    dts = {}
    for f in iset:
        p = os.path.join(md, f)
        fb = mm(p)
        raw += fb.size
        for (k, dt, shp, off, nb) in st_entries(p)[1]:
            dts[dt] = dts.get(dt, 0) + nb
            if dt == "F8_E4M3" and nb >= MIN_CODED_ELEMS:
                c = np.bincount(np.asarray(fb[off:off + nb]), minlength=256)
                coded8 += nb * H(c) / 8
                n8 += nb
            elif dt == "BF16" and nb >= 2 * MIN_CODED_ELEMS:
                j = np.asarray(fb[off:off + nb]).view(np.uint16)
                c9 = np.bincount(j >> 7, minlength=512)
                bf16_bound += (nb // 2) * (H(c9) + 7) / 8
                bf16_n += nb
    intended = (raw - n8 + coded8) / raw
    return {"raw_bytes": int(raw), "f8_bytes": int(n8), "f8_coded_bound_bytes": coded8, "dtype_bytes": dts,
            "r8_intended": intended, "r8_literal": coded8 / raw,
            "r8_with_bf16_at_e_bound_postfreeze": (raw - n8 + coded8 - bf16_n + bf16_bound) / raw}


def summarize_v2(rt_path, bench_path=None, fp8_json=None):
    """The v2 pass bars (C3) over a --roundtrip-v2 JSON (+ the bench JSON for the descriptive speed rows). Reads only.
    exit 1 if not EXACT or REPLAY fails (then no size claim stands)."""
    R = json.load(open(rt_path))
    exact = all(r["ok"] for M in R["models"].values() for part in ("files", "xd") for a in M[part].values() for r in a.values())
    rp = R.get("replay") or {}
    replay_ok = bool(rp.get("n")) and not rp.get("mismatch")
    S = {"exact": exact, "replay_ok": replay_ok, "replay_n": rp.get("n"), "models": {}, "dups": []}
    fam = {a: [0, 0] for a in ("INC", "D", "E", "DE", "DE-PT")}  # [standalone family bytes, family bytes with deltas]
    raw_fam = 0
    for name in sorted(R["models"]):
        M = R["models"][name]
        iset = M["index_set"]
        raw = sum(M["sizes"][f] for f in iset)
        P = raw / 2.0
        raw_fam += raw
        b = lambda part, arm: sum(M[part][arm][f]["container_bytes"] for f in iset) if arm in M.get(part, {}) else None
        inc = M["files"]["INC"]
        f16 = sum(inc[f]["dtypes"].get("F16", 0) for f in iset)
        bound_e = sum(inc[f]["bound_e"] for f in iset) + raw - sum(inc[f]["coded16_bytes"] for f in iset)
        rec = {"P": P, "bf16": f16 == 0, "xd_base": M.get("xd_base")}
        for arm in ("INC", "D", "E", "DE"):
            rec[arm] = 8 * b("files", arm) / P
            if b("xd", arm) is not None:
                rec["XD-" + arm] = 8 * b("xd", arm) / P
            fam[arm][0] += b("files", arm)
            fam[arm][1] += b("xd", arm) if b("xd", arm) is not None else b("files", arm)
        fam["DE-PT"][0] += b("files", "DE")                 # post-freeze: DE with the twin-preferring XD root where it ran
        fam["DE-PT"][1] += (b("xd_pt", "DE") if b("xd_pt", "DE") is not None else
                            b("xd", "DE") if b("xd", "DE") is not None else b("files", "DE"))
        rec["dE"] = 8 * (b("files", "INC") - b("files", "E")) / P
        rec["eff_E"] = (raw - b("files", "E")) / (raw - bound_e) if raw > bound_e else None   # QA C1
        rec["E_le_INC"] = b("files", "E") <= b("files", "INC")
        if b("xd", "INC") is not None:
            rec["dE_xd"] = 8 * (b("xd", "INC") - b("xd", "E")) / P
        rec["n_rans9_tensors_E"] = sum(M["files"]["E"][f]["n_rans9"] for f in iset)
        for arm, rows_pt in M.get("xd_pt", {}).items():
            rec["XD-%s-PT" % arm] = 8 * sum(rows_pt[f]["container_bytes"] for f in iset) / P
        S["models"][name] = rec
        S["dups"] += [dict(d, model=name) for d in M["dups"]]
    bf = [r for r in S["models"].values() if r["bf16"]]
    dEs = sorted(r["dE"] for r in bf)
    med = (dEs[len(dEs) // 2] if len(dEs) % 2 else (dEs[len(dEs) // 2 - 1] + dEs[len(dEs) // 2]) / 2) if dEs else None
    effs = [r["eff_E"] for r in bf if r["eff_E"] is not None]
    S["E_bar"] = {"n_bf16": len(bf), "median_dE": med, "min_eff_E": min(effs) if effs else None,
                  "all_E_le_INC": all(r["E_le_INC"] for r in bf)}
    S["E_bar"]["PASS"] = bool(bf and effs) and len(effs) == len(bf) and med >= 0.120 and min(effs) >= 0.97 \
        and S["E_bar"]["all_E_le_INC"] and exact and replay_ok                         # QA C12
    pyth = next((r for n_, r in S["models"].items() if n_.startswith("Pythagoras")), None)
    guard = next((r for n_, r in S["models"].items() if "Guard" in n_), None)
    dd = {}
    if pyth and guard:
        dd = {"pyth_XD-D": pyth.get("XD-D"), "guard_XD-D": guard.get("XD-D"), "pyth_D": pyth["D"], "guard_D": guard["D"]}
        dd["PASS"] = (None not in dd.values() and dd["pyth_XD-D"] <= 7.600 and dd["guard_XD-D"] <= 3.840   # QA C2
                      and dd["pyth_D"] <= 9.850 and dd["guard_D"] <= 9.850 and exact and replay_ok)
        dd["pyth_XD-D-PT_postfreeze"], dd["guard_XD-D-PT_postfreeze"] = pyth.get("XD-D-PT"), guard.get("XD-D-PT")
    S["D_bar"] = dd
    tunes = {n_: r for n_, r in S["models"].items() if "XD-INC" in r}
    S["FAM_le8"] = {a: sum(1 for r in tunes.values() if r["XD-" + a] <= 8.0) for a in ("INC", "D", "E", "DE")}
    S["FAM_le8"]["DE-PT_postfreeze"] = sum(1 for r in tunes.values() if r.get("XD-DE-PT", r["XD-DE"]) <= 8.0)
    S["FAM_le8"]["n"] = len(tunes)
    S["family_GB"] = {"raw": raw_fam / 1e9, **{a: {"standalone": fam[a][0] / 1e9, "with_deltas": fam[a][1] / 1e9} for a in fam}}
    if bench_path and os.path.exists(bench_path):
        B = json.load(open(bench_path))
        S["speed"] = {c: {T: round(r["gbps_median"], 2) for T, r in v["rows"].items()} for c, v in B["cases"].items()}
        S["speed_verified"] = all(r.get("verified_every_run", True) for v in B["cases"].values() for r in v["rows"].values())
    if fp8_json and os.path.exists(fp8_json):
        S["fp8"] = json.load(open(fp8_json))
    print(json.dumps(S, indent=1, sort_keys=True))
    print("RESULT: summarize_v2 EXACT=%s REPLAY=%s D=%s E=%s" % (exact, "OK" if replay_ok else "FAIL",
          "PASS" if dd.get("PASS") else "CONCERN", "PASS" if S["E_bar"]["PASS"] else "CONCERN"))
    return 0 if (exact and replay_ok) else 1


# ------------------------------------------------------------------------------------------------ bench
def other_cpu(me):
    """processes other than this one above 50% CPU over a 0.5 s window (via /proc)."""
    def snap():
        d = {}
        for p in os.listdir("/proc"):
            if not p.isdigit():
                continue
            try:
                with open("/proc/%s/stat" % p) as f:
                    s = f.read()
                r = s.rfind(")")
                fl = s[r + 2:].split()
                d[int(p)] = (int(fl[11]) + int(fl[12]), s[s.find("(") + 1:r])
            except (OSError, ValueError, IndexError):
                pass
        return d
    a = snap(); time.sleep(0.5); b = snap()
    hz = os.sysconf("SC_CLK_TCK")
    out = []
    for pid, (t1, nm) in b.items():
        if pid == me or pid not in a:
            continue
        pct = 100.0 * (t1 - a[pid][0]) / hz / 0.5
        if pct > 50:
            out.append((pid, nm, round(pct)))
    return out


def merge_containers(paths):
    """Merge per-file containers into one manifest + blob (files back to back; blob offsets and table ids re-based)."""
    tabs, tabs9, files, blobs, off, v = [], [], [], [], 0, 1
    for pth in paths:
        m, b = load_container(pth)
        b = b[:-PAD]
        t0, t90 = len(tabs), len(tabs9)
        tabs += [t + off for t in m["tables"]]
        tabs9 += [t + off for t in m.get("tables9", [])]
        v = max(v, m.get("v", 1))
        for F in m["files"]:
            F2 = json.loads(json.dumps(F))
            for s in F2["segs"]:
                if s["k"] == "r":
                    s["off"] += off
                    continue
                if s["k"] == "d":           # v2 DUP: refs name files, so they need no re-basing
                    continue
                for st in ("hi", "lo"):
                    if s[st]["m"] in ("raw", "pack7"):
                        s[st]["off"] += off
                    elif s[st]["m"] == "rans":
                        s[st]["ch"] = [c + off for c in s[st]["ch"]]
                        s[st]["t"] += t0
                    elif s[st]["m"] == "rans9":
                        s[st]["ch"] = [c + off for c in s[st]["ch"]]
                        s[st]["t9"] += t90
            files.append(F2)
        blobs.append(b)
        off += b.size
    man = {"v": v, "tables": tabs, "files": files}
    if v >= 2:
        man["tables9"] = tabs9
    return man, np.concatenate(blobs + [np.zeros(PAD, np.uint8)])


def verify_files(out, sizes, shas):
    """sha256 of each file's slice of out, files hashed in parallel threads (hashlib releases the GIL)."""
    from concurrent.futures import ThreadPoolExecutor
    offs = np.cumsum([0] + list(sizes))
    with ThreadPoolExecutor(max(1, len(sizes))) as ex:
        got = list(ex.map(lambda i: sha(out[offs[i]:offs[i + 1]]), range(len(sizes))))
    return got == list(shas)


def cpu_name():
    """the CPU label for a bench row: x86 'model name', arm 'Model', else the machine string."""
    import platform
    try:
        txt = open("/proc/cpuinfo").read()
    except OSError:
        txt = ""
    for key in ("model name", "Model"):
        for line in txt.splitlines():
            if line.startswith(key):
                return line.split(":", 1)[1].strip()
    return platform.machine()


def bench(keep_dir, out_json, models_dir, ref_json, grid, runs):
    L = build_lib()
    me = os.getpid()
    rt = json.load(open(ref_json))
    groups = {}
    for f in sorted(os.listdir(keep_dir)):
        if f.endswith(".bbl"):
            name, rest = f.rsplit("__", 1)          # model names contain "__" (Qwen__Qwen3-4B); the file part never does
            groups.setdefault((name, rest.rsplit(".", 2)[-2]), []).append(os.path.join(keep_dir, f))
    res = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "grid": grid, "runs": runs,
           "cpu": cpu_name(),
           "nproc": os.cpu_count(), "cases": {}}

    def srcarr(srcs):
        return (ctypes.c_void_p * max(1, len(srcs)))(*[s.ctypes.data for s in srcs]) if srcs else (ctypes.c_void_p * 1)(None)

    def timed(case, man, blob, srcs, shas, pre=None, extra_out=0):
        tasks, total = tasks_for(man)
        fr, ntab = freqs_from(man, blob)
        fr9, ntab9 = freqs9_from(man, blob)            # v2: 9-bit tables (0 for a v1 container)
        sizes = [F["size"] for F in man["files"]]
        out = prefault(total)                      # the ready (pinned-in-production) buffer exists before the wake
        sp = srcarr(srcs)
        rows = {}
        for T in grid:
            busy = other_cpu(me)
            la = open("/proc/loadavg").read().split()[:3]
            ts, ok_all, first = [], True, None
            for r in range(runs + 1):
                out[:1 << 20] = 0                  # disturb the head so a no-op decode cannot verify
                t0 = time.perf_counter()
                if pre:
                    pre(T)
                err = L.bbl_decode2(ptr(blob), blob.size - PAD, ptr(tasks), tasks.shape[0], ptr(fr), ntab, ptr(fr9), ntab9,
                                    ptr(out), sp, T)
                dup_pass(L, man, out, None, T)              # v2 DUP copies are part of the wake
                t1 = time.perf_counter()
                good = (err == 0) and verify_files(out, sizes, shas)
                ok_all &= good
                if r == 0:
                    first = t1 - t0
                else:
                    ts.append(t1 - t0)
            med = float(np.median(ts))
            nbytes = total + extra_out
            rows[T] = {"median_s": med, "min_s": min(ts), "max_s": max(ts), "first_run_s": first,
                       "gbps_median": total / med / 1e9, "gbps_min": total / max(ts) / 1e9, "gbps_max": total / min(ts) / 1e9,
                       "verified_every_run": ok_all, "loadavg": la, "busy_others": busy, "bytes_written": nbytes}
            print("%-44s T=%2d %6.2f GB/s (%.2f-%.2f) %s%s" % (case, T, rows[T]["gbps_median"], rows[T]["gbps_min"],
                  rows[T]["gbps_max"], "VERIFIED" if ok_all else "VERIFY-FAIL", (" BUSY %s" % busy) if busy else ""), flush=True)
        res["cases"][case] = {"out_bytes": total, "comp_bytes": int(blob.size - PAD), "rows": rows}
        json.dump(res, open(out_json + ".tmp", "w"), indent=1)
        os.replace(out_json + ".tmp", out_json)
        del out

    for (name, kind) in sorted(groups):
        man, blob = merge_containers(groups[(name, kind)])
        shas = [rt["models"][name]["files"][F["name"]]["sha256"] for F in man["files"]]
        if kind in ("std", "e", "de"):                  # v2: E / DE standalone containers time like std
            timed("%s.%s" % (name, kind), man, blob, [], shas)
            continue
        base = rt["models"][name]["delta"]["base"]
        bfiles = model_files(os.path.join(models_dir, base))[1]
        braw = [np.fromfile(os.path.join(models_dir, base, bf), dtype=np.uint8) for bf in bfiles]
        timed("%s.xd.base-raw" % name, man, blob, braw, shas)
        del braw
        if (base, "std") in groups:
            bman, bblob = merge_containers(groups[(base, "std")])
            btasks, btotal = tasks_for(bman)
            bfr, bntab = freqs_from(bman, bblob)
            bout = prefault(btotal)
            offs = np.cumsum([0] + [F["size"] for F in bman["files"]])
            bnames = [F["name"] for F in bman["files"]]
            bviews = [bout[offs[bnames.index(bf)]:offs[bnames.index(bf) + 1]] for bf in bfiles]

            def chain(T, bblob=bblob, btasks=btasks, bfr=bfr, bntab=bntab, bout=bout):
                e = L.bbl_decode(ptr(bblob), bblob.size - PAD, ptr(btasks), btasks.shape[0], ptr(bfr), bntab,
                                 ptr(bout), srcarr([]), T)
                if e:
                    raise RuntimeError("base decode lane error %d" % e)
            timed("%s.xd.base-compressed(chained)" % name, man, blob, bviews, shas, pre=chain)
            del bout, bblob, bviews
        del blob
    big = max(c["out_bytes"] for c in res["cases"].values())
    srcb, dst = prefault(big), prefault(big)
    rows = {}
    for T in grid:
        ts = []
        for r in range(runs + 1):
            t0 = time.perf_counter(); L.bbl_memcpy_mt(ptr(dst), ptr(srcb), big, T); t1 = time.perf_counter()
            if r > 0:
                ts.append(t1 - t0)
        med = float(np.median(ts))
        rows[T] = {"median_s": med, "gbps_median": big / med / 1e9, "gbps_min": big / max(ts) / 1e9, "gbps_max": big / min(ts) / 1e9}
        print("%-44s T=%2d %6.2f GB/s" % ("memcpy (host copy ceiling)", T, rows[T]["gbps_median"]), flush=True)
    res["cases"]["memcpy"] = {"out_bytes": big, "rows": rows}
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(res, open(out_json + ".tmp", "w"), indent=1)
    os.replace(out_json + ".tmp", out_json)
    allv = all(r["verified_every_run"] for c in res["cases"].values() for r in c["rows"].values() if "verified_every_run" in r)
    print("RESULT: bench %s cases=%d" % ("VERIFIED" if allv else "VERIFY-FAIL", len(res["cases"])))
    return 0 if allv else 1


# ------------------------------------------------------------------------------------------------ summarize (report numbers)
P_EFF, P_FLOOR, P12, P5 = 22.6, 27.8, 12.4, 44.0     # GB/s: measured effective wake / pinned floor, a second host, PCIe 5 projection


def band(d_all, t_p):
    if d_all >= P_EFF:
        return "FREE" if (t_p is not None and t_p <= 8) else "FREE-WITH-CPU"
    return "BINDS-ON-PCIe4-HOST" if d_all >= P12 else "BINDS"


def summarize(rt_path, bench_path=None):
    """Print the report tables from a --roundtrip RESULT.json (+ optional --bench BENCH.json) under the pass bars.
    Reads only. -> exit 0 when the round trip is EXACT, 1 otherwise."""
    rt = json.load(open(rt_path))
    M = rt["models"]
    exact = all(F["ok"] for m in M.values() for F in m["files"].values()) and all(
        D["ok"] for m in M.values() for o in ("xor", "sub") for D in m["delta"].get(o, {}).values())
    nfiles = sum(len(m["files"]) for m in M.values())
    nbytes = sum(F["size"] for m in M.values() for F in m["files"].values())
    lane_err = sum(F["lane_err"] for m in M.values() for F in m["files"].values())
    print("EXACT: %s  files=%d (%.1f GB) models=%d lane_errors=%d" % (exact, nfiles, nbytes / 1e9, len(M), lane_err))
    rows, bf_sav, effs = [], [], []
    print("\n%-34s %5s %8s %8s %8s %7s %7s %7s %7s %7s" % ("model (index set)", "dtype", "GB", "saving", "bound",
          "eff", "1tab", "C2", "H16", "lo"))
    for name in sorted(M):
        m = M[name]
        iset = m["index_set"]
        raw = sum(m["files"][f]["size"] for f in iset)
        cont = sum(m["files"][f]["container_bytes"] for f in iset)
        ts = [(k, t) for k, t in m["tensors"].items() if k.split("::")[0] in iset]
        n = sum(t["n"] for _, t in ts)
        dts = sorted({t["dtype"] for _, t in ts})
        tens_bytes = sum(2 * t["n"] for _, t in ts)
        bound = raw - tens_bytes + sum(t["n"] * (t["H_hi"] + 8) / 8 for _, t in ts)
        if dts == ["F16"]:
            bound = raw - tens_bytes + sum(t["n"] * (t["H_hi"] + t["H_lo"]) / 8 for _, t in ts)
        eff = (raw - cont) / (raw - bound)
        g = np.sum([t["hi_hist"] for _, t in ts], axis=0)
        per = sum(t["H_hi"] * t["n"] for _, t in ts) / n
        one_tab = H(g) - per
        c2 = sum((t["H_t9"] + 7) * t["n"] for _, t in ts) / n - sum((t["H_hi"] + 8) * t["n"] for _, t in ts) / n
        h16 = sum(t["H16"] * t["n"] for _, t in ts) / n
        lo_coded = sum(1 for _, t in ts if t["lo_m"] == "rans")
        sav = 1 - cont / raw
        if dts == ["BF16"]:
            bf_sav.append(sav); effs.append(eff)
        print("%-34s %5s %8.3f %8.4f %8.4f %7.4f %7.4f %7.3f %7.3f %4d/%d" % (name[:34], ",".join(dts), raw / 1e9, sav,
              1 - bound / raw, eff, one_tab, c2, h16, lo_coded, len(ts)))
        rows.append((name, raw, cont))
        stray = [f for f in m["files"] if f not in iset]
        if stray:
            sr = sum(m["files"][f]["size"] for f in stray); sc = sum(m["files"][f]["container_bytes"] for f in stray)
            print("   stray (not in the index): %d files %.3f GB saving %.4f, all ok=%s" % (len(stray), sr / 1e9, 1 - sc / sr,
                  all(m["files"][f]["ok"] for f in stray)))
    med = float(np.median(bf_sav)) if bf_sav else float("nan")
    size_pass = bool(bf_sav) and med >= 0.319 and min(effs) >= 0.97
    print("\nSIZE: BF16 index sets n=%d median saving %.4f (bar >= 0.319), efficiency min %.4f (bar >= 0.97) -> %s" % (
        len(bf_sav), med, min(effs) if effs else float("nan"), "PASS" if size_pass else "CONCERN"))
    print("\n%-34s %-10s %8s %8s %6s %8s %8s %8s" % ("fine-tune", "base", "XD b/p", "SUB b/p", "ident", "H(x)sum", "H16(x)",
          "std b/p"))
    le8, fam = 0, []
    for name in sorted(M):
        d = M[name]["delta"]
        if "xor" not in d:
            continue
        iset = M[name]["index_set"]
        ts = [t for k, t in M[name]["tensors"].items() if k.split("::")[0] in iset]
        n = sum(t["n"] for t in ts)
        xd = sum(D["container_bytes"] for D in d["xor"].values())
        sb = sum(D["container_bytes"] for D in d["sub"].values())
        ntens = sum(len(D["tensors"]) for D in d["xor"].values())
        pb = d["per_tensor_bounds"]
        # identical = d all zero (a tensor under 4096 elements is stored RAW even then: rule A1's small-tensor rule)
        ident = sum(1 for v in pb.values() if "xor" in v and v["xor"]["same"] == v["xor"]["n"])
        hx = sum((v["xor"]["H_hi"] + v["xor"]["H_lo"]) * v["xor"]["n"] for v in pb.values() if "xor" in v)
        h16 = sum(v["xor"]["H16"] * v["xor"]["n"] for v in pb.values() if "xor" in v)
        nb = max(1, sum(v["xor"]["n"] for v in pb.values() if "xor" in v))
        std = sum(M[name]["files"][f]["container_bytes"] for f in iset)
        bxd = 8 * xd / n
        le8 += bxd <= 8.0
        fam.append((name, std, xd))
        print("%-34s %-10s %8.3f %8.3f %3d/%-3d %8.3f %8.3f %8.3f" % (name[:34], d["base"][6:16], bxd, 8 * sb / n, ident,
              ntens, hx / nb, h16 / nb, 8 * std / n))
    print("\nDELTA: fine-tunes with XD <= 8 b/p: %d of %d (family store KEEP iff >= 3) -> %s" % (
        le8, len(fam), "KEEP" if le8 >= 3 else "NOT-KEEP"))
    raw_all = sum(r for _, r, _ in rows)
    std_all = sum(c for _, _, c in rows)
    fam_all = sum(c for nme, _, c in rows if nme in REFS) + sum(min(s, x) for _, s, x in fam) + sum(
        c for nme, _, c in rows if nme not in REFS and nme not in [f[0] for f in fam])
    print("FAMILY (index sets, %d models): raw %.2f GB | standalone-coded %.2f GB (%.4f) | anchors coded + min(coded, XD) %.2f GB (%.4f)" % (
        len(rows), raw_all / 1e9, std_all / 1e9, std_all / raw_all, fam_all / 1e9, fam_all / raw_all))
    if bench_path:
        b = json.load(open(bench_path))
        print("\nSPEED (%s, %d threads max; GB/s = decoded output 1e9 B/s; P %.1f / floor %.1f / P12 %.1f / P5 %.0f):" % (
            b.get("cpu"), b.get("nproc", 0), P_EFF, P_FLOOR, P12, P5))
        for case, c in b["cases"].items():
            rws = {int(k): v for k, v in c["rows"].items()}
            meds = {T: v["gbps_median"] for T, v in rws.items()}
            d1 = meds.get(1)
            dall = max(meds.values())
            tbest = max(meds, key=meds.get)
            tp = min([T for T, v in meds.items() if v >= P_EFF], default=None)
            ver = all(v.get("verified_every_run", True) for v in rws.values())
            busy = [T for T, v in rws.items() if v.get("busy_others")]
            gb = c["out_bytes"] / 1e9
            line = "%-46s D1 %5.2f  Dall %6.2f @T=%-2d  T_P %-4s" % (case[:46], d1 or 0, dall, tbest, tp)
            if case != "memcpy":
                tdec, tdma = gb / dall, gb / P_EFF
                line += " band %-19s wake %.2f GB: dec %.3f s vs dma %.3f s -> %.3f s; @P5 dma %.3f s  verified=%s busy=%s" % (
                    band(dall, tp), gb, tdec, tdma, max(tdec, tdma) + 0.02, gb / P5, ver, busy)
            print(line)
    print("RESULT: summarize EXACT=%s SIZE=%s" % (exact, "PASS" if size_pass else "CONCERN"))
    return 0 if exact else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--roundtrip", metavar="MODELS_DIR")
    ap.add_argument("--bench", metavar="KEEP_DIR")
    ap.add_argument("--summarize", metavar="RESULT_JSON", help="print the report tables (reads only); add --bench-json")
    ap.add_argument("--bench-json")
    ap.add_argument("--out")
    ap.add_argument("--keep-dir")
    ap.add_argument("--keep-std", action="append", default=[], help="model name whose standalone containers are saved")
    ap.add_argument("--keep-xd", action="append", default=[], help="model name whose XOR-delta containers are saved")
    ap.add_argument("--base-dir")
    ap.add_argument("--ref", help="the --roundtrip RESULT.json (expected sha256 per file + delta base)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--threads-grid", default="1,2,4,8,12,16,24,32")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--roundtrip-v2", metavar="MODELS_DIR", help="v2 arms INC/D/E/DE (+ XD-*); WRITES --out")
    ap.add_argument("--evalset", help="eval-set manifest (sha256 per file + the v1 container bytes to replay)")
    ap.add_argument("--only", action="append", default=[], help="restrict --roundtrip-v2 to these model dirs")
    ap.add_argument("--keep-e", action="append", default=[], help="model whose E containers are saved to --keep-dir")
    ap.add_argument("--summarize-v2", metavar="RESULT_JSON", help="v2 pass bars (reads only); + --bench-json --fp8-json")
    ap.add_argument("--fp8-bound", metavar="MODEL_DIR", help="fp8 order-0 bound of one checkpoint; WRITES --out")
    ap.add_argument("--fp8-json")
    ap.add_argument("--xd-prefer-twin", action="store_true",
                    help="post-freeze challenger: also run XD-D/XD-DE with a twin-preferring dedup root (rows under xd_pt)")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.summarize:
        return summarize(a.summarize, a.bench_json)
    if a.summarize_v2:
        return summarize_v2(a.summarize_v2, a.bench_json, a.fp8_json)
    if a.fp8_bound and a.out:
        r = fp8_bound(a.fp8_bound)
        json.dump(r, open(a.out, "w"), indent=1)
        print(json.dumps(r))
        print("RESULT: fp8-bound r8_intended=%.4f" % r["r8_intended"])
        return 0
    try:
        build_lib()
    except (BuildError, OSError) as e:
        print("BUILD-FAILED", str(e)[-500:])
        return 3
    if a.roundtrip and a.out:
        if a.keep_dir:
            os.makedirs(a.keep_dir, exist_ok=True)
        return roundtrip(a.roundtrip, a.out, a.keep_dir, {"std": set(a.keep_std), "xd": set(a.keep_xd)}, a.threads)
    if a.roundtrip_v2 and a.out:
        if a.keep_dir:
            os.makedirs(a.keep_dir, exist_ok=True)
        return roundtrip_v2(a.roundtrip_v2, a.evalset, a.out, a.threads, only=set(a.only) or None, keep_dir=a.keep_dir,
                             keep_e=set(a.keep_e), xd_prefer_twin=a.xd_prefer_twin)
    if a.bench and a.runs < 1:
        print("USAGE: --runs must be >= 1 (the median needs at least one timed run)")
        return 2
    if a.bench and a.out and a.base_dir and a.ref:
        return bench(a.bench, a.out, a.base_dir, a.ref, [int(x) for x in a.threads_grid.split(",")], a.runs)
    ap.print_usage()
    print("USAGE: one of --self-test | --roundtrip DIR --out J | --bench KEEP --out J --base-dir DIR --ref J")
    return 2


if __name__ == "__main__":
    sys.exit(main())
