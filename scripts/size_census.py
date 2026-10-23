#!/usr/bin/env python3
"""size_census.py — +(b): read the PUBLIC Hugging Face config.json of every gpu-bunk tail model so its
weights and KV bytes per token stop being default-8B guesses then classify the tail by KV DEMAND in its busiest
held-out minute (FITS-ONE-CARD / NEEDS-BIG-CARD / NEEDS-REPLICAS / WEIGHTS-EXCEED-BIG-CARD). Descriptive census, $0, no
decision rule. Method (frozen before (b) was computed): (private report) §1.

Output contract:
  --self-test                     exit 0 + "SELF-TEST PASS n/n" (hand goldens: Qwen3-8B, Llama-3.1-8B, an MLA, a gemma3
                                  sliding-window config, a hybrid linear-attention config, classes); exit 1 + FAIL lines.
  --fetch --tail F --cache D      WRITES raw HTTP bodies + one status JSON per request into D (public HF only, no token,
                                  >= 1.05 s between ANY two requests, cached files are not re-fetched); one JSON summary on
                                  stdout; exit 0. A repo that cannot be read is RECORDED, never an error.
  --derive --tail F --cache D --out DIR
                                  WRITES DIR/census.json + DIR/census.csv (one row per tail model: MEASURED* / GUESSED and
                                  why, params, KV bytes/token); one JSON summary on stdout; exit 0.
  --demand --root R --cache NPZ --census DIR/census.json --out F
                                  read-only on the trace; imports the sim UNCHANGED (kv_conc_tok on a held-out-only
                                  trace); WRITES F (per-model busy-minute KV demand + class under every pre-registered
                                  reading); one JSON summary on stdout; exit 0.
  Bad input (missing file, unknown mode) = exit 2 + one "ERROR:" line on stderr.

Top pitfalls (observed, not imagined):
  1. **A UUID or hotkey name parses as a size** (: '98119c55-…' read as 9B, 'Affine-0001-…' as 1B). Sizes here come only
     from config.json; a name is a lookup key, never a size.
  2. **HF answers a DELETED repo and a GATED repo the same way on resolve/ (401)**. The api/models call separates
     them: gated = 200 with `gated` set; deleted/private = 401/404. Only gated repos try the mirror rule.
  3. **Stale shards inflate a weight count**. safetensors.total from the API is cross-checked against a config-derived
     count; a > 15 % disagreement is flagged, not silently trusted.
"""
import argparse, csv, json, math, os, re, sys, time, urllib.error, urllib.request

UA = "gpu-bunk-size-census/1 (research; public config reads only)"
HF = "https://huggingface.co"
MIN_GAP_S = 1.05
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
KV_BYTES = 2.0                 # bf16 KV (the sim's convention, bpt_of); int8 = 1.0 is a named sensitivity
FP8_GIB_PER_B = 1.07           # (the sim's GIB_PER_B["fp8"])
CARD_5090_GIB = 28.22          # G4 (T.CARD_GIB)
BIG_CARD_GIB = 88.0            # big card (ASSUMED, sensitivity 84.7-92)
OVH_GIB = 0.82                 # engine overhead, (T.OVERHEAD_GIB)
G30 = float(2 ** 30)
PACKED_QUANT = {"gptq", "awq", "bitsandbytes", "aqlm", "exl2", "hqq", "marlin"}


# ------------------------------------------------------------------------------------------------ fetch (public HTTP)
_last = [0.0]


def _get(url, tries=3):
    """-> (status, bytes). Global >= MIN_GAP_S between requests; 429/5xx back off and retry; never raises on HTTP."""
    for k in range(tries):
        gap = MIN_GAP_S - (time.time() - _last[0])
        if gap > 0:
            time.sleep(gap)
        _last[0] = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(30.0 * (k + 1))
                continue
            return e.code, b""
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(10.0 * (k + 1))
    return -1, b""


def _safe(repo):
    return re.sub(r"[^A-Za-z0-9._-]", "__", repo)


def _cached(cache, key, url):
    meta = os.path.join(cache, key + ".status.json")
    body = os.path.join(cache, key + ".body")
    if os.path.exists(meta):
        st = json.load(open(meta))["status"]
        return st, (open(body, "rb").read() if os.path.exists(body) else b"")
    st, b = _get(url)
    if st == -1 or st >= 500:          # QA r2: a transient failure is returned, never cached, so a re-run retries it
        return st, b
    if b:
        open(body, "wb").write(b)
    json.dump({"url": url, "status": st, "bytes": len(b), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(meta, "w"))
    return st, b


def _json(b):
    try:
        return json.loads(b.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def fetch_repo(repo, cache):
    """api/models then config.json at the API's sha. -> dict(repo, api_status, gated, sha, cfg_status)."""
    st, b = _cached(cache, _safe(repo) + ".api", f"{HF}/api/models/{repo}")
    out = {"repo": repo, "api_status": st, "gated": None, "sha": None, "cfg_status": None}
    if st != 200:
        return out
    api = _json(b) or {}
    out["gated"] = api.get("gated")
    out["sha"] = api.get("sha")
    rev = out["sha"] or "main"
    st2, _ = _cached(cache, _safe(repo) + ".config", f"{HF}/{repo}/resolve/{rev}/config.json")
    out["cfg_status"] = st2
    return out


def candidates(name):
    """Pre-registered resolution order (report §1): exact; '-TEE' stripped; for a gated official repo, unsloth/<base>."""
    c = [("MEASURED", name)]
    if name.endswith("-TEE"):
        c.append(("MEASURED-STRIPPED", name[:-4]))
    return c


def do_fetch(tail_path, cache):
    os.makedirs(cache, exist_ok=True)
    rows = json.load(open(tail_path))["rows"]
    rows = sorted(rows, key=lambda r: (-r["held_req"], r["cid"]))
    log = []
    for r in rows:
        nm = r["name"] or ""
        if UUID_RE.match(nm) or "/" not in nm:
            continue
        for tag, repo in candidates(nm):
            f = fetch_repo(repo, cache)
            f["tag"] = tag
            log.append(f)
            if f["api_status"] == 200 and f["cfg_status"] == 200:
                break
            if f["api_status"] == 200 and f["gated"] and f["cfg_status"] in (401, 403):
                m = fetch_repo("unsloth/" + repo.split("/", 1)[1], cache)
                m["tag"] = "MEASURED-MIRROR"
                m["mirror_of"] = repo
                log.append(m)
                break
    json.dump(log, open(os.path.join(cache, "fetch_log.json"), "w"), indent=0)
    c = {}
    for f in log:
        k = f"api{f['api_status']}/cfg{f['cfg_status']}"
        c[k] = c.get(k, 0) + 1
    return {"requests_log_rows": len(log), "by_status": c}


# ------------------------------------------------------------------------------------------------ config physics
def text_cfg(cfg):
    """The language-model config of a (possibly multimodal) config.json."""
    for k in ("text_config", "llm_config", "language_config", "thinker_config"):
        v = cfg.get(k)
        if isinstance(v, dict) and (v.get("num_hidden_layers") or v.get("text_config")):
            return text_cfg(v) if v.get("text_config") else v
    return cfg


def layer_kinds(tc):
    """-> (kinds list of 'full'|'sliding'|'linear'|'other', window or None, source)."""
    L = int(tc.get("num_hidden_layers") or tc.get("num_layers") or tc.get("n_layer") or 0)
    win = tc.get("sliding_window")
    lt = tc.get("layer_types")
    if isinstance(lt, list) and len(lt) == L and L:
        kinds = []
        for x in lt:
            x = str(x)
            if x in ("full_attention", "attention", "global_attention", "full"):
                kinds.append("full")
            elif "sliding" in x or x in ("local_attention", "chunked_attention"):
                kinds.append("sliding")
            elif "linear" in x or "mamba" in x or "conv" in x or "deltanet" in x or "ssm" in x:
                kinds.append("linear")
            else:
                kinds.append("other")
        return kinds, (int(win) if win else None), "layer_types"
    fi = tc.get("full_attention_interval")
    if fi:
        return ["full" if (i + 1) % int(fi) == 0 else "linear" for i in range(L)], None, "full_attention_interval"
    pat = tc.get("sliding_window_pattern")
    if pat and win:
        return ["full" if (i + 1) % int(pat) == 0 else "sliding" for i in range(L)], int(win), "sliding_window_pattern"
    if win and tc.get("use_sliding_window", True) and tc.get("model_type") not in ("qwen2", "qwen3", "qwen3_moe",
                                                                                     "qwen2_moe"):
        mpe = tc.get("max_position_embeddings")
        if not mpe or int(win) < int(mpe):
            return ["sliding"] * L, int(win), "sliding_window_all"
    return ["full"] * L, None, "all_full"


def kv_bytes(cfg, kv_bytes=KV_BYTES):
    """KV bytes per token. bpt_all counts every attention layer per token (an allocator without sliding-window support);
    bpt_global counts full-attention layers only; bpt_sw is the per-token cost of the sliding layers (held for <= window
    tokens per sequence). Linear-attention / mamba layers hold a constant state per sequence, not per token."""
    tc = text_cfg(cfg)
    kinds, win, src = layer_kinds(tc)
    nf = kinds.count("full")
    ns = kinds.count("sliding")
    out = {"layers": len(kinds), "full_layers": nf, "sliding_layers": ns, "linear_layers": kinds.count("linear"),
           "other_layers": kinds.count("other"), "window": win, "layer_src": src}
    if tc.get("kv_lora_rank"):
        per = (int(tc["kv_lora_rank"]) + int(tc.get("qk_rope_head_dim") or 0)) * kv_bytes
        idx = tc.get("index_head_dim")
        if idx:
            per += int(idx) * 1.0 + 4.0          # DSA indexer key: fp8 + one fp32 scale per token-layer (approximate)
        out.update(attn="MLA" + ("+DSA" if idx else ""), kv_lora_rank=int(tc["kv_lora_rank"]),
                   rope_dim=int(tc.get("qk_rope_head_dim") or 0), n_kv=None, head_dim=None)
    else:
        nh = int(tc.get("num_attention_heads") or tc.get("n_head") or 0)
        if nh <= 0:
            raise ValueError("config has no attention-head count")
        nkv = int(tc.get("num_key_value_heads") or tc.get("multi_query_group_num") or nh)
        hd = tc.get("head_dim")
        hd = int(hd) if hd else int(tc.get("hidden_size") or tc.get("n_embd")) // nh
        per = 2.0 * nkv * hd * kv_bytes
        out.update(attn="GQA", n_kv=nkv, head_dim=hd, n_heads=nh)
    out["bpt_all"] = per * (nf + ns + out["other_layers"])
    out["bpt_global"] = per * (nf + out["other_layers"])
    out["bpt_sw"] = per * ns
    return out


def bpt_effective(k, occ_tok, reading="SW_AWARE"):
    """Per-token bytes for a request holding occ_tok tokens: sliding layers cost bpt_sw * min(1, window / occ)."""
    if reading == "ALL_FULL" or not k["sliding_layers"]:
        return k["bpt_all"]
    w = k["window"] or 0
    return k["bpt_global"] + k["bpt_sw"] * (min(1.0, w / occ_tok) if occ_tok > 0 else 1.0)


def moe_info(tc):
    E = tc.get("n_routed_experts") or tc.get("num_experts") or tc.get("num_local_experts") or tc.get("moe_num_experts")
    k = tc.get("num_experts_per_tok") or tc.get("moe_topk") or tc.get("top_k")
    if not E or not k or int(E) <= 1:
        return None
    L = int(tc.get("num_hidden_layers") or 0)
    I = tc.get("moe_intermediate_size") or tc.get("expert_intermediate_size") or tc.get("intermediate_size")
    dense = int(tc.get("first_k_dense_replace") or 0)
    mlp_only = len(tc.get("mlp_only_layers") or [])
    step = int(tc.get("decoder_sparse_step") or 1)
    n_moe = (L - dense - mlp_only) // max(1, step) if step > 1 else L - dense - mlp_only
    return {"experts": int(E), "topk": int(k), "moe_inter": int(I), "moe_layers": max(0, int(n_moe))}


def active_params(total, tc):
    """MoE active = total - (E - k) * 3 * hidden * moe_inter * moe_layers (shared experts stay in both)."""
    mi = moe_info(tc)
    if not mi or not total:
        return total, mi
    H = int(tc.get("hidden_size") or 0)
    act = total - (mi["experts"] - mi["topk"]) * 3 * H * mi["moe_inter"] * mi["moe_layers"]
    return (act if act > 0 else None), mi


def est_params_dense(tc):
    """Config-derived total for a dense GQA decoder (cross-check of the API count); None if not that shape."""
    if moe_info(tc) or tc.get("kv_lora_rank") or layer_kinds(tc)[0].count("linear"):
        return None
    try:
        H = int(tc["hidden_size"]); L = int(tc["num_hidden_layers"]); V = int(tc["vocab_size"])
        I = int(tc["intermediate_size"]); nh = int(tc["num_attention_heads"])
    except (KeyError, TypeError, ValueError):
        return None
    nkv = int(tc.get("num_key_value_heads") or nh)
    hd = int(tc.get("head_dim") or H // nh)
    attn = H * nh * hd + 2 * H * nkv * hd + nh * hd * H
    if tc.get("attention_bias") or tc.get("model_type") in ("qwen2",):
        attn += nh * hd + 2 * nkv * hd
    qk_norm = 2 * hd if tc.get("model_type") in ("qwen3", "gemma3_text", "gemma3") else 0
    norms = 2 * H + (2 * H if tc.get("model_type") in ("gemma3_text", "gemma3", "gemma2") else 0)
    per = attn + 3 * H * I + norms + qk_norm
    emb = V * H * (1 if tc.get("tie_word_embeddings", False) else 2)
    return L * per + emb + H


def dtype_of(cfg):
    tc = text_cfg(cfg)
    return str(tc.get("torch_dtype") or tc.get("dtype") or cfg.get("torch_dtype") or cfg.get("dtype") or "")


def quant_of(cfg):
    q = cfg.get("quantization_config") or text_cfg(cfg).get("quantization_config") or {}
    if not isinstance(q, dict):
        return ""
    m = str(q.get("quant_method") or "")
    if m == "compressed-tensors":
        fmt = str(q.get("format") or "")
        bits = ""
        for g in (q.get("config_groups") or {}).values():
            w = (g or {}).get("weights") or {}
            bits = str(w.get("num_bits") or bits)
        return f"compressed-tensors:{fmt}:{bits}"
    return m


def is_packed(quant):
    return quant.split(":")[0] in PACKED_QUANT or ("pack" in quant) or quant.endswith(":4")


def weights_gib_fp8(params):
    return params / 1e9 * FP8_GIB_PER_B


def classify(w_gib, demand_bytes, card_gib=CARD_5090_GIB, big_gib=BIG_CARD_GIB, ovh=OVH_GIB):
    """Pre-registered classes (report §1.4). demand_bytes = busiest held-out minute's mean in-flight KV bytes."""
    if w_gib + ovh > big_gib:
        return "WEIGHTS-EXCEED-BIG-CARD"
    kv5 = (card_gib - ovh - w_gib) * G30
    kvb = (big_gib - ovh - w_gib) * G30
    if w_gib + ovh <= card_gib and demand_bytes <= kv5:
        return "FITS-ONE-CARD"
    if demand_bytes <= kvb:
        return "NEEDS-BIG-CARD"
    return "NEEDS-REPLICAS"


# ------------------------------------------------------------------------------------------------ derive
def _load_cached(cache, repo, kind):
    m = os.path.join(cache, _safe(repo) + f".{kind}.status.json")
    if not os.path.exists(m):
        return None, None
    st = json.load(open(m))["status"]
    b = os.path.join(cache, _safe(repo) + f".{kind}.body")
    return st, (_json(open(b, "rb").read()) if os.path.exists(b) else None)


def derive_row(r, cache, log_by_repo):
    nm = r["name"] or ""
    base = dict(cid=r["cid"], name=nm, held_req=r["held_req"], train_req=r["train_req"], guess_8b=r["guess"],
                sim_tb=r["tb"], sim_ab=r["ab"], sim_bpt=None)
    if UUID_RE.match(nm) or "/" not in nm:
        base.update(status="GUESSED", why="unmapped chute (no repo name in Chutes metadata)")
        return base
    chain = [("MEASURED", nm)] + ([("MEASURED-STRIPPED", nm[:-4])] if nm.endswith("-TEE") else [])
    why = "not fetched"
    for tag, repo in chain:
        sa, api = _load_cached(cache, repo, "api")
        if sa is None:
            continue
        if sa != 200:
            why = f"api {sa}: deleted or private"
            continue
        sc, cfg = _load_cached(cache, repo, "config")
        if sc in (401, 403) and (api or {}).get("gated"):
            mrepo = "unsloth/" + repo.split("/", 1)[1]
            sm, mapi = _load_cached(cache, mrepo, "api")
            smc, mcfg = _load_cached(cache, mrepo, "config")
            if smc == 200 and mcfg:
                return _measured(base, "MEASURED-MIRROR", mrepo, mapi, mcfg, mirror_of=repo)
            why = f"gated (config {sc}); mirror {mrepo} api {sm} config {smc}"
            continue
        if sc != 200 or not cfg:
            why = f"config.json {sc} (api 200)"
            continue
        try:
            return _measured(base, tag, repo, api, cfg)
        except (ValueError, TypeError, KeyError) as e:
            why = f"config unparseable: {e}"
            continue
    base.update(status="GUESSED", why=why)
    return base


def _measured(base, tag, repo, api, cfg, mirror_of=None):
    tc = text_cfg(cfg)
    k = kv_bytes(cfg)
    st = (api or {}).get("safetensors") or {}
    api_total = st.get("total")
    quant = quant_of(cfg)
    est = est_params_dense(tc)
    total, src = None, None
    if api_total and not is_packed(quant):
        total, src = int(api_total), "api-safetensors"
        if est and abs(api_total / est - 1.0) > 0.15:
            src = "api-safetensors(CONFLICT-config-est)"
    elif est:
        total, src = est, "config-estimate" + ("(packed-quant)" if is_packed(quant) else "")
    elif api_total:
        total, src = int(api_total), "api-safetensors(packed-quant-UNCORRECTED)"
    act, mi = active_params(total, tc)
    psum = sum(int(v) for v in (st.get("parameters") or {}).values()) or None
    # R2 (report §3.1): HF's own `total` can disagree with its per-dtype `parameters` dict; R2 takes the dict's sum
    tot2 = psum if (psum and api_total and psum != api_total and "CONFLICT" in (src or "")) else total
    act2, _ = active_params(tot2, tc)
    base.update(params_api_sum=psum, params_total_R2=tot2, params_active_R2=act2)
    base.update(status=tag, why="", repo=repo, mirror_of=mirror_of, sha=(api or {}).get("sha"),
                arch=(tc.get("architectures") or cfg.get("architectures") or [""])[0], model_type=tc.get("model_type"),
                dtype=dtype_of(cfg), quant=quant, params_total=total, params_src=src, params_est_config=est,
                params_active=act, moe=mi, **{f"kv_{a}": b for a, b in k.items()})
    if total:
        base["weights_gib_fp8"] = weights_gib_fp8(total)
    return base


def do_derive(tail_path, cache, outdir):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    rows = json.load(open(tail_path))["rows"]
    out = []
    for r in sorted(rows, key=lambda r: (-r["held_req"], r["cid"])):
        d = derive_row(r, cache, None)
        d["sim_bpt"] = _sim_bpt(r["tb"])
        out.append(d)
    os.makedirs(outdir, exist_ok=True)
    json.dump({"n": len(out), "rows": out}, open(os.path.join(outdir, "census.json"), "w"), indent=0, default=str)
    cols = ["cid", "name", "status", "why", "repo", "mirror_of", "sha", "model_type", "arch", "dtype", "quant",
            "params_total", "params_src", "params_est_config", "params_active", "weights_gib_fp8", "kv_attn",
            "kv_layers", "kv_full_layers", "kv_sliding_layers", "kv_linear_layers", "kv_window", "kv_n_kv",
            "kv_head_dim", "kv_kv_lora_rank", "kv_rope_dim", "kv_bpt_all", "kv_bpt_global", "kv_bpt_sw", "sim_tb",
            "sim_ab", "sim_bpt", "guess_8b", "held_req", "train_req"]
    with open(os.path.join(outdir, "census.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for d in out:
            w.writerow(d)
    tot = sum(d["held_req"] for d in out)
    meas = [d for d in out if d["status"].startswith("MEASURED")]
    c = {}
    for d in out:
        c[d["status"]] = c.get(d["status"], 0) + 1
    return {"n": len(out), "by_status": c, "measured": len(meas),
            "measured_held_req_share": sum(d["held_req"] for d in meas) / tot if tot else 0.0}


def _sim_bpt(tb):
    if tb <= 2.5:
        return 114688.0
    if tb <= 9.5:
        return 147456.0
    if tb <= 16.0:
        return 163840.0
    return 262144.0



# ------------------------------------------------------------------------------------------------ (b) demand
def busy_tokens(F, T, tr_h, models, phys, q):
    """Busiest (q=1.0) or q-quantile active minute of D(t)/60 per model, in tokens, with KV_UTIL undone.
    tr_h holds held-out cells only; held is set past the end so every cell counts (kv_conc_tok reads minute < held)."""
    M = len(models)
    return F.kv_conc_tok(tr_h, list(range(M)), M, models, phys, q=q, held=10 ** 12) * F.KV_UTIL


def _heldout_trace(T, tr, models, held):
    k = tr.minute >= held
    h = T.Trace.__new__(T.Trace)
    h.minute, h.mid, h.n, h.it, h.ot, h.explicit = tr.minute[k], tr.mid[k], tr.n[k], tr.it[k], tr.ot[k], None
    h.models = models
    return h


def _with_models(T, tr, models):
    h = T.Trace.__new__(T.Trace)
    h.minute, h.mid, h.n, h.it, h.ot, h.explicit = tr.minute, tr.mid, tr.n, tr.it, tr.ot, None
    h.models = models
    return h


def do_demand(root, npz, census_path, out_path, big_gibs=(84.7, 92.0), params="R1"):
    import numpy as np
    root, npz, census_path, out_path = (os.path.abspath(x) for x in (root, npz, census_path, out_path))  # before chdir
    sys.path.insert(0, os.path.join(root, "scripts"))
    os.chdir(root)
    import tailpool_sim as T, tailpool_v2 as V, fabric_sim as F  # noqa: E401 (unchanged, read-only)
    cen = {r["cid"]: r for r in json.load(open(census_path))["rows"]}
    tr, _, _ = F.build_pop("P8", root, npz)
    HELD = T.HELDOUT_MIN
    M = len(tr.models)
    phys = F.phys_with({"tpot_sat": 0.06})
    kap = phys["kappa"]
    k = tr.minute >= HELD
    sn = np.bincount(tr.mid[k], weights=tr.n[k].astype(float), minlength=M)
    si = np.bincount(tr.mid[k], weights=tr.it[k], minlength=M)
    so = np.bincount(tr.mid[k], weights=tr.ot[k], minlength=M)
    L_occ = np.where(sn > 0, (si + kap * so) / np.maximum(sn, 1), 0.0)
    cell_len = (tr.it[k] + tr.ot[k]) / tr.n[k]
    max_len = np.zeros(M)
    np.maximum.at(max_len, tr.mid[k], cell_len)
    # measured sizes (MEASURED*) or the sim's own (GUESSED)
    meas_models, info = [], []
    for m, mod in enumerate(tr.models):
        cid, tb, ab, prec, w, f = mod
        c = cen[cid]
        pt = c.get("params_total_R2") if params == "R2" else c.get("params_total")
        pa = c.get("params_active_R2") if params == "R2" else c.get("params_active")
        if c["status"].startswith("MEASURED") and pt:
            tbm = pt / 1e9
            abm = (pa or pt) / 1e9
            kd = {a[3:]: c[a] for a in c if a.startswith("kv_")}
            info.append(dict(measured=True, tb=tbm, ab=abm, w=weights_gib_fp8(pt), k=kd))
            meas_models.append((cid, tbm, abm, prec, weights_gib_fp8(pt), f))
        else:
            b = F.bpt_of(tb)
            kd = dict(bpt_all=b, bpt_global=b, bpt_sw=0.0, sliding_layers=0, window=None)
            info.append(dict(measured=False, tb=tb, ab=ab, w=w, k=kd))
            meas_models.append(mod)
    tr_h = _heldout_trace(T, tr, meas_models, HELD)
    dem = {"max": busy_tokens(F, T, tr_h, meas_models, phys, 1.0),
           "p99min": busy_tokens(F, T, tr_h, meas_models, phys, 0.99),
           "tpotq": busy_tokens(F, T, tr_h, meas_models, F.phys_with({"tpot_sat": 0.06, "tpot_q": phys["tpot_q"] / 2.0}),
                                1.0)}
    readings = {"PRIMARY": ("max", "SW_AWARE", 1.0, BIG_CARD_GIB), "ALL_FULL": ("max", "ALL_FULL", 1.0, BIG_CARD_GIB),
                "INT8_KV": ("max", "SW_AWARE", 0.5, BIG_CARD_GIB), "TPOT_Q": ("tpotq", "SW_AWARE", 1.0, BIG_CARD_GIB),
                "P99_MINUTE": ("p99min", "SW_AWARE", 1.0, BIG_CARD_GIB)}
    for g in big_gibs:
        readings[f"BIG_{g:g}"] = ("max", "SW_AWARE", 1.0, g)
    rows = []
    for m in range(M):
        I = info[m]
        c = cen[tr.models[m][0]]
        r = dict(cid=c["cid"], name=c["name"], status=c["status"], held_req=c["held_req"], measured=I["measured"],
                 tb=I["tb"], ab=I["ab"], w_gib=I["w"], L_occ=float(L_occ[m]), max_cell_len=float(max_len[m]),
                 bpt_eff=bpt_effective(I["k"], float(L_occ[m]), "SW_AWARE"),
                 bpt_all=I["k"]["bpt_all"], sim_bpt=F.bpt_of(tr.models[m][1]), sim_w=tr.models[m][4])
        for nm, (dk, rd, scale, big) in readings.items():
            tok = float(dem[dk][m])
            byt = tok * bpt_effective(I["k"], float(L_occ[m]), rd) * scale
            r[f"demand_tok_{nm}"] = tok
            r[f"demand_gib_{nm}"] = byt / G30
            r[f"class_{nm}"] = classify(I["w"], byt, big_gib=big)
        kv5 = (CARD_5090_GIB - OVH_GIB - I["w"]) * G30
        kvb = (BIG_CARD_GIB - OVH_GIB - I["w"]) * G30
        byt = r["demand_gib_PRIMARY"] * G30
        r["replicas_5090"] = math.ceil(byt / kv5) if kv5 > 0 else None
        r["replicas_big"] = math.ceil(byt / kvb) if kvb > 0 else None
        r["single_req_over_5090_kv"] = bool(kv5 <= 0 or max_len[m] * r["bpt_eff"] > kv5)
        rows.append(r)
    # ---- comparison: dedicated reference, unchanged functions
    def ded_scores(trx, kt, keep):
        d0 = V.simulate_dedicated_v2(trx, None)
        dk = F.dedicated(trx, kt, phys)
        res, p95 = {}, {}
        for slo in (10.0, 2.0):
            a = T.score([d0[i] for i in keep], [d0[i] for i in keep], slo)
            b = T.score([dk[i] for i in keep], [dk[i] for i in keep], slo)
            res[f"S{int(slo)}"] = {"eligible": a["eligible"], "feasible_compute_only": a["slo_feasible_models"],
                                   "feasible_honest": b["slo_feasible_models"]}
        for i in keep:
            nd = sum(d0[i].values())
            p95[i] = (nd, T._pct(d0[i], 0.95), T._pct(dk[i], 0.95))
        return res, p95
    kt0 = [(T.CARD_GIB - mm[4] - F.OVH) * G30 / F.bpt_of(mm[1]) for mm in tr.models]
    ref, p95_0 = ded_scores(tr, kt0, list(range(M)))
    no5090 = [m for m in range(M) if info[m]["w"] + OVH_GIB > CARD_5090_GIB]
    keep = [m for m in range(M) if m not in set(no5090)]
    ktm = [((CARD_5090_GIB - info[m]["w"] - OVH_GIB) * G30 / rows[m]["bpt_eff"]) if m in set(keep) else 1e12
           for m in range(M)]
    trm = _with_models(T, tr, meas_models)
    mes, p95_m = ded_scores(trm, ktm, keep)
    fail31 = [m for m in range(M) if p95_0[m][0] >= 20 and p95_0[m][2] is not None and p95_0[m][2] > 10.0]
    per = []
    for m in fail31:
        r = rows[m]
        e = dict(cid=r["cid"][:8], name=r["name"][:60], status=r["status"], held_req=r["held_req"], w_gib=r["w_gib"],
                 sim_w=r["sim_w"], bpt_eff=r["bpt_eff"], sim_bpt=r["sim_bpt"], p95_guess=p95_0[m][2],
                 p95_compute_only_guess=p95_0[m][1], class_PRIMARY=r["class_PRIMARY"])
        if m in set(no5090):
            e["measured_run"] = "NO-5090"
        else:
            e["measured_run"] = ("FEASIBLE" if (p95_m[m][2] is not None and p95_m[m][2] <= 10.0) else "STILL-FAILS")
            e["p95_measured"] = p95_m[m][2]
        per.append(e)
    rev = [rows[m]["cid"][:8] for m in keep if m not in set(fail31) and p95_0[m][0] >= 20
           and p95_m[m][0] >= 20 and p95_m[m][2] is not None and p95_m[m][2] > 10.0]
    out = {"n": M, "params_reading": params, "readings": {k: list(v) for k, v in readings.items()}, "rows": rows,
           "dedicated_reference_unchanged": ref, "dedicated_measured": mes, "no5090": [rows[m]["cid"][:8] for m in no5090],
           "fail31": per, "newly_failing_with_measured_sizes": rev}
    json.dump(out, open(out_path, "w"), indent=0, default=str)
    summ = {"reference": ref, "measured": mes, "no5090": len(no5090),
            "fail31_breakdown": {s: sum(1 for e in per if e["measured_run"] == s)
                                 for s in ("NO-5090", "FEASIBLE", "STILL-FAILS")},
            "newly_failing": len(rev)}
    return summ


# ------------------------------------------------------------------------------------------------ self-test
QWEN3_8B = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36,
            "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 12288,
            "vocab_size": 151936, "tie_word_embeddings": False, "sliding_window": None, "use_sliding_window": False,
            "max_window_layers": 36, "attention_bias": False, "torch_dtype": "bfloat16"}
LLAMA31_8B = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32,
              "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256,
              "tie_word_embeddings": False, "torch_dtype": "bfloat16", "max_position_embeddings": 131072}
MLA_LITE = {"model_type": "deepseek_v2", "hidden_size": 2048, "num_hidden_layers": 27, "num_attention_heads": 16,
            "kv_lora_rank": 512, "qk_rope_head_dim": 64, "qk_nope_head_dim": 128, "v_head_dim": 128}
GEMMA3_4B = {"architectures": ["Gemma3ForConditionalGeneration"], "model_type": "gemma3",
             "text_config": {"model_type": "gemma3_text", "hidden_size": 2560, "num_hidden_layers": 34,
                             "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256,
                             "sliding_window": 1024, "sliding_window_pattern": 6, "intermediate_size": 10240}}
HYBRID = {"model_type": "qwen3_next", "hidden_size": 2048, "num_hidden_layers": 48, "num_attention_heads": 16,
          "num_key_value_heads": 2, "head_dim": 256, "full_attention_interval": 4}
Q3_30B_A3B = {"model_type": "qwen3_moe", "hidden_size": 2048, "num_hidden_layers": 48, "num_experts": 128,
              "num_experts_per_tok": 8, "moe_intermediate_size": 768, "decoder_sparse_step": 1, "mlp_only_layers": []}


def self_test():
    fails, n = [], [0]

    def chk(name, got, want, tol=1e-9):
        n[0] += 1
        ok = (got == want) if not isinstance(want, float) else (got is not None and abs(got - want) <= tol * max(1.0, abs(want)))
        if not ok:
            fails.append(f"FAIL {name}: got {got!r} want {want!r}")

    # G1/G2: hand-computed (2 x layers x kv_heads x head_dim x 2 bytes; params by layer, see report §2)
    chk("G1 Qwen3-8B KV B/token", kv_bytes(QWEN3_8B)["bpt_all"], 147456.0)
    chk("G1 Qwen3-8B params (config)", est_params_dense(QWEN3_8B), 8190735360)
    chk("G2 Llama-3.1-8B KV B/token", kv_bytes(LLAMA31_8B)["bpt_all"], 131072.0)
    chk("G2 Llama-3.1-8B params (config)", est_params_dense(LLAMA31_8B), 8030261248)
    chk("G2 head_dim from hidden/heads", kv_bytes(LLAMA31_8B)["head_dim"], 128)
    # G3: MLA latent cache = layers x (kv_lora_rank + rope) x 2; DSA indexer adds index_head_dim x 1 + 4
    chk("G3 MLA B/token", kv_bytes(MLA_LITE)["bpt_all"], 31104.0)
    chk("G3 MLA+DSA B/token", kv_bytes(dict(MLA_LITE, index_head_dim=128))["bpt_all"], 34668.0)
    chk("G3 MLA attn tag", kv_bytes(MLA_LITE)["attn"], "MLA")
    # G4: gemma3 5:1 sliding (window 1024): globals at i = 5, 11, 17, 23, 29
    g = kv_bytes(GEMMA3_4B)
    chk("G4 gemma3 full layers", g["full_layers"], 5)
    chk("G4 gemma3 bpt_all", g["bpt_all"], 139264.0)
    chk("G4 gemma3 bpt_global", g["bpt_global"], 20480.0)
    chk("G4 gemma3 effective @4096 occ", bpt_effective(g, 4096.0), 50176.0)
    chk("G4 gemma3 effective @512 occ", bpt_effective(g, 512.0), 139264.0)
    chk("G4 gemma3 ALL_FULL @4096", bpt_effective(g, 4096.0, "ALL_FULL"), 139264.0)
    lt = dict(GEMMA3_4B["text_config"], layer_types=["full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
                                                     for i in range(34)])
    chk("G4b layer_types path agrees", kv_bytes(lt)["bpt_global"], 20480.0)
    # G5: hybrid linear attention: 12 full of 48, 2 kv heads x 256 -> 12 x 2048
    chk("G5 hybrid bpt (linear layers excluded)", kv_bytes(HYBRID)["bpt_all"], 24576.0)
    # G6: Qwen3-30B-A3B active params
    act, mi = active_params(30532122624, Q3_30B_A3B)
    chk("G6 MoE active params", act, 3353032704)
    chk("G6 dense model has no MoE", active_params(8190735360, QWEN3_8B)[0], 8190735360)
    # G7: classes at the frozen thresholds
    w8 = weights_gib_fp8(8e9)
    edge = (CARD_5090_GIB - OVH_GIB - w8) * G30
    chk("G7 fits", classify(w8, 0.0), "FITS-ONE-CARD")
    chk("G7 fits at the edge", classify(w8, edge), "FITS-ONE-CARD")
    chk("G7 just over -> big", classify(w8, edge + 1.0), "NEEDS-BIG-CARD")
    chk("G7 31B weights -> big", classify(weights_gib_fp8(31.0e9), 0.0), "NEEDS-BIG-CARD")
    chk("G7 over big -> replicas", classify(w8, (BIG_CARD_GIB - OVH_GIB - w8) * G30 + 1.0), "NEEDS-REPLICAS")
    chk("G7 355B -> exceeds", classify(weights_gib_fp8(355e9), 0.0), "WEIGHTS-EXCEED-BIG-CARD")
    # G8: a UUID name is never measured; a -TEE name tries the stripped repo second
    r = derive_row(dict(cid="x", name="98119c55-b8d6-5be9-9b4a-d612834167eb", held_req=1, train_req=1, guess=False,
                        tb=9.0, ab=9.0), "/nonexistent", None)
    chk("G8 UUID -> GUESSED", r["status"], "GUESSED")
    chk("G8 UUID reason = unmapped (never looked up)", r["why"].startswith("unmapped"), True)
    chk("G8 -TEE candidates", [c[1] for c in candidates("org/Model-TEE")], ["org/Model-TEE", "org/Model"])
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for kind, body in (("api", {"sha": "x", "gated": False}), ("config", {"num_hidden_layers": 2, "hidden_size": 8})):
            open(os.path.join(td, f"org__bad.{kind}.body"), "w").write(json.dumps(body))
            json.dump({"status": 200}, open(os.path.join(td, f"org__bad.{kind}.status.json"), "w"))
        rb = derive_row(dict(cid="y", name="org/bad", held_req=1, train_req=1, guess=True, tb=8.0, ab=8.0), td, None)
    chk("G8b headless config -> GUESSED (unparseable), no crash", (rb["status"], rb["why"].startswith("config unparseable")),
        ("GUESSED", True))
    # G9: the busy-minute demand wrapper on a hand case (one model; minute 0: n=60, in 100, out 1000 per request)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    import tailpool_sim as T, fabric_sim as F  # noqa: E401
    from real_trace_analysis import C_REF
    tr = T.Trace.__new__(T.Trace)
    tr.minute = np.array([0, 1]); tr.mid = np.array([0, 0]); tr.n = np.array([60, 6])
    tr.it = np.array([6000.0, 600.0]); tr.ot = np.array([60000.0, 6000.0]); tr.explicit = None
    mods = [("m", 3.1, 3.1, "fp8", 3.317, 5.0)]
    tr.models = mods
    phys = F.phys_with({"tpot_sat": 0.06})
    want = 60 * (100 + 0.5 * 1000) * (100 / C_REF + 1000 * 0.013) / 60.0
    chk("G9 busiest-minute tokens", float(busy_tokens(F, T, tr, mods, phys, 1.0)[0]), want, 1e-9)
    chk("G9 sim bpt mapping = F.bpt_of", [_sim_bpt(x) for x in (1.0, 4.0, 14.0, 24.0)],
        [F.bpt_of(x) for x in (1.0, 4.0, 14.0, 24.0)])
    for f in fails:
        print(f)
    if fails:
        print(f"SELF-TEST FAIL {n[0] - len(fails)}/{n[0]}")
        return 1
    print(f"SELF-TEST PASS {n[0]}/{n[0]}")
    return 0


# ------------------------------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--derive", action="store_true")
    ap.add_argument("--demand", action="store_true")
    ap.add_argument("--tail")
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--root")
    ap.add_argument("--census")
    ap.add_argument("--params", choices=["R1", "R2"], default="R1",
                    help="R1 = HF safetensors.total (frozen literal); R2 = sum of its per-dtype parameters on CONFLICT rows")
    a = ap.parse_args()
    try:
        if a.self_test:
            return self_test()
        if a.fetch:
            for p in (a.tail,):
                if not p or not os.path.exists(p):
                    raise FileNotFoundError(f"--tail {p}")
            if not a.cache:
                raise FileNotFoundError("--fetch needs --cache DIR")
            print(json.dumps(do_fetch(a.tail, a.cache)))
            return 0
        if a.derive:
            if not a.tail or not os.path.exists(a.tail) or not a.cache or not os.path.isdir(a.cache) or not a.out:
                raise FileNotFoundError("--derive needs --tail FILE --cache DIR --out DIR")
            print(json.dumps(do_derive(a.tail, a.cache, a.out)))
            return 0
        if a.demand:
            for p in (a.root, a.cache, a.census):
                if not p or not os.path.exists(p):
                    raise FileNotFoundError(f"--demand input {p}")
            if not a.out:
                raise FileNotFoundError("--demand needs --out FILE")
            print(json.dumps(do_demand(a.root, a.cache, a.census, a.out, params=a.params)))
            return 0
        raise ValueError("no mode (one of --self-test --fetch --derive --demand)")
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
