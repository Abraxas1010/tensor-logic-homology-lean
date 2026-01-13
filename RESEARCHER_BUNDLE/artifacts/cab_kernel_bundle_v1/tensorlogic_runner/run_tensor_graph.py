#!/usr/bin/env python3
"""
Reference runner for HeytingLean TensorLogic tensor graph IR (PyTorch backend).

This is a *debug/reference* implementation intended to:
  - load the exported JSON IR (`tensor_logic_export_graph`),
  - execute the specified fixpoint semantics, and
  - emit derived facts in a JSON format similar to `tensor_logic_cli`.

It is not optimized; it uses dense tensors and is suitable for small domains/examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _require_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for this runner (pip install torch). "
            f"Import failed: {exc}"
        ) from exc


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _require_safetensors():
    try:
        from safetensors.torch import save_file  # type: ignore
        from safetensors import safe_open  # type: ignore

        return save_file, safe_open
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "safetensors is required for --checkpoint-* (pip install safetensors). "
            f"Import failed: {exc}"
        ) from exc


def _sym_kind(sym: Dict[str, Any]) -> Tuple[str, str]:
    if "var" in sym:
        return ("var", str(sym["var"]))
    if "const" in sym:
        return ("const", str(sym["const"]))
    raise ValueError(f"bad sym: {sym}")


@dataclass(frozen=True)
class Semantics:
    mode: str
    tnorm: str
    and_kind: str
    or_kind: str
    sharpness: float
    max_iter: int
    eps: float


def _parse_semantics(g: Dict[str, Any]) -> Semantics:
    s = g["semantics"]
    fp = s["fixpoint"]
    return Semantics(
        mode=str(s["mode"]),
        tnorm=str(s["tnorm"]),
        and_kind=str(s["and_kind"]),
        or_kind=str(s["or_kind"]),
        sharpness=float(s.get("sharpness", 1.0)),
        max_iter=int(fp["max_iter"]),
        eps=float(fp["eps"]),
    )


def _and_op(torch, kind: str, a, b):
    if kind == "mul":
        return a * b
    if kind == "min":
        return torch.minimum(a, b)
    if kind == "luk_and":
        return torch.clamp(a + b - 1.0, min=0.0)
    if kind == "bool_and":
        return ((a != 0.0) & (b != 0.0)).to(dtype=a.dtype)
    raise ValueError(f"unknown and_kind: {kind}")


def _or_op(torch, kind: str, a, b):
    if kind == "noisy_or":
        return torch.clamp(a + b - a * b, min=0.0, max=1.0)
    if kind == "max":
        return torch.maximum(a, b)
    if kind == "luk_or":
        return torch.clamp(a + b, max=1.0)
    if kind == "bool_or":
        return ((a != 0.0) | (b != 0.0)).to(dtype=a.dtype)
    if kind == "xor":
        ai = (a != 0.0).to(dtype=torch.int64)
        bi = (b != 0.0).to(dtype=torch.int64)
        return ((ai + bi) & 1).to(dtype=a.dtype)
    raise ValueError(f"unknown or_kind: {kind}")


def _or_reduce(torch, kind: str, x, dim: int):
    if kind == "noisy_or":
        return 1.0 - torch.prod(1.0 - x, dim=dim)
    if kind == "luk_or":
        return torch.clamp(torch.sum(x, dim=dim), max=1.0)
    if kind in ("max", "bool_or"):
        return torch.amax(x, dim=dim)
    if kind == "xor":
        s = torch.sum((x != 0.0).to(dtype=torch.int64), dim=dim)
        return (s & 1).to(dtype=x.dtype)
    raise ValueError(f"unknown or_kind: {kind}")


def _collapse_equal_axes(torch, t, vars_by_axis: List[str]):
    """
    Enforce equality constraints for repeated variables within a single atom by extracting diagonals.

    `torch.diagonal` removes two axes and appends one diagonal axis at the end; we track this by
    removing the two var entries and appending the var name once.
    """
    while True:
        dup: Optional[str] = None
        seen = set()
        for v in vars_by_axis:
            if v in seen:
                dup = v
                break
            seen.add(v)
        if dup is None:
            return t, vars_by_axis
        idxs = [i for i, v in enumerate(vars_by_axis) if v == dup]
        i, j = idxs[0], idxs[1]
        t = torch.diagonal(t, dim1=i, dim2=j)
        # Remove higher index first.
        vars_by_axis.pop(j)
        vars_by_axis.pop(i)
        vars_by_axis.append(dup)


def _lift_atom_tensor(
    torch,
    pred_tensor,
    atom_args: Sequence[Dict[str, Any]],
    rule_vars: Sequence[str],
    domain_idx: Dict[str, int],
    n: int,
):
    # Index constants, keep variable axes.
    index = []
    occ_vars: List[str] = []
    for sym in atom_args:
        k, name = _sym_kind(sym)
        if k == "const":
            if name not in domain_idx:
                raise KeyError(f"constant '{name}' missing from domain")
            index.append(domain_idx[name])
        else:
            index.append(slice(None))
            occ_vars.append(name)

    t = pred_tensor[tuple(index)]
    t, occ_vars = _collapse_equal_axes(torch, t, occ_vars)

    present = [v for v in rule_vars if v in set(occ_vars)]
    if len(present) != len(occ_vars):
        raise ValueError(f"internal: occ_vars={occ_vars} present={present} rule_vars={rule_vars}")

    if present:
        perm = [occ_vars.index(v) for v in present]
        if perm != list(range(len(perm))):
            t = t.permute(*perm)

    full_shape = [n if v in set(present) else 1 for v in rule_vars]
    return t.reshape(full_shape)


def _head_tensor_from_keep(
    torch,
    keep_tensor,
    keep_vars: Sequence[str],
    head_args: Sequence[Dict[str, Any]],
    domain_idx: Dict[str, int],
    n: int,
    device,
    dtype,
):
    arity = len(head_args)

    # Unique vars in order of first appearance in the head.
    head_first_vars: List[str] = []
    head_positions: Dict[str, List[int]] = {}
    const_positions: List[Tuple[int, str]] = []
    for pos, sym in enumerate(head_args):
        k, name = _sym_kind(sym)
        if k == "var":
            head_positions.setdefault(name, []).append(pos)
            if name not in head_first_vars:
                head_first_vars.append(name)
        else:
            const_positions.append((pos, name))

    if head_first_vars:
        perm = [keep_vars.index(v) for v in head_first_vars]
        t = keep_tensor
        if perm != list(range(len(perm))):
            t = t.permute(*perm)
    else:
        t = keep_tensor

    first_pos = {v: ps[0] for v, ps in head_positions.items()}
    shape = []
    for pos, sym in enumerate(head_args):
        k, name = _sym_kind(sym)
        if k == "var" and first_pos[name] == pos:
            shape.append(n)
        else:
            shape.append(1)

    out = t.reshape(shape)

    if any(len(ps) > 1 for ps in head_positions.values()):
        eye = torch.eye(n, device=device, dtype=dtype)
        for v, ps in head_positions.items():
            p0 = ps[0]
            for p in ps[1:]:
                mask_shape = [1] * arity
                mask_shape[p0] = n
                mask_shape[p] = n
                out = out * eye.reshape(mask_shape)

    for pos, c in const_positions:
        if c not in domain_idx:
            raise KeyError(f"constant '{c}' missing from domain")
        one_hot = torch.zeros(n, device=device, dtype=dtype)
        one_hot[domain_idx[c]] = 1.0
        mask_shape = [1] * arity
        mask_shape[pos] = n
        out = out * one_hot.reshape(mask_shape)

    return out


def run_fixpoint(
    graph: Dict[str, Any],
    *,
    device: str = "cpu",
    dtype: str = "float32",
    override_max_iter: Optional[int] = None,
    override_eps: Optional[float] = None,
    rule_weights: Optional[Sequence[Any]] = None,
    fact_weights: Optional[Sequence[Any]] = None,
    stop_on_converge: bool = True,
) -> Dict[str, Any]:
    torch = _require_torch()

    sem = _parse_semantics(graph)
    max_iter = int(override_max_iter) if override_max_iter is not None else sem.max_iter
    eps = float(override_eps) if override_eps is not None else sem.eps

    if graph["semantics"]["fixpoint"]["kind"] != "anchored_step_from_base":
        raise ValueError(f"unsupported fixpoint kind: {graph['semantics']['fixpoint']}")

    n = int(graph["domain"]["size"])
    domain = list(graph["domain"]["symbols"])
    domain_idx = {s: i for i, s in enumerate(domain)}

    torch_dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
    dev = torch.device(device)

    pred_meta: Dict[str, int] = {p["name"]: int(p["arity"]) for p in graph["predicates"]}
    base: Dict[str, Any] = {}
    for pred, arity in pred_meta.items():
        shape = (n,) * arity
        base[pred] = torch.zeros(shape, device=dev, dtype=torch_dtype)

    if fact_weights is None:
        # Fast path: constants only (no gradients w.r.t. fact weights).
        for f in graph["facts"]:
            pred = str(f["pred"])
            arg_ids = list(map(int, f["arg_ids"]))
            if pred not in base:
                raise KeyError(f"fact references unknown predicate: {pred}")
            if len(arg_ids) != pred_meta[pred]:
                raise ValueError(f"arity mismatch in fact for {pred}: {arg_ids}")
            idx = tuple(arg_ids)
            cur = base[pred][idx]
            w = torch.tensor(float(f["weight"]), device=dev, dtype=torch_dtype)
            base[pred][idx] = _or_op(torch, sem.or_kind, cur, w)
    else:
        # Differentiable path: build predicate base tensors via a "one row per fact" scatter,
        # then reduce across facts using the mode's OR semantics.
        facts_by_pred: Dict[str, List[Tuple[int, Any]]] = {}
        for fi, f in enumerate(graph["facts"]):
            if fi >= len(fact_weights):
                raise ValueError(f"fact_weights too short (need {len(graph['facts'])})")
            pred = str(f["pred"])
            arg_ids = list(map(int, f["arg_ids"]))
            if pred not in base:
                raise KeyError(f"fact references unknown predicate: {pred}")
            if len(arg_ids) != pred_meta[pred]:
                raise ValueError(f"arity mismatch in fact for {pred}: {arg_ids}")
            flat = 0
            for a in arg_ids:
                flat = flat * n + int(a)
            facts_by_pred.setdefault(pred, []).append((flat, fact_weights[fi]))

        for pred, entries in facts_by_pred.items():
            arity = int(pred_meta[pred])
            nflat = (n**arity) if arity > 0 else 1
            idxs = torch.tensor([flat for flat, _w in entries], device=dev, dtype=torch.int64).reshape(-1, 1)
            src = torch.stack([w for _flat, w in entries]).reshape(-1, 1).to(dtype=torch_dtype)
            vals = torch.zeros((len(entries), nflat), device=dev, dtype=torch_dtype)
            vals = vals.scatter(1, idxs, src)
            base_flat = _or_reduce(torch, sem.or_kind, vals, dim=0)
            base[pred] = base_flat.reshape((n,) * arity)

    cur_state = {k: v.clone() for k, v in base.items()}

    iters = 0
    last_delta = 0.0
    converged = False

    for it in range(max_iter):
        iters = it + 1
        next_state = {k: v.clone() for k, v in base.items()}

        for ri, r in enumerate(graph["rules"]):
            head = r["head"]
            head_pred = str(head["pred"])
            head_args = list(head["args"])
            rule_vars = list(map(str, r["vars"]))
            elim_vars = list(map(str, r["elim_vars"]))
            keep_vars = [v for v in rule_vars if v not in set(elim_vars)]

            if head_pred not in next_state:
                raise KeyError(f"rule head references unknown predicate: {head_pred}")

            if rule_vars:
                body_shape = (n,) * len(rule_vars)
                body_val = torch.ones(body_shape, device=dev, dtype=torch_dtype)
            else:
                body_val = torch.ones((), device=dev, dtype=torch_dtype)

            for lit in r["body"]:
                lit_pred = str(lit["pred"])
                lit_args = list(lit["args"])
                lit_tensor = _lift_atom_tensor(
                    torch, cur_state[lit_pred], lit_args, rule_vars, domain_idx, n
                )
                body_val = _and_op(torch, sem.and_kind, body_val, lit_tensor)

            if rule_weights is None:
                rule_w = torch.tensor(float(r.get("weight", 1.0)), device=dev, dtype=torch_dtype)
            else:
                if ri >= len(rule_weights):
                    raise ValueError(f"rule_weights too short (need {len(graph['rules'])})")
                rule_w = rule_weights[ri]
            body_val = _and_op(torch, sem.and_kind, body_val, rule_w)

            if elim_vars:
                pos = {v: i for i, v in enumerate(rule_vars)}
                perm = [pos[v] for v in keep_vars + elim_vars]
                if perm != list(range(len(perm))):
                    body_val = body_val.permute(*perm)
                keep_shape = (n,) * len(keep_vars)
                body_val = body_val.reshape(keep_shape + (-1,))
                keep_val = _or_reduce(torch, sem.or_kind, body_val, dim=-1)
            else:
                keep_val = body_val

            head_contrib = _head_tensor_from_keep(
                torch,
                keep_val,
                keep_vars,
                head_args,
                domain_idx,
                n,
                device=dev,
                dtype=torch_dtype,
            )

            next_state[head_pred] = _or_op(torch, sem.or_kind, next_state[head_pred], head_contrib)

        # delta across all predicate tensors
        delta = 0.0
        for pred in next_state.keys():
            d = torch.max(torch.abs(next_state[pred] - cur_state[pred])).item()
            delta = max(delta, float(d))
        last_delta = delta
        cur_state = next_state

        if stop_on_converge and delta <= eps:
            converged = True
            break

    return {
        "semantics": sem,
        "domain": domain,
        "pred_meta": pred_meta,
        "tensors": cur_state,
        "iters": iters,
        "delta": last_delta,
        "converged": converged,
    }


def _emit_facts(
    torch,
    domain: List[str],
    pred_meta: Dict[str, int],
    tensors: Dict[str, Any],
    *,
    pred_filter: Optional[str],
    min_weight: float,
    limit: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pred, arity in pred_meta.items():
        if pred_filter is not None and pred != pred_filter:
            continue
        t = tensors[pred]
        if arity == 0:
            w = float(t.item())
            if w > min_weight:
                out.append({"pred": pred, "args": [], "weight": w})
            continue
        nz = torch.nonzero(t > min_weight, as_tuple=False)
        for row in nz.tolist():
            args = [domain[int(i)] for i in row]
            w = float(t[tuple(row)].item())
            out.append({"pred": pred, "args": args, "weight": w})
            if len(out) >= limit:
                return out
    return out


@dataclass(frozen=True)
class Label:
    pred: str
    args: List[str]
    target: float
    weight: float = 1.0


def _load_labels(path: Path) -> List[Label]:
    raw = _load_json(path)
    if not isinstance(raw, list):
        raise ValueError("--labels must be a JSON array")
    out: List[Label] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"labels[{i}]: expected object")
        pred = str(row.get("pred", ""))
        if not pred:
            raise ValueError(f"labels[{i}]: missing pred")
        args = row.get("args", [])
        if not isinstance(args, list):
            raise ValueError(f"labels[{i}]: args must be array")
        target = float(row.get("target", 0.0))
        weight = float(row.get("weight", 1.0))
        out.append(Label(pred=pred, args=[str(a) for a in args], target=target, weight=weight))
    return out


def _init_rule_weight_logits(torch, graph: Dict[str, Any], *, device, dtype):
    ws: List[float] = []
    for r in graph["rules"]:
        w = float(r.get("weight", 1.0))
        w = max(1e-4, min(1.0 - 1e-4, w))
        ws.append(w)
    w0 = torch.tensor(ws, device=device, dtype=dtype)
    return torch.log(w0) - torch.log(1.0 - w0)


def _init_fact_weight_logits(torch, graph: Dict[str, Any], *, device, dtype):
    ws: List[float] = []
    for f in graph["facts"]:
        w = float(f.get("weight", 0.0))
        w = max(1e-4, min(1.0 - 1e-4, w))
        ws.append(w)
    w0 = torch.tensor(ws, device=device, dtype=dtype)
    return torch.log(w0) - torch.log(1.0 - w0)


def _rule_weights_from_logits(torch, logits):
    return torch.sigmoid(logits)


def _load_checkpoint_tensor(
    torch,
    path: Path,
    *,
    spec_hash: str,
    device: str,
    dtype,
    allow_spec_mismatch: bool,
    tensor_name: str,
):
    save_file, safe_open = _require_safetensors()
    del save_file
    with safe_open(str(path), framework="pt", device=device) as f:
        meta = f.metadata() or {}
        ck_spec = meta.get("spec_hash")
        if ck_spec is not None and ck_spec != spec_hash and not allow_spec_mismatch:
            raise ValueError(
                f"checkpoint spec_hash mismatch: checkpoint={ck_spec} graph={spec_hash} "
                "(pass --allow-spec-mismatch to override)"
            )
        if tensor_name not in f.keys():
            return None
        return f.get_tensor(tensor_name).to(dtype=dtype)


def _apply_rule_weights_to_graph(graph: Dict[str, Any], ws: List[float]) -> None:
    if len(ws) != len(graph["rules"]):
        raise ValueError(f"checkpoint has {len(ws)} weights but graph has {len(graph['rules'])} rules")
    for i, w in enumerate(ws):
        graph["rules"][i]["weight"] = float(w)


def _apply_fact_weights_to_graph(graph: Dict[str, Any], ws: List[float]) -> None:
    if len(ws) != len(graph["facts"]):
        raise ValueError(f"checkpoint has {len(ws)} weights but graph has {len(graph['facts'])} facts")
    for i, w in enumerate(ws):
        graph["facts"][i]["weight"] = float(w)


def train_rule_weights(
    graph: Dict[str, Any],
    labels: List[Label],
    *,
    graph_path: Path,
    device: str,
    dtype: str,
    max_iter: Optional[int],
    epochs: int,
    lr: float,
    trainable: str,
    checkpoint_in: Optional[Path],
    checkpoint_out: Optional[Path],
    allow_spec_mismatch: bool,
) -> Dict[str, Any]:
    torch = _require_torch()
    spec_hash = _sha256_file(graph_path)

    sem = _parse_semantics(graph)
    if sem.mode != "fuzzy":
        raise ValueError(f"--train currently supports only mode=fuzzy (got {sem.mode})")
    if sem.and_kind != "mul":
        raise ValueError(f"--train currently supports only and_kind=mul (got {sem.and_kind})")

    torch_dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
    dev = torch.device(device)

    domain = list(graph["domain"]["symbols"])
    domain_idx = {s: i for i, s in enumerate(domain)}
    pred_meta: Dict[str, int] = {p["name"]: int(p["arity"]) for p in graph["predicates"]}

    label_rows: List[Tuple[str, Tuple[int, ...], float, float]] = []
    for lab in labels:
        if lab.pred not in pred_meta:
            raise ValueError(f"label predicate missing from graph: {lab.pred}")
        if len(lab.args) != int(pred_meta[lab.pred]):
            raise ValueError(
                f"label arity mismatch for {lab.pred}: {lab.args} (expected {pred_meta[lab.pred]})"
            )
        idx: List[int] = []
        for a in lab.args:
            if a not in domain_idx:
                raise ValueError(f"label constant missing from domain: {a}")
            idx.append(domain_idx[a])
        label_rows.append((lab.pred, tuple(idx), float(lab.target), float(lab.weight)))
    if not label_rows:
        raise ValueError("no labels provided")

    if trainable not in ("rules", "facts", "both"):
        raise ValueError(f"invalid --trainable: {trainable}")

    rule_logits = _init_rule_weight_logits(torch, graph, device=dev, dtype=torch_dtype) if trainable in ("rules", "both") else None
    fact_logits = _init_fact_weight_logits(torch, graph, device=dev, dtype=torch_dtype) if trainable in ("facts", "both") else None

    if checkpoint_in is not None:
        if rule_logits is not None:
            t = _load_checkpoint_tensor(
                torch,
                checkpoint_in,
                spec_hash=spec_hash,
                device=str(dev),
                dtype=torch_dtype,
                allow_spec_mismatch=allow_spec_mismatch,
                tensor_name="rule_weight_logits",
            )
            if t is not None:
                rule_logits = t.to(device=dev, dtype=torch_dtype)
        if fact_logits is not None:
            t = _load_checkpoint_tensor(
                torch,
                checkpoint_in,
                spec_hash=spec_hash,
                device=str(dev),
                dtype=torch_dtype,
                allow_spec_mismatch=allow_spec_mismatch,
                tensor_name="fact_weight_logits",
            )
            if t is not None:
                fact_logits = t.to(device=dev, dtype=torch_dtype)

    params = []
    if rule_logits is not None:
        rule_logits = torch.nn.Parameter(rule_logits)
        params.append(rule_logits)
    if fact_logits is not None:
        fact_logits = torch.nn.Parameter(fact_logits)
        params.append(fact_logits)
    opt = torch.optim.Adam(params, lr=float(lr))

    last_loss = 0.0
    for ep in range(int(epochs)):
        opt.zero_grad()
        weights = _rule_weights_from_logits(torch, rule_logits) if rule_logits is not None else None
        fact_weights = _rule_weights_from_logits(torch, fact_logits) if fact_logits is not None else None
        res = run_fixpoint(
            graph,
            device=device,
            dtype=dtype,
            override_max_iter=max_iter,
            override_eps=0.0,
            rule_weights=weights,
            fact_weights=fact_weights,
            stop_on_converge=False,
        )
        tensors = res["tensors"]

        preds = []
        tgts = []
        wts = []
        for pred, idx, target, wt in label_rows:
            preds.append(tensors[pred][idx])
            tgts.append(torch.tensor(target, device=dev, dtype=torch_dtype))
            wts.append(torch.tensor(wt, device=dev, dtype=torch_dtype))
        pred_t = torch.stack(preds)
        tgt_t = torch.stack(tgts)
        wt_t = torch.stack(wts)

        import torch.nn.functional as F  # type: ignore

        loss = F.binary_cross_entropy(pred_t, tgt_t, weight=wt_t, reduction="mean")
        last_loss = float(loss.item())
        loss.backward()
        opt.step()

        if ep == 0 or (ep + 1) % max(1, int(epochs) // 10) == 0 or ep == int(epochs) - 1:
            if rule_logits is not None:
                w_mean = float(torch.mean(_rule_weights_from_logits(torch, rule_logits)).item())
            else:
                w_mean = float("nan")
            print(
                f"[train] epoch {ep+1}/{epochs} loss={last_loss:.6f} mean_rule_weight={w_mean:.4f}",
                file=sys.stderr,
            )

    out: Dict[str, Any] = {"loss": last_loss, "spec_hash": spec_hash}
    ck_tensors: Dict[str, Any] = {}
    if rule_logits is not None:
        out_weights = _rule_weights_from_logits(torch, rule_logits).detach().cpu().tolist()
        _apply_rule_weights_to_graph(graph, out_weights)
        out["rule_weights"] = out_weights
        ck_tensors["rule_weight_logits"] = rule_logits.detach().cpu()
    if fact_logits is not None:
        out_facts = _rule_weights_from_logits(torch, fact_logits).detach().cpu().tolist()
        _apply_fact_weights_to_graph(graph, out_facts)
        out["fact_weights"] = out_facts
        ck_tensors["fact_weight_logits"] = fact_logits.detach().cpu()

    if checkpoint_out is not None:
        save_file, _safe_open = _require_safetensors()
        save_file(
            ck_tensors,
            str(checkpoint_out),
            metadata={
                "spec_hash": spec_hash,
                "format": "heytinglean.tensorlogic.checkpoint",
                "version": "1",
                "trainable": trainable,
            },
        )

    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True, help="Path to *.tensorgraph.json")
    p.add_argument("--device", default="cpu", help="torch device (cpu, cuda, ...)")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--pred", default=None, help="Only emit facts for this predicate")
    p.add_argument("--min-weight", type=float, default=0.0, help="Emit facts with weight > threshold")
    p.add_argument("--limit", type=int, default=2000, help="Max number of emitted facts")
    p.add_argument("--train", action="store_true", help="Train rule/fact weights from labeled facts (mode=fuzzy only)")
    p.add_argument("--labels", default=None, help="JSON array of labeled facts: [{pred,args,target,weight?}, ...]")
    p.add_argument("--epochs", type=int, default=200, help="Training epochs (when --train)")
    p.add_argument("--lr", type=float, default=0.1, help="Learning rate (when --train)")
    p.add_argument("--trainable", default="rules", choices=["rules", "facts", "both"], help="Which parameters to train")
    p.add_argument("--checkpoint-in", default=None, help="Load safetensors checkpoint for rule/fact weights")
    p.add_argument("--checkpoint-out", default=None, help="Save safetensors checkpoint for rule/fact weights (when --train)")
    p.add_argument("--allow-spec-mismatch", action="store_true", help="Allow checkpoint spec_hash mismatch")
    p.add_argument("--export-graph", default=None, help="Write an updated *.tensorgraph.json with learned weights")
    args = p.parse_args(list(argv) if argv is not None else None)

    graph_path = Path(args.graph)
    graph = _load_json(graph_path)
    torch = _require_torch()

    if args.train:
        if args.labels is None:
            raise SystemExit("--train requires --labels")
        train_rule_weights(
            graph,
            _load_labels(Path(args.labels)),
            graph_path=graph_path,
            device=args.device,
            dtype=args.dtype,
            max_iter=args.max_iter,
            epochs=int(args.epochs),
            lr=float(args.lr),
            trainable=str(args.trainable),
            checkpoint_in=Path(args.checkpoint_in) if args.checkpoint_in is not None else None,
            checkpoint_out=Path(args.checkpoint_out) if args.checkpoint_out is not None else None,
            allow_spec_mismatch=bool(args.allow_spec_mismatch),
        )

    if not args.train and args.checkpoint_in is not None:
        torch_dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
        dev = torch.device(args.device)
        spec_hash = _sha256_file(graph_path)
        rule_logits = _load_checkpoint_tensor(
            torch,
            Path(args.checkpoint_in),
            spec_hash=spec_hash,
            device=str(dev),
            dtype=torch_dtype,
            allow_spec_mismatch=bool(args.allow_spec_mismatch),
            tensor_name="rule_weight_logits",
        )
        if rule_logits is not None:
            ws = _rule_weights_from_logits(torch, rule_logits).detach().cpu().tolist()
            _apply_rule_weights_to_graph(graph, ws)

        fact_logits = _load_checkpoint_tensor(
            torch,
            Path(args.checkpoint_in),
            spec_hash=spec_hash,
            device=str(dev),
            dtype=torch_dtype,
            allow_spec_mismatch=bool(args.allow_spec_mismatch),
            tensor_name="fact_weight_logits",
        )
        if fact_logits is not None:
            ws = _rule_weights_from_logits(torch, fact_logits).detach().cpu().tolist()
            _apply_fact_weights_to_graph(graph, ws)

    if args.export_graph is not None and (args.train or args.checkpoint_in is not None):
        out_path = Path(args.export_graph)
        out_path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    res = run_fixpoint(
        graph,
        device=args.device,
        dtype=args.dtype,
        override_max_iter=args.max_iter,
        override_eps=args.eps,
    )

    sem: Semantics = res["semantics"]
    facts = _emit_facts(
        torch,
        res["domain"],
        res["pred_meta"],
        res["tensors"],
        pred_filter=args.pred,
        min_weight=float(args.min_weight),
        limit=int(args.limit),
    )

    payload = {
        "meta": {
            "source_graph": str(graph_path),
            "mode": sem.mode,
            "tnorm": sem.tnorm,
            "and_kind": sem.and_kind,
            "or_kind": sem.or_kind,
            "iters": res["iters"],
            "delta": res["delta"],
            "converged": res["converged"],
            "device": args.device,
            "dtype": args.dtype,
        },
        "facts": facts,
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if res["converged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
