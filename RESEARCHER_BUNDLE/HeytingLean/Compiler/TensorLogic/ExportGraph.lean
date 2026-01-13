import Std
import Std.Data.TreeMap
import Std.Data.TreeSet
import Lean.Data.Json

import HeytingLean.Compiler.TensorLogic.AST
import HeytingLean.Compiler.TensorLogic.Eval

namespace HeytingLean
namespace Compiler
namespace TensorLogic

open Std
open Lean
open Lean.Json

namespace ExportGraph

private def jsonNumOrStr (x : Float) : Json :=
  match JsonNumber.fromFloat? x with
  | .inr n => Json.num n
  | .inl s => Json.str s

private def symToJson : Sym → Json
  | .var v => Json.mkObj [("var", Json.str v)]
  | .const c => Json.mkObj [("const", Json.str c)]

private def atomPatToJson (a : AtomPat) : Json :=
  Json.mkObj
    [ ("pred", Json.str a.pred)
    , ("args", Json.arr (a.args.map symToJson |>.toArray))
    ]

private def insertAll (s : TreeSet String) (xs : List String) : TreeSet String :=
  xs.foldl (fun acc x => acc.insert x) s

private def collectDomain (p : Program) (baseFacts : List (Atom × Float)) : List String :=
  let fromFacts :=
    baseFacts.foldl (init := ({} : TreeSet String)) (fun acc (aw : Atom × Float) =>
      insertAll acc aw.1.args)
  let fromRules :=
    p.rules.foldl (init := fromFacts) (fun acc r =>
      let acc := insertAll acc r.head.consts
      let acc := insertAll acc (r.body.flatMap AtomPat.consts)
      acc)
  fromRules.toList

private def checkArity (m : TreeMap String Nat) (pred : String) (arity : Nat) :
    Except String (TreeMap String Nat) := do
  match m.get? pred with
  | none => pure (m.insert pred arity)
  | some n =>
      if n = arity then
        pure m
      else
        throw s!"arity mismatch for predicate '{pred}': saw {n} and {arity}"

private def collectPredArities (p : Program) (baseFacts : List (Atom × Float)) :
    Except String (TreeMap String Nat) := do
  let m1 ←
    p.rules.foldlM (init := ({} : TreeMap String Nat)) (fun m r => do
      let m ← checkArity m r.head.pred r.head.arity
      r.body.foldlM (init := m) (fun m a => checkArity m a.pred a.arity))
  baseFacts.foldlM (init := m1) (fun m aw => checkArity m aw.1.pred aw.1.arity)

private def modeToString : Mode → String
  | .boolean => "boolean"
  | .f2 => "f2"
  | .fuzzy => "fuzzy"
  | .heyting => "heyting"

private def tnormToString : TNorm → String
  | .product => "product"
  | .lukasiewicz => "lukasiewicz"

private def semanticsAndKind (mode : Mode) (tn : TNorm) : String :=
  match mode with
  | .boolean => "bool_and"
  | .f2 => "bool_and"
  | .heyting => "min"
  | .fuzzy =>
      match tn with
      | .product => "mul"
      | .lukasiewicz => "luk_and"

private def semanticsOrKind (mode : Mode) (tn : TNorm) : String :=
  match mode with
  | .boolean => "bool_or"
  | .f2 => "xor"
  | .heyting => "max"
  | .fuzzy =>
      match tn with
      | .product => "noisy_or"
      | .lukasiewicz => "luk_or"

private def q16Scale : Nat := 65536

private def quantizeQ16 (x : Float) : Nat :=
  -- clamp for the fixed-point view; the float view remains unchanged.
  let x :=
    if x < 0.0 then 0.0 else if x > 1.0 then 1.0 else x
  let q : Int64 := Float.toInt64 (Float.round (x * Float.ofNat q16Scale))
  q.toNatClampNeg

private def lexLt (a b : List String) : Bool :=
  match a, b with
  | [], [] => false
  | [], _ => true
  | _, [] => false
  | x :: xs, y :: ys =>
      if x < y then true
      else if x = y then lexLt xs ys
      else false

private def atomLt (a b : Atom) : Bool :=
  if a.pred < b.pred then true
  else if a.pred = b.pred then lexLt a.args b.args
  else false

private def canonicalizeBaseFacts (cfg : RunConfig) (facts : List (Atom × Float)) : Facts :=
  let ops := Ops.forConfig cfg.mode cfg.tnorm
  match cfg.mode with
  | .boolean => Facts.fromList ops (facts.map (fun (a, _) => (a, 1.0)))
  | .f2 => Facts.fromList ops (facts.map (fun (a, w) => (a, if w == 0.0 then 0.0 else 1.0)))
  | .fuzzy => Facts.fromList ops facts
  | .heyting => Facts.fromList ops facts

private def factToJson
    (domainIdx : HashMap String Nat)
    (predIdx : HashMap String Nat)
    (a : Atom) (w : Float) : Except String Json := do
  let argIds ←
    a.args.mapM (fun c =>
      match domainIdx.get? c with
      | some i => pure i
      | none => throw s!"internal error: constant '{c}' missing from domain")
  let pid :=
    match predIdx.get? a.pred with
    | some i => i
    | none => 0
  pure <|
    Json.mkObj
      [ ("pred", Json.str a.pred)
      , ("pred_id", Json.num (JsonNumber.fromNat pid))
      , ("args", Json.arr (a.args.map Json.str |>.toArray))
      , ("arg_ids", Json.arr (argIds.map (fun i => Json.num (JsonNumber.fromNat i)) |>.toArray))
      , ("weight", jsonNumOrStr w)
      , ("q16", Json.num (JsonNumber.fromNat (quantizeQ16 w)))
      ]

private def ruleVars (r : Rule) : List String :=
  let s : TreeSet String :=
    (r.varsHead ++ r.varsBody).foldl (init := ({} : TreeSet String)) (fun acc v => acc.insert v)
  s.toList

private def ruleElimVars (r : Rule) : List String :=
  let head : TreeSet String :=
    r.varsHead.foldl (init := ({} : TreeSet String)) (fun acc v => acc.insert v)
  let body : TreeSet String :=
    r.varsBody.foldl (init := ({} : TreeSet String)) (fun acc v => acc.insert v)
  body.toList.filter (fun v => !(head.contains v))

private def ruleToJson (r : Rule) : Json :=
  let vars := ruleVars r
  let elim := ruleElimVars r
  let body := Json.arr (r.body.map atomPatToJson |>.toArray)
  let w := r.weight.getD 1.0
  Json.mkObj
    [ ("head", atomPatToJson r.head)
    , ("body", body)
    , ("weight", jsonNumOrStr w)
    , ("vars", Json.arr (vars.map Json.str |>.toArray))
    , ("elim_vars", Json.arr (elim.map Json.str |>.toArray))
    ]

/-- Export a TensorLogic program + base facts as a backend-neutral “tensor graph IR” JSON.

This is intentionally a *graph/spec* rather than an executable model:
downstream tensor backends (PyTorch/JAX/Candle/etc) can lower it to einsum/contract operations
and run fixpoint iterations as described in the `semantics.fixpoint` section.
-/
def tensorGraphJson
    (cfg : RunConfig)
    (sharpness : Float)
    (p : Program)
    (facts : List (Atom × Float)) : Except String Json := do
  let baseF := canonicalizeBaseFacts cfg facts
  let baseFacts := (baseF.toList.toArray.qsort (fun a b => atomLt a.1 b.1)).toList
  let domain := collectDomain p baseFacts
  let predArities := (← collectPredArities p baseFacts).toList
  let preds := predArities.map (fun (nm, ar) => (nm, ar))

  let domainIdx : HashMap String Nat :=
    domain.zipIdx.foldl (init := ({} : HashMap String Nat)) (fun m (ci : String × Nat) => m.insert ci.1 ci.2)
  let predIdx : HashMap String Nat :=
    preds.zipIdx.foldl (init := ({} : HashMap String Nat)) (fun m (pi : (String × Nat) × Nat) => m.insert pi.1.1 pi.2)

  let extSet : TreeSet String :=
    baseFacts.foldl (init := ({} : TreeSet String)) (fun acc (aw : Atom × Float) => acc.insert aw.1.pred)
  let headSet : TreeSet String :=
    p.rules.foldl (init := ({} : TreeSet String)) (fun acc r => acc.insert r.head.pred)

  let predObjs : Array Json :=
    (preds.zipIdx.map (fun (pinfo, i) =>
        let nm := pinfo.1
        let ar := pinfo.2
        let roles :=
          (#[] : Array Json)
            |> (fun acc => if extSet.contains nm then acc.push (Json.str "extensional") else acc)
            |> (fun acc => if headSet.contains nm then acc.push (Json.str "intensional") else acc)
        Json.mkObj
          [ ("id", Json.num (JsonNumber.fromNat i))
          , ("name", Json.str nm)
          , ("arity", Json.num (JsonNumber.fromNat ar))
          , ("roles", Json.arr roles)
          ])
      ).toArray

  let factsJson ← baseFacts.mapM (fun (aw : Atom × Float) => factToJson domainIdx predIdx aw.1 aw.2)

  let semantics :=
    Json.mkObj
      [ ("mode", Json.str (modeToString cfg.mode))
      , ("tnorm", Json.str (tnormToString cfg.tnorm))
      , ("and_kind", Json.str (semanticsAndKind cfg.mode cfg.tnorm))
      , ("or_kind", Json.str (semanticsOrKind cfg.mode cfg.tnorm))
      , ("sharpness", jsonNumOrStr sharpness)
      , ("fixpoint", Json.mkObj
          [ ("kind", Json.str "anchored_step_from_base")
          , ("max_iter", Json.num (JsonNumber.fromNat cfg.maxIter))
          , ("eps", jsonNumOrStr cfg.eps)
          ])
      ]

  pure <|
    Json.mkObj
      [ ("format", Json.str "heytinglean.tensorlogic.tensorgraph")
      , ("version", Json.num (JsonNumber.fromNat 1))
      , ("semantics", semantics)
      , ("truth_values", Json.mkObj
          [ ("float", Json.mkObj [("kind", Json.str "float")])
          , ("q16", Json.mkObj
              [ ("kind", Json.str "q16.16")
              , ("scale", Json.num (JsonNumber.fromNat q16Scale))
              ])
          ])
      , ("domain", Json.mkObj
          [ ("size", Json.num (JsonNumber.fromNat domain.length))
          , ("symbols", Json.arr (domain.map Json.str |>.toArray))
          ])
      , ("predicates", Json.arr predObjs)
      , ("facts", Json.arr (factsJson.toArray))
      , ("rules", Json.arr (p.rules.map ruleToJson |>.toArray))
      ]

end ExportGraph

end TensorLogic
end Compiler
end HeytingLean
