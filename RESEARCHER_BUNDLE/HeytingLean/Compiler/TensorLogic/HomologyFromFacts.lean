import Std

import HeytingLean.Compiler.TensorLogic.AST
import HeytingLean.Computational.Homology.ChainComplex

/-!
Bridge: build an `F₂` chain complex from TensorLogic-style ground facts describing a (≤2)-dimensional
simplicial complex.

This is the missing “XOR layer” for β₁/β₂: once we have boundary matrices over `F₂`, we can reuse
Gaussian elimination (`F2Matrix.rank`) and `ChainComplexF2.bettis`.

Supported predicates (weights are ignored except that weight 0 drops the fact):
- `vertex(V)` (0-simplices)
- `edge(V1,V2)` (1-simplices; undirected, order ignored)
- `face_edge(F,V1,V2)` (2-simplex boundary incidence; order ignored)
-/

namespace HeytingLean
namespace Compiler
namespace TensorLogic

open Std
open HeytingLean.Computational.Homology

namespace HomologyFromFacts

structure UndirEdge where
  a : String
  b : String
  deriving Repr, BEq, Hashable, DecidableEq

instance : Inhabited UndirEdge :=
  ⟨{ a := "", b := "" }⟩

private def mkEdge (v1 v2 : String) : UndirEdge :=
  if v1 ≤ v2 then
    { a := v1, b := v2 }
  else
    { a := v2, b := v1 }

private def pushUnique [BEq α] (xs : Array α) (x : α) : Array α :=
  if xs.any (fun y => y == x) then xs else xs.push x

private def getArg? (a : Atom) (i : Nat) : Option String :=
  a.args[i]?

private def arityErr (pred : String) (got expected : Nat) : String :=
  s!"bad arity for '{pred}': got {got}, expected {expected}"

private def mkIndexMap (xs : Array String) : HashMap String Nat :=
  Id.run do
    let mut m : HashMap String Nat := {}
    for i in [:xs.size] do
      m := m.insert xs[i]! i
    m

private def mkEdgeIndexMap (xs : Array UndirEdge) : HashMap UndirEdge Nat :=
  Id.run do
    let mut m : HashMap UndirEdge Nat := {}
    for i in [:xs.size] do
      m := m.insert xs[i]! i
    m

private def lookupIndex (m : HashMap String Nat) (k : String) : Except String Nat :=
  match m.get? k with
  | some i => pure i
  | none => throw s!"internal error: missing index for '{k}'"

/-- Build an `F₂` chain complex `C₂ → C₁ → C₀` (or a truncated version) from ground facts. -/
def chainComplexF2 (facts : List (Atom × Float)) : Except String ChainComplexF2 := do
  let mut vertices : Array String := #[]
  let mut edges : Array UndirEdge := #[]
  let mut faces : Array String := #[]
  let mut faceEdges : HashMap String (Array UndirEdge) := {}

  for (atom, w) in facts do
    if w == 0.0 then
      continue
    match atom.pred with
    | "vertex" =>
        if atom.arity != 1 then
          throw (arityErr atom.pred atom.arity 1)
        match getArg? atom 0 with
        | none => throw (arityErr atom.pred atom.arity 1)
        | some v => vertices := pushUnique vertices v
    | "edge" =>
        if atom.arity != 2 then
          throw (arityErr atom.pred atom.arity 2)
        match getArg? atom 0, getArg? atom 1 with
        | some v1, some v2 =>
            vertices := pushUnique vertices v1
            vertices := pushUnique vertices v2
            if v1 = v2 then
              throw s!"degenerate edge '{v1},{v2}'"
            edges := pushUnique edges (mkEdge v1 v2)
        | _, _ => throw (arityErr atom.pred atom.arity 2)
    | "face_edge" =>
        if atom.arity != 3 then
          throw (arityErr atom.pred atom.arity 3)
        match getArg? atom 0, getArg? atom 1, getArg? atom 2 with
        | some f, some v1, some v2 =>
            vertices := pushUnique vertices v1
            vertices := pushUnique vertices v2
            if v1 = v2 then
              throw s!"degenerate face_edge '{f}' uses '{v1},{v2}'"
            let e := mkEdge v1 v2
            edges := pushUnique edges e
            faces := pushUnique faces f
            let es := faceEdges.getD f #[]
            faceEdges := faceEdges.insert f (pushUnique es e)
        | _, _, _ => throw (arityErr atom.pred atom.arity 3)
    | _ => continue

  if vertices.isEmpty then
    throw "no vertices found (expected facts like: vertex\\tv0)"

  let vIndex := mkIndexMap vertices
  let eIndex := mkEdgeIndexMap edges

  -- ∂₁ : C₁ → C₀  (rows = |V|, cols = |E|)
  let mut d1Cols : Array (Array Nat) := Array.mkEmpty edges.size
  for e in edges do
    let i1 ← lookupIndex vIndex e.a
    let i2 ← lookupIndex vIndex e.b
    d1Cols := d1Cols.push #[i1, i2]

  -- ∂₂ : C₂ → C₁  (rows = |E|, cols = |F|)
  let mut d2Cols : Array (Array Nat) := Array.mkEmpty faces.size
  for f in faces do
    let es := faceEdges.getD f #[]
    if es.size != 3 then
      throw s!"face '{f}' has {es.size} edges (expected 3)"
    let mut col : Array Nat := Array.mkEmpty es.size
    for e in es do
      match eIndex.get? e with
      | none => throw s!"internal error: missing edge index for '{e.a},{e.b}'"
      | some j => col := col.push j
    d2Cols := d2Cols.push col

  let n0 := vertices.size
  let n1 := edges.size
  let n2 := faces.size

  if n2 > 0 && n1 == 0 then
    throw "found faces but no edges"

  let mut dims : Array Nat := #[n0]
  let mut boundaries : Array F2Matrix := #[]

  if n1 > 0 then
    dims := dims.push n1
    boundaries := boundaries.push (← F2Matrix.ofColOnes n0 n1 d1Cols)

  if n2 > 0 then
    dims := dims.push n2
    boundaries := boundaries.push (← F2Matrix.ofColOnes n1 n2 d2Cols)

  let C : ChainComplexF2 := { dims := dims, boundaries := boundaries }
  C.validateShapes
  C.checkD2
  pure C

end HomologyFromFacts

end TensorLogic
end Compiler
end HeytingLean
