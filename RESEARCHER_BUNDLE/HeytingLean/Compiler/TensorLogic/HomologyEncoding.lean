import Std

import HeytingLean.Compiler.TensorLogic.AST
import HeytingLean.Computational.Homology.ChainComplex

namespace HeytingLean
namespace Computational
namespace Homology

open HeytingLean.Compiler.TensorLogic

namespace ChainComplexF2

private def vertexName (i : Nat) : String :=
  s!"v{i}"

private def edgeName (i : Nat) : String :=
  s!"e{i}"

private def faceName (i : Nat) : String :=
  s!"f{i}"

private def simplexName (k i : Nat) : String :=
  match k with
  | 0 => vertexName i
  | 1 => edgeName i
  | 2 => faceName i
  | _ => s!"s{k}_{i}"

private def mkAtom (pred : String) (args : List String) : Atom :=
  { pred := pred, args := args }

private def mkPat (pred : String) (args : List String) : AtomPat :=
  { pred := pred, args := args.map Sym.ofString }

private def connectivityProgram : Program :=
  { rules :=
      [ { head := mkPat "adjacent" ["V1", "V2"]
        , body := [mkPat "edge" ["V1", "V2"]]
        }
      , { head := mkPat "adjacent" ["V1", "V2"]
        , body := [mkPat "edge" ["V2", "V1"]]
        }
      , { head := mkPat "adjacent" ["V1", "V2"]
        , body := [mkPat "edge_vertex" ["E", "V1"], mkPat "edge_vertex" ["E", "V2"]]
        }
      , { head := mkPat "connected" ["V", "V"]
        , body := [mkPat "vertex" ["V"]]
        }
      , { head := mkPat "connected" ["V1", "V2"]
        , body := [mkPat "adjacent" ["V1", "V2"]]
        }
      , { head := mkPat "connected" ["V1", "V3"]
        , body := [mkPat "adjacent" ["V1", "V2"], mkPat "connected" ["V2", "V3"]]
        }
      ]
  }

private def edgeEndpoints? (d1 : F2Matrix) (e : Nat) : Option (String × String) :=
  if e < d1.cols then
    let vs : Array Nat := Id.run do
      let mut acc : Array Nat := #[]
      for i in [:d1.rows] do
        if d1.data[i]![e]! then
          acc := acc.push i
      acc
    if vs.size = 2 then
      let a := vs[0]!
      let b := vs[1]!
      let v1 := vertexName (Nat.min a b)
      let v2 := vertexName (Nat.max a b)
      some (v1, v2)
    else
      none
  else
    none

/-- Encode a finite `F₂` chain complex as a positive logic program + ground facts.

Outputs:
- facts describing the basis "simplices" (`vertex`, `edge_id`, `face`, `simplex`)
- boundary incidence facts (`boundary`, plus specialized `edge_vertex` / `face_edge_idx`)
- derived geometric facts when available (`edge(V1,V2)`, `face_edge(F,V1,V2)`)
- a small connectivity program (`adjacent` / `connected`) to witness `β₀` (components) via reachability.

Names:
- vertices are `v0`, `v1`, ...
- edges are `e0`, `e1`, ...
- faces are `f0`, `f1`, ...
- higher simplices are `s{k}_{i}`.
-/
def toLogicProgram (C : ChainComplexF2) : Program × List (Atom × Float) :=
  let prog := connectivityProgram
  let facts : Array (Atom × Float) := Id.run do
    let mut facts : Array (Atom × Float) := #[]

    -- Basis elements
    for k in [:C.dims.size] do
      let nk := C.dims[k]!
      for i in [:nk] do
        let name := simplexName k i
        facts := facts.push (mkAtom "simplex" [toString k, name], 1.0)
        if k == 0 then
          facts := facts.push (mkAtom "vertex" [name], 1.0)
        else if k == 1 then
          facts := facts.push (mkAtom "edge_id" [name], 1.0)
        else if k == 2 then
          facts := facts.push (mkAtom "face" [name], 1.0)

    -- Boundary incidence
    for k in [:C.boundaries.size] do
      let M := C.boundaries[k]!
      for i in [:M.rows] do
        let row := M.data[i]!
        for j in [:M.cols] do
          if row[j]! then
            let low := simplexName k i
            let high := simplexName (k + 1) j
            facts := facts.push (mkAtom "boundary" [toString k, low, high], 1.0)
            if k == 0 then
              facts := facts.push (mkAtom "edge_vertex" [high, low], 1.0)
            else if k == 1 then
              facts := facts.push (mkAtom "face_edge_idx" [high, low], 1.0)

    -- Derived `edge(V1,V2)` facts when ∂₁ columns have exactly two vertices.
    let edgePairs : Array (Option (String × String)) :=
      if 0 < C.boundaries.size then
        let d1 := C.boundaries[0]!
        Id.run do
          let mut acc : Array (Option (String × String)) := Array.mkEmpty d1.cols
          for e in [:d1.cols] do
            acc := acc.push (edgeEndpoints? d1 e)
          acc
      else
        #[]

    for e in [:edgePairs.size] do
      match edgePairs[e]! with
      | none => continue
      | some (v1, v2) =>
          facts := facts.push (mkAtom "edge" [v1, v2], 1.0)

    -- Derived `face_edge(F,V1,V2)` facts when ∂₂ columns select edges with known endpoints.
    if 1 < C.boundaries.size then
      let d2 := C.boundaries[1]!
      for e in [:d2.rows] do
        for f in [:d2.cols] do
          if d2.data[e]![f]! then
            match edgePairs.getD e none with
            | none => continue
            | some (v1, v2) =>
                facts := facts.push (mkAtom "face_edge" [faceName f, v1, v2], 1.0)

    facts

  (prog, facts.toList)

end ChainComplexF2

end Homology
end Computational
end HeytingLean
