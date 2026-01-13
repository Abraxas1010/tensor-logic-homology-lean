import Lean.Data.Json
import Std

import HeytingLean.Computational.Homology.F2Matrix

namespace HeytingLean
namespace Computational
namespace Homology

open Std
open Lean
open Lean.Json

/-- A small finite chain complex over `F₂`, represented by boundary matrices.

Conventions:
- `dims[k] = nₖ = dim(Cₖ)`
- `boundaries[k]` represents `∂ₖ₊₁ : Cₖ₊₁ → Cₖ`, so it has shape `nₖ × nₖ₊₁`.
- Validity requires `∂ₖ ∘ ∂ₖ₊₁ = 0` for all `k`. -/
structure ChainComplexF2 where
  dims : Array Nat
  boundaries : Array F2Matrix
  deriving Repr

namespace ChainComplexF2

def maxDim (C : ChainComplexF2) : Nat :=
  if C.dims.isEmpty then 0 else C.dims.size - 1

def validateShapes (C : ChainComplexF2) : Except String Unit := do
  if C.dims.isEmpty then
    throw "dims must be nonempty"
  if C.boundaries.size + 1 != C.dims.size then
    throw s!"expected boundaries length {C.dims.size - 1}, got {C.boundaries.size}"
  for k in [:C.boundaries.size] do
    let M := C.boundaries[k]!
    let rowsExpected := C.dims[k]!
    let colsExpected := C.dims[k+1]!
    if M.rows != rowsExpected || M.cols != colsExpected then
      throw s!"boundary[{k}] has shape {M.rows}×{M.cols}, expected {rowsExpected}×{colsExpected}"

def checkD2 (C : ChainComplexF2) : Except String Unit := do
  for k in [:C.boundaries.size] do
    if k + 1 < C.boundaries.size then
      let A := C.boundaries[k]!
      let B := C.boundaries[k+1]!
      let AB ← F2Matrix.mul A B
      if !AB.isZero then
        throw s!"∂²≠0 at k={k} (boundary[{k}] ∘ boundary[{k+1}])"
    else
      pure ()

def boundaryRanks (C : ChainComplexF2) : Array Nat :=
  C.boundaries.map (fun M => M.rank)

def bettiAt (C : ChainComplexF2) (k : Nat) : Except String Nat := do
  if k < C.dims.size then
    let n := C.dims[k]!
    let ranks := C.boundaryRanks
    let rPrev := if k == 0 then 0 else ranks[k-1]!
    let rNext := if k < ranks.size then ranks[k]! else 0
    if rPrev + rNext > n then
      throw s!"invalid ranks for β_{k}: {rPrev}+{rNext} > {n}"
    pure (n - rPrev - rNext)
  else
    throw s!"k out of range: {k} (dims length={C.dims.size})"

def bettis (C : ChainComplexF2) : Except String (Array Nat) := do
  let mut out : Array Nat := Array.mkEmpty C.dims.size
  for k in [:C.dims.size] do
    out := out.push (← C.bettiAt k)
  pure out

/-! ## JSON parsing -/

private def expectObj (j : Json) : Except String Json := do
  match j.getObj? with
  | .ok _ => pure j
  | .error _ => throw "expected JSON object"

private def getArr (obj : Json) (k : String) : Except String (Array Json) := do
  match obj.getObjVal? k with
  | .error _ => throw s!"missing field '{k}'"
  | .ok v =>
      match v.getArr? with
      | .ok a => pure a
      | .error _ => throw s!"field '{k}' not an array"

private def parseNat (j : Json) : Except String Nat := do
  match j.getNum? with
  | .ok n =>
      match String.toNat? n.toString with
      | some k => pure k
      | none => throw s!"expected Nat, got {n}"
  | .error _ => throw "expected JSON number"

private def parseNatArray (j : Json) : Except String (Array Nat) := do
  match j with
  | Json.arr xs =>
      let mut out : Array Nat := Array.mkEmpty xs.size
      for v in xs do
        out := out.push (← parseNat v)
      pure out
  | _ => throw "expected JSON array"

private def parseColOnes (j : Json) : Except String (Array (Array Nat)) := do
  match j with
  | Json.arr cols =>
      let mut out : Array (Array Nat) := Array.mkEmpty cols.size
      for c in cols do
        out := out.push (← parseNatArray c)
      pure out
  | _ => throw "expected JSON array for 'cols'"

/-- Parse a chain complex in the following JSON shape:

```json
{
  "dims": [n0, n1, ..., nd],
  "boundaries": [
    { "cols": [ [..], .. ] },  // ∂1 : C1→C0, n0×n1 (cols length = n1)
    { "cols": [ [..], .. ] }   // ∂2 : C2→C1, n1×n2
  ]
}
``` -/
def parseJson (j : Json) : Except String ChainComplexF2 := do
  let obj ← expectObj j
  let dimsJson ← getArr obj "dims"
  let mut dims : Array Nat := Array.mkEmpty dimsJson.size
  for v in dimsJson do
    dims := dims.push (← parseNat v)
  if dims.isEmpty then
    throw "dims must be nonempty"
  let boundsJson ← getArr obj "boundaries"
  if boundsJson.size + 1 != dims.size then
    throw s!"expected boundaries length {dims.size - 1}, got {boundsJson.size}"
  let mut boundaries : Array F2Matrix := Array.mkEmpty boundsJson.size
  for k in [:boundsJson.size] do
    let bObj ← expectObj boundsJson[k]!
    let colsJson ←
      match bObj.getObjVal? "cols" with
      | .error _ => throw s!"missing field 'cols' in boundaries[{k}]"
      | .ok v => pure v
    let colOnes ← parseColOnes colsJson
    let rows := dims[k]!
    let cols := dims[k+1]!
    boundaries := boundaries.push (← F2Matrix.ofColOnes rows cols colOnes)
  let C : ChainComplexF2 := { dims := dims, boundaries := boundaries }
  C.validateShapes
  pure C

end ChainComplexF2

end Homology
end Computational
end HeytingLean
