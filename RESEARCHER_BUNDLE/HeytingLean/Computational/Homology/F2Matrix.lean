import Std

namespace HeytingLean
namespace Computational
namespace Homology

open Std

/-- A small, executable matrix over `F₂`, represented row-major as `Bool`s. -/
structure F2Matrix where
  rows : Nat
  cols : Nat
  data : Array (Array Bool)   -- each row has length `cols`
  deriving Repr

namespace F2Matrix

instance : Inhabited F2Matrix :=
  ⟨{ rows := 0, cols := 0, data := #[] }⟩

def zero (rows cols : Nat) : F2Matrix :=
  { rows := rows
  , cols := cols
  , data := Array.replicate rows (Array.replicate cols false)
  }

/-- Build a matrix from a "columns of ones" sparse encoding.

`colOnes[j]` lists the row indices `i` where `M[i,j] = 1`.

This is convenient for boundary matrices of simplicial complexes over `F₂`. -/
def ofColOnes (rows cols : Nat) (colOnes : Array (Array Nat)) : Except String F2Matrix := do
  if colOnes.size != cols then
    throw s!"expected {cols} columns, got {colOnes.size}"
  let mut data : Array (Array Bool) := Array.replicate rows (Array.replicate cols false)
  for j in [:cols] do
    for i in colOnes[j]! do
      if i < rows then
        let row := data[i]!
        data := data.set! i (row.set! j true)
      else
        throw s!"row index out of bounds: {i} (rows={rows})"
  pure { rows := rows, cols := cols, data := data }

private def rowXor (a b : Array Bool) : Array Bool :=
  let m := Nat.min a.size b.size
  Id.run do
    let mut out : Array Bool := Array.mkEmpty m
    for i in [:m] do
      out := out.push (Bool.xor a[i]! b[i]!)
    out

def isZero (M : F2Matrix) : Bool :=
  M.data.all (fun row => row.all (fun b => b == false))

def mul (A B : F2Matrix) : Except String F2Matrix := do
  if A.cols != B.rows then
    throw s!"dimension mismatch: {A.rows}×{A.cols} times {B.rows}×{B.cols}"
  let n := A.rows
  let k := A.cols
  let m := B.cols
  let out := Id.run do
    let mut rowsAcc : Array (Array Bool) := Array.mkEmpty n
    for i in [:n] do
      let rowA := A.data[i]!
      let mut rowC : Array Bool := Array.mkEmpty m
      for j in [:m] do
        let s := Id.run do
          let mut t : Bool := false
          for p in [:k] do
            t := Bool.xor t (rowA[p]! && B.data[p]![j]!)
          t
        rowC := rowC.push s
      rowsAcc := rowsAcc.push rowC
    rowsAcc
  pure { rows := n, cols := m, data := out }

private def findPivotRow (data : Array (Array Bool)) (col : Nat) (start : Nat) : Option Nat :=
  let rec loop (i : Nat) : Option Nat :=
    if h : i < data.size then
      if data[i]![col]! then
        some i
      else
        loop (i + 1)
    else
      none
  loop start

/-- Compute matrix rank over `F₂` by row reduction (Gaussian elimination with XOR). -/
def rank (M : F2Matrix) : Nat :=
  Id.run do
    let mut data := M.data
    let mut pivotRow : Nat := 0
    let mut r : Nat := 0

    for col in [:M.cols] do
      if pivotRow ≥ M.rows then
        continue
      match findPivotRow data col pivotRow with
      | none => continue
      | some pr =>
          if pr != pivotRow then
            let tmp := data[pr]!
            data := (data.set! pr data[pivotRow]!).set! pivotRow tmp
          let piv := data[pivotRow]!
          for i in [:M.rows] do
            if i != pivotRow && data[i]![col]! then
              data := data.set! i (rowXor data[i]! piv)
          pivotRow := pivotRow + 1
          r := r + 1

    r

end F2Matrix

end Homology
end Computational
end HeytingLean
