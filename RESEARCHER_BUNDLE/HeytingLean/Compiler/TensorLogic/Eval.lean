import Std
import HeytingLean.Compiler.TensorLogic.AST

namespace HeytingLean
namespace Compiler
namespace TensorLogic

open Std

abbrev Subst := List (String × String)

namespace Subst

def lookup (s : Subst) (v : String) : Option String :=
  (s.find? (fun kv => kv.1 = v)).map (·.2)

def bindVar (s : Subst) (v : String) (c : String) : Option Subst :=
  match lookup s v with
  | none => some ((v, c) :: s)
  | some c' => if c' = c then some s else none

end Subst

private def clamp01 (x : Float) : Float :=
  if x < 0.0 then 0.0 else if x > 1.0 then 1.0 else x

private def maxF (a b : Float) : Float :=
  if a < b then b else a

private def bit (x : Float) : Bool :=
  x != 0.0

private def bitToFloat (b : Bool) : Float :=
  if b then 1.0 else 0.0

inductive Mode where
  | boolean
  | f2      -- F₂ semiring: AND = Boolean AND, OR = XOR (addition mod 2)
  | fuzzy
  | heyting  -- Gödel t-norm: min/max with intuitionistic implication
  deriving Repr, BEq, DecidableEq

inductive TNorm where
  | product
  | lukasiewicz
  deriving Repr, BEq, DecidableEq

structure Ops where
  and : Float → Float → Float
  or : Float → Float → Float

namespace Ops

def forConfig (mode : Mode) (tn : TNorm := .product) : Ops :=
  match mode with
  | .boolean =>
      { and := fun a b => if (a == 0.0) || (b == 0.0) then 0.0 else 1.0
      , or := fun a b => if (a == 0.0) && (b == 0.0) then 0.0 else 1.0
      }
  | .f2 =>
      { and := fun a b => bitToFloat (bit a && bit b)
      , or := fun a b => bitToFloat (Bool.xor (bit a) (bit b))
      }
  | .fuzzy =>
      match tn with
      | .product =>
          { and := fun a b => clamp01 (a * b)
          , or := fun a b => clamp01 (a + b - a * b)
          }
      | .lukasiewicz =>
          { and := fun a b =>
              let t := a + b - 1.0
              if t < 0.0 then 0.0 else t
          , or := fun a b =>
              let t := a + b
              if t > 1.0 then 1.0 else t
          }
  | .heyting =>
      -- Gödel t-norm: min/max with intuitionistic semantics
      { and := fun a b => if a < b then a else b  -- min
      , or := fun a b => if a > b then a else b   -- max
      }

end Ops

abbrev Facts := HashMap Atom Float

namespace Facts

def empty : Facts := {}

def get (F : Facts) (a : Atom) : Float :=
  F.getD a 0.0

def addOr (ops : Ops) (F : Facts) (a : Atom) (w : Float) : Facts :=
  if w == 0.0 then
    F
  else
    let w' := ops.or (F.get a) w
    if w' == 0.0 then
      F.erase a
    else
      F.insert a w'

def fromList (ops : Ops) (xs : List (Atom × Float)) : Facts :=
  xs.foldl (fun acc (p : Atom × Float) => addOr ops acc p.1 p.2) empty

def indexByPred (F : Facts) : HashMap String (List (Atom × Float)) :=
  F.fold (init := ({} : HashMap String (List (Atom × Float))))
    (fun acc atom w =>
      if w == 0.0 then
        acc
      else
      let bucket := acc.getD atom.pred []
      acc.insert atom.pred ((atom, w) :: bucket))

def maxDelta (A B : Facts) : Float :=
  let step (acc : Float) (atom : Atom) (wA : Float) : Float :=
    let d := Float.abs (wA - (B.get atom))
    maxF acc d
  let acc1 := A.fold (init := 0.0) step
  let acc2 :=
    B.fold (init := acc1) (fun acc atom wB =>
      if A.contains atom then acc else maxF acc (Float.abs wB))
  acc2

end Facts

private def unifyArg (s : Subst) (p : Sym) (c : String) : Option Subst :=
  match p with
  | .const k => if k = c then some s else none
  | .var v => Subst.bindVar s v c

private def unifyArgs (s : Subst) : List Sym → List String → Option Subst
  | [], [] => some s
  | p :: ps, c :: cs => do
      let s' ← unifyArg s p c
      unifyArgs s' ps cs
  | _, _ => none

def unify (pat : AtomPat) (atom : Atom) (s : Subst := []) : Option Subst := do
  if pat.pred != atom.pred then
    none
  else if pat.arity != atom.arity then
    none
  else
    unifyArgs s pat.args atom.args

private def instantiateArg (s : Subst) : Sym → Option String
  | .const c => some c
  | .var v => Subst.lookup s v

def instantiate (pat : AtomPat) (s : Subst) : Option Atom := do
  let args ← pat.args.mapM (instantiateArg s)
  pure { pred := pat.pred, args := args }

private def extendMatches (ops : Ops) (facts : List (Atom × Float)) (lit : AtomPat)
    (acc : List (Subst × Float)) : List (Subst × Float) :=
  acc.flatMap (fun (m : Subst × Float) =>
    facts.filterMap (fun (fw : Atom × Float) =>
      match unify lit fw.1 m.1 with
      | none => none
      | some s' =>
          let w := ops.and m.2 fw.2
          if w == 0.0 then none else some (s', w)))

private def evalBody
    (ops : Ops)
    (idx : HashMap String (List (Atom × Float)))
    (body : List AtomPat) :
    List (Subst × Float) :=
  body.foldl
    (fun acc lit => extendMatches ops (idx.getD lit.pred []) lit acc)
    [([], 1.0)]

structure RunConfig where
  mode : Mode := .boolean
  tnorm : TNorm := .product
  maxIter : Nat := 50
  eps : Float := 1e-6

structure RunResult where
  facts : Facts
  iters : Nat
  delta : Float
  converged : Bool

private def applyRule (ops : Ops) (idx : HashMap String (List (Atom × Float))) (r : Rule)
    (F : Facts) : Facts :=
  let ms := evalBody ops idx r.body
  ms.foldl
    (fun acc (m : Subst × Float) =>
      match instantiate r.head m.1 with
      | none => acc
      | some headAtom =>
          let rw := r.weight.getD 1.0
          let w := ops.and m.2 rw
          Facts.addOr ops acc headAtom w)
    F

private def step (ops : Ops) (p : Program) (F : Facts) : Facts :=
  let idx := Facts.indexByPred F
  p.rules.foldl (fun acc r => applyRule ops idx r acc) F

/-- One iteration of the immediate consequence operator, *anchored* at a fixed base fact set.

This avoids repeatedly "re-adding" the same base-derived contributions across iterations, which is
harmless in Boolean mode (idempotent `or`) but can inflate weights for fuzzy modes with
non-idempotent t-conorms (e.g. noisy-OR). -/
private def stepFromBase (ops : Ops) (p : Program) (base cur : Facts) : Facts :=
  let idx := Facts.indexByPred cur
  p.rules.foldl (fun acc r => applyRule ops idx r acc) base

def run (cfg : RunConfig) (p : Program) (init : Facts) : RunResult :=
  let ops := Ops.forConfig cfg.mode cfg.tnorm
  let rec loop (fuel : Nat) (iters : Nat) (lastDelta : Float) (cur : Facts) : RunResult :=
    match fuel with
    | 0 =>
        { facts := cur
        , iters := iters
        , delta := lastDelta
        , converged := false
        }
    | Nat.succ fuel =>
        let next := stepFromBase ops p init cur
        let d := Facts.maxDelta cur next
        if d ≤ cfg.eps then
          { facts := next
          , iters := iters + 1
          , delta := d
          , converged := true
          }
        else
          loop fuel (iters + 1) d next
  loop cfg.maxIter 0 0.0 init

end TensorLogic
end Compiler
end HeytingLean
