import HeytingLean.Compiler.TensorLogic.Eval

/-!
Sanity checks for `Mode.f2` (F₂/XOR aggregation).

We only test the non-recursive case: multiple derivations of the same atom cancel mod 2.
-/

namespace HeytingLean.Tests.TensorLogic.F2Sanity

open HeytingLean.Compiler.TensorLogic

private def atomPat (pred : String) (args : List String) : AtomPat :=
  { pred := pred, args := args.map Sym.ofString }

private def prog : Program :=
  { rules :=
      [ { head := atomPat "r" ["X"], body := [atomPat "p" ["X"]] }
      , { head := atomPat "r" ["X"], body := [atomPat "q" ["X"]] }
      ]
  }

private def cfgF2 : RunConfig :=
  { mode := .f2, maxIter := 8, eps := 0.0 }

private def atomR (x : String) : Atom :=
  { pred := "r", args := [x] }

private def runOn (facts : List (Atom × Float)) : RunResult :=
  let ops := Ops.forConfig cfgF2.mode cfgF2.tnorm
  let initFacts := Facts.fromList ops facts
  run cfgF2 prog initFacts

private def resCancel : RunResult :=
  runOn
    [ ({ pred := "p", args := ["a"] }, 1.0)
    , ({ pred := "q", args := ["a"] }, 1.0)
    ]

private def resSingle : RunResult :=
  runOn
    [ ({ pred := "p", args := ["a"] }, 1.0)
    ]

example : resCancel.converged = true := by native_decide
example : resCancel.facts.contains (atomR "a") = false := by native_decide

example : resSingle.converged = true := by native_decide
example : resSingle.facts.contains (atomR "a") = true := by native_decide

end HeytingLean.Tests.TensorLogic.F2Sanity

