import HeytingLean.Compiler.TensorLogic.Validate
import HeytingLean.Compiler.TensorLogic.Eval

/-!
Compile-time sanity checks for the TensorLogic scaffold (positive Datalog closure).
-/

namespace HeytingLean.Tests.TensorLogic

open HeytingLean.Compiler.TensorLogic

private def atomPat (pred : String) (args : List String) : AtomPat :=
  { pred := pred, args := args.map Sym.ofString }

private def ruleReach1 : Rule :=
  { head := atomPat "reachable" ["X", "Y"]
  , body := [atomPat "edge" ["X", "Y"]]
  }

private def ruleReach2 : Rule :=
  { head := atomPat "reachable" ["X", "Z"]
  , body := [atomPat "edge" ["X", "Y"], atomPat "reachable" ["Y", "Z"]]
  }

private def prog : Program :=
  { rules := [ruleReach1, ruleReach2] }

private def initEdges : List (Atom × Float) :=
  [ ({ pred := "edge", args := ["a", "b"] }, 1.0)
  , ({ pred := "edge", args := ["b", "c"] }, 1.0)
  ]

private def cfgBool : RunConfig :=
  { mode := .boolean, maxIter := 16 }

private def resBool : RunResult :=
  let ops := Ops.forConfig cfgBool.mode cfgBool.tnorm
  let initFacts := Facts.fromList ops initEdges
  run cfgBool prog initFacts

private def ac : Atom := { pred := "reachable", args := ["a", "c"] }

example : resBool.converged = true := by native_decide
example : resBool.facts.contains ac = true := by native_decide

end HeytingLean.Tests.TensorLogic

