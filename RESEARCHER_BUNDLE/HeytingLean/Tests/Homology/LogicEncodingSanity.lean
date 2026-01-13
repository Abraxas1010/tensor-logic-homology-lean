import HeytingLean.Compiler.TensorLogic.HomologyEncoding
import HeytingLean.Compiler.TensorLogic.Eval
import HeytingLean.Computational.Homology.ChainComplex

namespace HeytingLean
namespace Tests
namespace Homology
namespace LogicEncodingSanity

open HeytingLean.Compiler.TensorLogic
open HeytingLean.Computational.Homology

private def mkDemo : Except String ChainComplexF2 := do
  let dims : Array Nat := #[4, 6, 4]
  let d1Cols : Array (Array Nat) :=
    #[ #[0, 1]  -- 01
     , #[0, 2]  -- 02
     , #[0, 3]  -- 03
     , #[1, 2]  -- 12
     , #[1, 3]  -- 13
     , #[2, 3]  -- 23
     ]
  let d2Cols : Array (Array Nat) :=
    #[ #[0, 1, 3]  -- 012
     , #[0, 2, 4]  -- 013
     , #[1, 2, 5]  -- 023
     , #[3, 4, 5]  -- 123
     ]
  let d1 ← F2Matrix.ofColOnes dims[0]! dims[1]! d1Cols
  let d2 ← F2Matrix.ofColOnes dims[1]! dims[2]! d2Cols
  let C : ChainComplexF2 := { dims := dims, boundaries := #[d1, d2] }
  C.validateShapes
  pure C

private def containsConnected (F : Facts) (a b : String) : Bool :=
  let atom : Atom := { pred := "connected", args := [a, b] }
  (F.get atom) == 1.0

private def allConnected : Bool :=
  match mkDemo with
  | .error _ => false
  | .ok C =>
      let (prog, facts) := ChainComplexF2.toLogicProgram C
      let cfg : RunConfig := { mode := .boolean, maxIter := 20, eps := 0.0 }
      let ops := Ops.forConfig cfg.mode cfg.tnorm
      let init := Facts.fromList ops (facts.map (fun (a, _) => (a, 1.0)))
      let res := run cfg prog init
      let vs := ["v0", "v1", "v2", "v3"]
      vs.all (fun a => vs.all (fun b => containsConnected res.facts a b))

example : allConnected = true := by
  native_decide

end LogicEncodingSanity
end Homology
end Tests
end HeytingLean
