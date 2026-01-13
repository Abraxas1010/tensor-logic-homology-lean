import HeytingLean.Compiler.TensorLogic.ParseFacts
import HeytingLean.Compiler.TensorLogic.HomologyFromFacts

namespace HeytingLean
namespace Tests
namespace Homology
namespace SimplicialFactsSanity

open HeytingLean.Compiler.TensorLogic
open HeytingLean.Computational.Homology

private def mk (tsv : String) : Except String ChainComplexF2 := do
  let facts ← parseFactsTSV tsv
  HomologyFromFacts.chainComplexF2 facts

private def bettisOf (tsv : String) : Array Nat :=
  match mk tsv with
  | .error _ => #[]
  | .ok C =>
      match C.bettis with
      | .ok bs => bs
      | .error _ => #[]

private def sphere2Facts : String :=
  String.intercalate "\n"
    [ "vertex\tv0"
    , "vertex\tv1"
    , "vertex\tv2"
    , "vertex\tv3"
    , "edge\tv0\tv1"
    , "edge\tv0\tv2"
    , "edge\tv0\tv3"
    , "edge\tv1\tv2"
    , "edge\tv1\tv3"
    , "edge\tv2\tv3"
    , "face_edge\tf012\tv0\tv1"
    , "face_edge\tf012\tv0\tv2"
    , "face_edge\tf012\tv1\tv2"
    , "face_edge\tf013\tv0\tv1"
    , "face_edge\tf013\tv0\tv3"
    , "face_edge\tf013\tv1\tv3"
    , "face_edge\tf023\tv0\tv2"
    , "face_edge\tf023\tv0\tv3"
    , "face_edge\tf023\tv2\tv3"
    , "face_edge\tf123\tv1\tv2"
    , "face_edge\tf123\tv1\tv3"
    , "face_edge\tf123\tv2\tv3"
    ]

private def circleFacts : String :=
  String.intercalate "\n"
    [ "vertex\tv0"
    , "vertex\tv1"
    , "vertex\tv2"
    , "edge\tv0\tv1"
    , "edge\tv1\tv2"
    , "edge\tv2\tv0"
    ]

example : bettisOf sphere2Facts = #[1, 0, 1] := by
  native_decide

example : bettisOf circleFacts = #[1, 1] := by
  native_decide

end SimplicialFactsSanity
end Homology
end Tests
end HeytingLean

