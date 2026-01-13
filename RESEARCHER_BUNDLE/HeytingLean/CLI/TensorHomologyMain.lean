import Lean
import Lean.Data.Json

import HeytingLean.CLI.Args
import HeytingLean.Compiler.TensorLogic.ParseFacts
import HeytingLean.Compiler.TensorLogic.HomologyFromFacts

namespace HeytingLean.CLI.TensorHomologyMain

open Lean
open Lean.Json
open HeytingLean.Compiler.TensorLogic
open HeytingLean.Computational.Homology

private def usage : String :=
  String.intercalate "\n"
    [ "tensor_homology_cli"
    , ""
    , "Usage:"
    , "  tensor_homology_cli --facts COMPLEX.tsv [--facts MORE.tsv ...] [--k K]"
    , ""
    , "Input facts predicates:"
    , "  vertex(V)"
    , "  edge(V1,V2)"
    , "  face_edge(F,V1,V2)   # one entry per boundary edge of triangle F"
    , ""
    , "Options:"
    , "  --facts FILE.tsv     (required unless demo mode)"
    , "  --k K                (compute only β_k; default: all)"
    , "  --help"
    ]

structure Args where
  factsPaths : List System.FilePath := []
  k : Option Nat := none
  help : Bool := false
  deriving Repr

private partial def parseArgs (argv : List String) (acc : Args := {}) : Except String Args := do
  match argv with
  | [] => pure acc
  | "--help" :: rest => parseArgs rest { acc with help := true }
  | "--facts" :: fp :: rest =>
      parseArgs rest { acc with factsPaths := acc.factsPaths ++ [System.FilePath.mk fp] }
  | "--k" :: ks :: rest =>
      match String.toNat? ks with
      | some k => parseArgs rest { acc with k := some k }
      | none => throw s!"bad --k '{ks}'"
  | x :: _ => throw s!"unknown argument '{x}' (try --help)"

private def jsonArrNat (xs : Array Nat) : Json :=
  Json.arr (xs.map (fun n => Json.num (JsonNumber.fromNat n)))

private def die (msg : String) : IO UInt32 := do
  IO.eprintln msg
  pure 1

private def demoFacts : String :=
  String.intercalate "\n"
    [ "# S² = boundary of tetrahedron (4 vertices, 6 edges, 4 faces)"
    , "vertex\tv0"
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

def main (argvRaw : List String) : IO UInt32 := do
  let argv := HeytingLean.CLI.stripLakeArgs argvRaw
  try
    let args ←
      match parseArgs argv with
      | .ok a => pure a
      | .error e => return (← die s!"{e}\n\n{usage}")

    if args.help then
      IO.println usage
      return 0

    let useDemo := args.factsPaths.isEmpty

    let facts ←
      if useDemo then
        match parseFactsTSV demoFacts with
        | .ok xs => pure xs
        | .error e => return (← die s!"demo facts parse error: {e}")
      else
        let mut acc : List (Atom × Float) := []
        for fp in args.factsPaths do
          let txt ← IO.FS.readFile fp
          match parseFactsTSV txt with
          | .ok xs => acc := acc ++ xs
          | .error e => return (← die s!"facts parse error ({fp}): {e}")
        pure acc

    let C ←
      match HomologyFromFacts.chainComplexF2 facts with
      | .ok c => pure c
      | .error e => return (← die s!"chain complex build error: {e}")

    let ranks := C.boundaryRanks
    let bettis ←
      match args.k with
      | none =>
          match C.bettis with
          | .ok bs => pure bs
          | .error e => return (← die s!"betti computation error: {e}")
      | some k =>
          match C.bettiAt k with
          | .ok b => pure #[b]
          | .error e => return (← die s!"betti computation error: {e}")

    let metaObj :=
      Json.mkObj
        [ ("demo", Json.bool useDemo)
        , ("maxDim", Json.num (JsonNumber.fromNat C.maxDim))
        , ("d2_ok", Json.bool true)
        ]

    let payload :=
      Json.mkObj
        [ ("meta", metaObj)
        , ("dims", jsonArrNat C.dims)
        , ("boundary_ranks", jsonArrNat ranks)
        , ("betti", jsonArrNat bettis)
        ]

    IO.println (toString payload)
    pure 0
  catch e =>
    die s!"tensor_homology_cli: FAILED: {e}"

end HeytingLean.CLI.TensorHomologyMain

open HeytingLean.CLI.TensorHomologyMain in
def main (args : List String) : IO UInt32 :=
  HeytingLean.CLI.TensorHomologyMain.main args

