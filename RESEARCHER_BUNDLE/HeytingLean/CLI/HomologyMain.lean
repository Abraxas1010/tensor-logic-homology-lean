import Lean
import Lean.Data.Json

import HeytingLean.CLI.Args
import HeytingLean.Computational.Homology.ChainComplex

namespace HeytingLean.CLI.HomologyMain

open Lean
open Lean.Json
open HeytingLean.Computational.Homology

private def usage : String :=
  String.intercalate "\n"
    [ "homology_cli"
    , ""
    , "Usage:"
    , "  homology_cli --input CHAIN.json [--k K]"
    , ""
    , "Options:"
    , "  --input FILE.json   (required unless demo mode)"
    , "  --k K               (compute only β_k; default: all)"
    , "  --help"
    ]

structure Args where
  input : Option System.FilePath := none
  k : Option Nat := none
  help : Bool := false
  deriving Repr

private partial def parseArgs (argv : List String) (acc : Args := {}) : Except String Args := do
  match argv with
  | [] => pure acc
  | "--help" :: rest => parseArgs rest { acc with help := true }
  | "--input" :: fp :: rest => parseArgs rest { acc with input := some (System.FilePath.mk fp) }
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

    let useDemo := args.input.isNone

    let C ←
      if useDemo then
        match mkDemo with
        | .ok c => pure c
        | .error e => return (← die s!"demo chain complex error: {e}")
      else
        let fp := args.input.get!
        let raw ← IO.FS.readFile fp
        let j ←
          match Json.parse raw with
          | .ok j => pure j
          | .error e => return (← die s!"JSON parse error: {e}")
        match ChainComplexF2.parseJson j with
        | .ok c => pure c
        | .error e => return (← die s!"chain complex parse error: {e}")

    match C.checkD2 with
    | .error e =>
        IO.eprintln s!"homology_cli: FAILED: {e}"
        return 2
    | .ok _ => pure ()

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
    die s!"homology_cli: FAILED: {e}"

end HeytingLean.CLI.HomologyMain

open HeytingLean.CLI.HomologyMain in
def main (args : List String) : IO UInt32 :=
  HeytingLean.CLI.HomologyMain.main args

