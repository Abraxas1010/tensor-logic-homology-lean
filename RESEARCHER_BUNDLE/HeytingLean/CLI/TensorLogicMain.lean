import Lean
import Lean.Data.Json

import HeytingLean.CLI.Args
import HeytingLean.Compiler.TensorLogic.ParseFacts
import HeytingLean.Compiler.TensorLogic.ParseRulesJson
import HeytingLean.Compiler.TensorLogic.Validate
import HeytingLean.Compiler.TensorLogic.Eval

namespace HeytingLean.CLI.TensorLogicMain

open Lean
open Lean.Json
open HeytingLean.Compiler.TensorLogic

private def usage : String :=
  String.intercalate "\n"
    [ "tensor_logic_cli"
    , ""
    , "Usage:"
    , "  tensor_logic_cli --rules RULES.json --facts FACTS.tsv [--facts FACTS2.tsv ...]"
    , ""
    , "Options:"
    , "  --mode boolean|f2|fuzzy|heyting (default: boolean)"
    , "  --tnorm product|lukasiewicz  (default: product; fuzzy-only)"
    , "  --sharpness F                (0.0..1.0; default: 1.0)"
    , "  --max-iter N                 (default: 50)"
    , "  --eps E                      (default: 1e-6; fuzzy-only)"
    , "  --pred P                     (filter output to predicate P)"
    , "  --help"
    ]

structure Args where
  rulesPath : Option System.FilePath := none
  factsPaths : List System.FilePath := []
  mode : Mode := .boolean
  tnorm : TNorm := .product
  sharpness : Float := 1.0
  maxIter : Nat := 50
  eps : Float := 1e-6
  predFilter : Option String := none
  help : Bool := false
  deriving Repr

private def parseMode (s : String) : Except String Mode :=
  match s with
  | "boolean" => pure .boolean
  | "f2" | "xor" | "mod2" => pure .f2
  | "fuzzy" => pure .fuzzy
  | "heyting" => pure .heyting
  | _ => throw s!"unknown mode '{s}'"

private def parseTNorm (s : String) : Except String TNorm :=
  match s with
  | "product" => pure .product
  | "lukasiewicz" => pure .lukasiewicz
  | _ => throw s!"unknown t-norm '{s}'"

private partial def parseArgs (argv : List String) (acc : Args := {}) : Except String Args := do
  match argv with
  | [] => pure acc
  | "--help" :: rest => parseArgs rest { acc with help := true }
  | "--rules" :: fp :: rest =>
      parseArgs rest { acc with rulesPath := some (System.FilePath.mk fp) }
  | "--facts" :: fp :: rest =>
      parseArgs rest { acc with factsPaths := acc.factsPaths ++ [System.FilePath.mk fp] }
  | "--pred" :: p :: rest => parseArgs rest { acc with predFilter := some p }
  | "--mode" :: m :: rest =>
      let m ← parseMode m
      parseArgs rest { acc with mode := m }
  | "--tnorm" :: t :: rest =>
      let t ← parseTNorm t
      parseArgs rest { acc with tnorm := t }
  | "--sharpness" :: s :: rest =>
      match parseFloatLit s with
      | .ok v => parseArgs rest { acc with sharpness := v }
      | .error msg => throw s!"bad --sharpness '{s}': {msg}"
  | "--max-iter" :: n :: rest =>
      match String.toNat? n with
      | some k => parseArgs rest { acc with maxIter := k }
      | none => throw s!"bad --max-iter '{n}'"
  | "--eps" :: e :: rest =>
      match parseFloatLit e with
      | .ok v => parseArgs rest { acc with eps := v }
      | .error msg => throw s!"bad --eps '{e}': {msg}"
  | x :: _ => throw s!"unknown argument '{x}' (try --help)"

private def jsonNumOrStr (x : Float) : Json :=
  match JsonNumber.fromFloat? x with
  | .inr n => Json.num n
  | .inl s => Json.str s

private def atomToJson (a : Atom) (w : Float) : Json :=
  Json.mkObj
    [ ("pred", Json.str a.pred)
    , ("args", Json.arr (a.args.map Json.str |>.toArray))
    , ("weight", jsonNumOrStr w)
    ]

private def die (msg : String) : IO UInt32 := do
  IO.eprintln msg
  pure 1

private def demoAtomPat (pred : String) (args : List String) : AtomPat :=
  { pred := pred, args := args.map Sym.ofString }

private def demoProgram : Program :=
  { rules :=
      [ { head := demoAtomPat "reachable" ["X", "Y"]
        , body := [demoAtomPat "edge" ["X", "Y"]]
        }
      , { head := demoAtomPat "reachable" ["X", "Z"]
        , body := [demoAtomPat "edge" ["X", "Y"], demoAtomPat "reachable" ["Y", "Z"]]
        }
      ]
  }

private def demoFacts : List (Atom × Float) :=
  [ ({ pred := "edge", args := ["a", "b"] }, 1.0)
  , ({ pred := "edge", args := ["b", "c"] }, 1.0)
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

    let useDemo := args.rulesPath.isNone || args.factsPaths.isEmpty

    let prog ←
      if useDemo then
        pure demoProgram
      else
        let rulesPath := args.rulesPath.get!
        let rulesRaw ← IO.FS.readFile rulesPath
        let rulesJson ←
          match Json.parse rulesRaw with
          | .ok j => pure j
          | .error e => return (← die s!"rules JSON parse error: {e}")
        match parseProgramJson rulesJson >>= validateProgram with
        | .ok p => pure p
        | .error e => return (← die s!"rules validation error: {e}")

    let facts ←
      if useDemo then
        pure demoFacts
      else
        let mut acc : List (Atom × Float) := []
        for fp in args.factsPaths do
          let txt ← IO.FS.readFile fp
          match parseFactsTSV txt with
          | .ok xs => acc := acc ++ xs
          | .error e => return (← die s!"facts parse error ({fp}): {e}")
        pure acc

    let cfg : RunConfig :=
      { mode := args.mode, tnorm := args.tnorm, maxIter := args.maxIter, eps := args.eps }
    let ops := Ops.forConfig cfg.mode cfg.tnorm
    let initFacts :=
      match cfg.mode with
      | .boolean => Facts.fromList ops (facts.map (fun (a, _) => (a, 1.0)))
      | .f2 => Facts.fromList ops (facts.map (fun (a, w) => (a, if w == 0.0 then 0.0 else 1.0)))
      | .fuzzy => Facts.fromList ops facts
      | .heyting => Facts.fromList ops facts  -- Heyting uses weights like fuzzy

    let res := run cfg prog initFacts

    let mut out : Array Json := #[]
    for (a, w) in res.facts.toList do
      if args.predFilter.map (fun p => p == a.pred) |>.getD true then
        out := out.push (atomToJson a w)

    let modeStr :=
      match cfg.mode with
      | .boolean => "boolean"
      | .f2 => "f2"
      | .fuzzy => "fuzzy"
      | .heyting => "heyting"

    let tnormStr :=
      match cfg.tnorm with
      | .product => "product"
      | .lukasiewicz => "lukasiewicz"

    let metaObj :=
      Json.mkObj
        [ ("mode", Json.str modeStr)
        , ("tnorm", Json.str tnormStr)
        , ("iters", Json.num (JsonNumber.fromNat res.iters))
        , ("delta", jsonNumOrStr res.delta)
        , ("converged", Json.bool res.converged)
        ]

    let payload :=
      Json.mkObj
        [ ("meta", metaObj)
        , ("facts", Json.arr out)
        ]

    IO.println (toString payload)
    if res.converged then
      pure 0
    else
      -- For fuzzy programs, non-convergence is treated as a runtime failure.
      pure 2
  catch e =>
    die s!"tensor_logic_cli: FAILED: {e}"

end HeytingLean.CLI.TensorLogicMain

open HeytingLean.CLI.TensorLogicMain in
def main (args : List String) : IO UInt32 :=
  HeytingLean.CLI.TensorLogicMain.main args
