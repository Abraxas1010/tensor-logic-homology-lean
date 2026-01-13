import Lean
import Lean.Data.Json

import HeytingLean.CLI.Args
import HeytingLean.Compiler.TensorLogic.ExportGraph
import HeytingLean.Compiler.TensorLogic.ParseFacts
import HeytingLean.Compiler.TensorLogic.ParseRulesJson
import HeytingLean.Compiler.TensorLogic.Validate

namespace HeytingLean.CLI.TensorLogicExportGraphMain

open Lean
open Lean.Json
open HeytingLean.Compiler.TensorLogic

private def usage : String :=
  String.intercalate "\n"
    [ "tensor_logic_export_graph"
    , ""
    , "Export a TensorLogic program + facts as a backend-neutral tensor graph IR (JSON)."
    , ""
    , "Usage:"
    , "  tensor_logic_export_graph --rules RULES.json --facts FACTS.tsv [--facts FACTS2.tsv ...] --out graph.json"
    , ""
    , "Options:"
    , "  --mode boolean|f2|fuzzy|heyting (default: fuzzy)"
    , "  --tnorm product|lukasiewicz  (default: product; fuzzy-only)"
    , "  --sharpness F                (0.0..1.0; default: 1.0)"
    , "  --max-iter N                 (default: 50)"
    , "  --eps E                      (default: 0.000001; decimal only)"
    , "  --out PATH                   (write JSON to PATH; relative to repo root unless absolute)"
    , "  --help"
    ]

structure Args where
  rulesPath : Option System.FilePath := none
  factsPaths : List System.FilePath := []
  mode : Mode := .fuzzy
  tnorm : TNorm := .product
  sharpness : Float := 1.0
  maxIter : Nat := 50
  eps : Float := 1e-6
  outPath : Option System.FilePath := none
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
  | "--out" :: fp :: rest =>
      parseArgs rest { acc with outPath := some (System.FilePath.mk fp) }
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

private def die (msg : String) : IO UInt32 := do
  IO.eprintln msg
  pure 1

private def resolveOutPath (p : System.FilePath) : IO System.FilePath := do
  if p.isAbsolute then
    return p
  let cwd ← IO.currentDir
  let base :=
    if cwd.fileName == some "lean" then
      cwd.parent.getD cwd
    else
      cwd
  return base / p

private def writeFile (path : System.FilePath) (contents : String) : IO Unit := do
  let some parent := path.parent
    | throw <| IO.userError s!"could not compute parent directory for {path}"
  IO.FS.createDirAll parent
  IO.FS.writeFile path contents

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
      return (0 : UInt32)

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

    let graphJson ←
      match ExportGraph.tensorGraphJson cfg args.sharpness prog facts with
      | .ok j => pure j
      | .error e => return (← die s!"tensor graph export error: {e}")

    let payload := toString graphJson
    match args.outPath with
    | none =>
        IO.println payload
        pure (0 : UInt32)
    | some p =>
        let out ← resolveOutPath p
        writeFile out (payload ++ "\n")
        IO.println s!"[tensor_logic_export_graph] wrote {out}"
        pure (0 : UInt32)
  catch e =>
    die s!"tensor_logic_export_graph: FAILED: {e}"

end HeytingLean.CLI.TensorLogicExportGraphMain

open HeytingLean.CLI.TensorLogicExportGraphMain in
def main (args : List String) : IO UInt32 :=
  HeytingLean.CLI.TensorLogicExportGraphMain.main args
