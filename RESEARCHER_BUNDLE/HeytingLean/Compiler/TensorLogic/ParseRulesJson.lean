import Lean.Data.Json
import HeytingLean.Compiler.TensorLogic.AST

namespace HeytingLean
namespace Compiler
namespace TensorLogic

open Lean
open Lean.Json

private def expectObj (j : Json) : Except String Json := do
  match j.getObj? with
  | .ok _ => pure j
  | .error _ => throw "expected JSON object"

private def getStr (obj : Json) (k : String) : Except String String := do
  match obj.getObjVal? k with
  | .ok v =>
      match v.getStr? with
      | .ok s => pure s
      | .error _ => throw s!"field '{k}' not a string"
  | .error _ => throw s!"missing field '{k}'"

private def getStrList (obj : Json) (k : String) : Except String (List String) := do
  match obj.getObjVal? k with
  | .ok (Json.arr xs) =>
      let mut out : List String := []
      for v in xs.toList do
        match v.getStr? with
        | .ok s => out := s :: out
        | .error _ => throw s!"non-string in '{k}'"
      pure out.reverse
  | .ok _ => throw s!"field '{k}' not an array"
  | .error _ => throw s!"missing field '{k}'"

private def getFloatOpt (obj : Json) (k : String) : Except String (Option Float) := do
  match obj.getObjVal? k with
  | .error _ => pure none
  | .ok v =>
      match v.getNum? with
      | .ok n =>
          -- parse via string to avoid Float/Real confusion
          match String.toInt? n.toString with
          | some i => pure (some (Float.ofInt i))
          | none =>
              match n.toString.splitOn "." with
              | [intPart, fracPart] =>
                  match String.toInt? intPart, String.toNat? fracPart with
                  | some i, some f =>
                      let scale := (10.0 : Float) ^ (Float.ofNat fracPart.length)
                      pure (some (Float.ofInt i + Float.ofNat f / scale))
                  | _, _ => throw s!"bad number in '{k}'"
              | _ => throw s!"bad number in '{k}'"
      | .error _ => throw s!"field '{k}' not a number"

private def parseAtomPatObj (obj : Json) : Except String AtomPat := do
  let pred ← getStr obj "pred"
  let args ← getStrList obj "args"
  pure { pred := pred, args := args.map Sym.ofString }

private def parseRuleObj (obj : Json) : Except String Rule := do
  let headPred ← getStr obj "head"
  let headArgs ← getStrList obj "args"
  let bodyVal ←
    match obj.getObjVal? "body" with
    | .ok v => pure v
    | .error _ => throw "missing field 'body'"
  let bodyArr ←
    match bodyVal.getArr? with
    | .ok a => pure a.toList
    | .error _ => throw "field 'body' not an array"
  let mut body : List AtomPat := []
  for lit in bodyArr do
    let litObj ← expectObj lit
    body := (← parseAtomPatObj litObj) :: body
  let w ← getFloatOpt obj "weight"
  pure
    { head := { pred := headPred, args := headArgs.map Sym.ofString }
    , body := body.reverse
    , weight := w
    }

def parseProgramJson (j : Json) : Except String Program := do
  match j with
  | Json.arr xs =>
      let mut rules : List Rule := []
      for rj in xs.toList do
        let robj ← expectObj rj
        rules := (← parseRuleObj robj) :: rules
      pure { rules := rules.reverse }
  | _ =>
      let obj ← expectObj j
      match obj.getObjVal? "rules" with
      | .error _ => throw "expected top-level array or object with 'rules'"
      | .ok (Json.arr xs) =>
          let mut rules : List Rule := []
          for rj in xs.toList do
            let robj ← expectObj rj
            rules := (← parseRuleObj robj) :: rules
          pure { rules := rules.reverse }
      | .ok _ => throw "field 'rules' not an array"

end TensorLogic
end Compiler
end HeytingLean

