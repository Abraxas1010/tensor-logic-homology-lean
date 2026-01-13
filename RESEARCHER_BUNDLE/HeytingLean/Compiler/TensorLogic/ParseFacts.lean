import Std
import HeytingLean.Compiler.TensorLogic.AST

namespace HeytingLean
namespace Compiler
namespace TensorLogic

open Std

/-- Minimal decimal float parser compatible with TensorLog `.cfacts` weights:

- integers like `7`
- decimals like `3.14`

We avoid scientific notation to keep the parser small and portable. -/
def parseFloatLit (s : String) : Except String Float := do
  match String.toInt? s with
  | some i =>
      pure (Float.ofInt i)
  | none =>
      match s.splitOn "." with
      | [intPart, fracPart] =>
          match String.toInt? intPart, String.toNat? fracPart with
          | some i, some f =>
              let scale := (10.0 : Float) ^ (Float.ofNat fracPart.length)
              pure (Float.ofInt i + Float.ofNat f / scale)
          | _, _ => throw "bad decimal float"
      | _ => throw "unsupported float encoding"

/-- Parse a single TensorLog-style `.cfacts` line.

Format: `pred<TAB>arg1<TAB>arg2[<TAB>weight]`

TensorLog allows a final numeric weight column, but warns that numeric constants
become ambiguous; we adopt the same convention. -/
def parseFactLine (line : String) : Except String (Option (Atom × Float)) := do
  let line := line.trim
  if line.isEmpty then
    return none
  if line.startsWith "#" || line.startsWith "--" || line.startsWith "%" then
    return none
  let cols := line.splitOn "\t"
  let pred := cols.getD 0 ""
  if pred.isEmpty then
    throw "empty predicate in facts line"
  let rawArgs := cols.drop 1
  let (args, w) :=
    match rawArgs.getLast? with
    | some last =>
        if rawArgs.length ≥ 2 then
          match parseFloatLit last with
          | .ok f => (rawArgs.dropLast, f)
          | .error _ => (rawArgs, 1.0)
        else
          (rawArgs, 1.0)
    | none => ([], 1.0)
  return some ({ pred := pred, args := args }, w)

def parseFactsTSV (contents : String) : Except String (List (Atom × Float)) := do
  let mut out : List (Atom × Float) := []
  for line in contents.splitOn "\n" do
    match (← parseFactLine line) with
    | none => continue
    | some f => out := f :: out
  pure out.reverse

end TensorLogic
end Compiler
end HeytingLean

