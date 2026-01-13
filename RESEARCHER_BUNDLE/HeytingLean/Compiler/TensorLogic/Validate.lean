import Std
import HeytingLean.Compiler.TensorLogic.AST

namespace HeytingLean
namespace Compiler
namespace TensorLogic

open Std

private def contains (xs : List String) (x : String) : Bool :=
  xs.any (fun y => y = x)

private def checkArity (m : HashMap String Nat) (pred : String) (arity : Nat) :
    Except String (HashMap String Nat) := do
  match m.get? pred with
  | none => pure (m.insert pred arity)
  | some n =>
      if n = arity then
        pure m
      else
        throw s!"arity mismatch for predicate '{pred}': saw {n} and {arity}"

def Rule.isSafe (r : Rule) : Bool :=
  let bodyVars := r.varsBody
  r.varsHead.all (fun v => contains bodyVars v)

def validateRule (r : Rule) : Except String Rule := do
  if r.head.pred.isEmpty then
    throw "rule head predicate is empty"
  if r.body.any (fun a => a.pred.isEmpty) then
    throw "rule body contains empty predicate"
  if !(Rule.isSafe r) then
    throw s!"unsafe rule: head vars {r.varsHead} not all bound in body vars {r.varsBody}"
  pure r

def validateProgram (p : Program) : Except String Program := do
  let rules ← p.rules.mapM validateRule
  let _arities ←
    rules.foldlM (init := ({} : HashMap String Nat)) (fun m r => do
      let m ← checkArity m r.head.pred r.head.arity
      r.body.foldlM (init := m) (fun m a => checkArity m a.pred a.arity))
  pure { p with rules := rules }

end TensorLogic
end Compiler
end HeytingLean
