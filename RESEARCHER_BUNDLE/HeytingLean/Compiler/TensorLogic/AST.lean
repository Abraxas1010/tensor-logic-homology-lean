import Std

namespace HeytingLean
namespace Compiler
namespace TensorLogic

/-- A symbol in a rule: either a variable or a constant. -/
inductive Sym where
  | var (name : String)
  | const (name : String)
  deriving Repr, BEq, DecidableEq

namespace Sym

private def stripQuotes (s : String) : Option String :=
  if s.length ≥ 2 then
    if (s.startsWith "'" && s.endsWith "'") || (s.startsWith "\"" && s.endsWith "\"") then
      some ((s.drop 1).dropRight 1)
    else
      none
  else
    none

private def isVarName (s : String) : Bool :=
  if s.isEmpty then
    false
  else
    let c := s.front
    c = '_' || c = '?' || c.isUpper

/-- Heuristic parser for rule arguments:

- quoted strings are constants (quotes stripped);
- leading `?` is a variable marker;
- otherwise, leading uppercase / `_` is treated as a variable (Prolog-style);
- everything else is a constant.

This only applies to *rules*. Facts are always constants. -/
def ofString (raw : String) : Sym :=
  match stripQuotes raw with
  | some s => Sym.const s
  | none =>
      if raw.startsWith "?" then
        Sym.var (raw.drop 1)
      else if isVarName raw then
        Sym.var raw
      else
        Sym.const raw

def vars : Sym → List String
  | .var v => [v]
  | .const _ => []

def consts : Sym → List String
  | .var _ => []
  | .const c => [c]

end Sym

/-- A ground atom (fact instance). -/
structure Atom where
  pred : String
  args : List String
  deriving Repr, BEq, Hashable, DecidableEq

namespace Atom

def arity (a : Atom) : Nat := a.args.length

end Atom

/-- A pattern atom (may contain variables). -/
structure AtomPat where
  pred : String
  args : List Sym
  deriving Repr, BEq, DecidableEq

namespace AtomPat

def arity (a : AtomPat) : Nat := a.args.length

def vars (a : AtomPat) : List String :=
  a.args.flatMap Sym.vars

def consts (a : AtomPat) : List String :=
  a.args.flatMap Sym.consts

end AtomPat

/-- A positive rule: `head :- body₁, ..., bodyₙ`. -/
structure Rule where
  head : AtomPat
  body : List AtomPat
  /-- Optional rule weight (used in fuzzy mode). -/
  weight : Option Float := none
  deriving Repr

namespace Rule

def varsHead (r : Rule) : List String := r.head.vars

def varsBody (r : Rule) : List String := r.body.flatMap AtomPat.vars

end Rule

structure Program where
  rules : List Rule
  deriving Repr

end TensorLogic
end Compiler
end HeytingLean
