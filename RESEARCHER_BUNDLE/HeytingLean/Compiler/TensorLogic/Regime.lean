import HeytingLean.Compiler.TensorLogic.Eval
import HeytingLean.Computational.Tensor.DialAdapter

/-!
# Regime: Dial/Stage → TensorLogic Mode Bridge

This module connects the modal dial infrastructure (`Stage`, `TempRange`)
to the tensor-logic runtime (`Mode`, `TNorm`, `Ops`).

## Key mappings

- `Stage.boolean` → `Mode.boolean`
- `Stage.heyting` → `Mode.heyting` (Gödel t-norm on [0,1])
- `Stage.mv` / `Stage.effect` / `Stage.orthomodular` → `Mode.fuzzy`
- `Stage.beyond` → `Mode.fuzzy` (fallback)

Sharpness (temperature) modulates fuzzy semantics but is ignored in boolean mode.
-/

namespace HeytingLean
namespace Compiler
namespace TensorLogic

open HeytingLean.Computational.Tensor (TempRange linearMap easedMap)

/-- Modal stages from the dial hierarchy. Mirrors `HeytingLean.Logic.Modal.Stage`. -/
inductive Stage where
  | boolean
  | heyting
  | mv
  | effect
  | orthomodular
  | beyond
  deriving Repr, BEq, DecidableEq

namespace Stage

def ofString (s : String) : Except String Stage :=
  match s.toLower with
  | "boolean" | "bool" => pure .boolean
  | "heyting" => pure .heyting
  | "mv" | "lukasiewicz" => pure .mv
  | "effect" => pure .effect
  | "orthomodular" | "oml" => pure .orthomodular
  | "beyond" => pure .beyond
  | _ => throw s!"unknown stage '{s}'"

def toMode : Stage → Mode
  | .boolean => .boolean
  | .heyting => .heyting
  | .mv => .fuzzy
  | .effect => .fuzzy
  | .orthomodular => .fuzzy
  | .beyond => .fuzzy

def defaultTNorm : Stage → TNorm
  | .boolean => .product  -- ignored
  | .heyting => .product  -- Gödel uses min/max, but we use product for rule weights
  | .mv => .lukasiewicz   -- classical MV logic uses Łukasiewicz
  | .effect => .product
  | .orthomodular => .product
  | .beyond => .product

end Stage

/-- A regime combines a modal stage, sharpness (temperature), and t-norm choice. -/
structure Regime where
  stage : Stage := .boolean
  sharpness : Float := 1.0  -- 0 = fully fuzzy, 1 = fully crisp (for applicable modes)
  tnorm : TNorm := .product
  deriving Repr

namespace Regime

def default : Regime := {}

def boolean : Regime := { stage := .boolean }

def heyting (sharpness : Float := 1.0) : Regime :=
  { stage := .heyting, sharpness := sharpness }

def fuzzy (tnorm : TNorm := .product) (sharpness : Float := 1.0) : Regime :=
  { stage := .mv, tnorm := tnorm, sharpness := sharpness }

def mv : Regime := { stage := .mv, tnorm := .lukasiewicz }

/-- Convert a regime to the runtime Mode. -/
def toMode (r : Regime) : Mode := r.stage.toMode

/-- Convert a regime to runtime Ops. -/
def toOps (r : Regime) : Ops := Ops.forConfig r.toMode r.tnorm

/-- Default temperature range for sharpness interpolation. -/
def defaultTempRange : TempRange := { minT := 0.0, maxT := 1.0 }

/-- Map sharpness [0,1] to temperature via linear interpolation. -/
def sharpnessToTemp (r : Regime) (range : TempRange := defaultTempRange) : Float :=
  -- sharpness 0 → minT (fully fuzzy), sharpness 1 → maxT (fully crisp)
  range.minT + r.sharpness * (range.maxT - range.minT)

/-- Build a RunConfig from a regime. -/
def toRunConfig (r : Regime) (maxIter : Nat := 50) (eps : Float := 1e-6) : RunConfig :=
  { mode := r.toMode
  , tnorm := r.tnorm
  , maxIter := maxIter
  , eps := eps
  }

end Regime

end TensorLogic
end Compiler
end HeytingLean
