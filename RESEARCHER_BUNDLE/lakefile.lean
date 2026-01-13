import Lake
open Lake DSL

package HeytingLean where
  leanOptions := #[
    ⟨`autoImplicit, false⟩,
    ⟨`pp.unicode.fun, true⟩
  ]

require std from git "https://github.com/leanprover/std4" @ "main"

@[default_target]
lean_lib HeytingLean where
  roots := #[
    `HeytingLean.Compiler.TensorLogic.AST,
    `HeytingLean.Compiler.TensorLogic.ParseFacts,
    `HeytingLean.Compiler.TensorLogic.ParseRulesJson,
    `HeytingLean.Compiler.TensorLogic.Validate,
    `HeytingLean.Compiler.TensorLogic.Eval,
    `HeytingLean.Compiler.TensorLogic.HomologyEncoding,
    `HeytingLean.Compiler.TensorLogic.HomologyFromFacts,
    `HeytingLean.Computational.Homology.F2Matrix,
    `HeytingLean.Computational.Homology.ChainComplex,
    `HeytingLean.Tests.TensorLogic.AllSanity,
    `HeytingLean.Tests.Homology.AllSanity
  ]

lean_exe tensor_logic_cli where
  root := `HeytingLean.CLI.TensorLogicMain

lean_exe tensor_homology_cli where
  root := `HeytingLean.CLI.TensorHomologyMain

lean_exe homology_cli where
  root := `HeytingLean.CLI.HomologyMain
