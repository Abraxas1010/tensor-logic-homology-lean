-- Wrapper lakefile for repo root
-- Actual build happens in RESEARCHER_BUNDLE/
import Lake
open Lake DSL

package tensorLogicHomology where
  srcDir := "RESEARCHER_BUNDLE"

require std from git "https://github.com/leanprover/std4" @ "main"
