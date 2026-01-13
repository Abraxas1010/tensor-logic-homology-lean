namespace HeytingLean
namespace CLI

/-- `lake exe foo -- args...` currently forwards a leading `--` into `main`'s `argv`.
Strip it when present so CLIs work both via `lake exe` and when running the built
binary directly. -/
def stripLakeArgs (argv : List String) : List String :=
  match argv with
  | "--" :: rest => rest
  | _ => argv

end CLI
end HeytingLean

