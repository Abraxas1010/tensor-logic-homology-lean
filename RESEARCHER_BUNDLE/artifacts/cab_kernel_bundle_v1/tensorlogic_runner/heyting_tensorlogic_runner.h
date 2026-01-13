#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Return codes for `heyting_tensorlogic_*` entrypoints.
// 0 = OK; nonzero = failure (inspect `heyting_tensorlogic_last_error_message()`).
#define HEYTING_TENSORLOGIC_OK 0
#define HEYTING_TENSORLOGIC_ERR 1

// Run a `*.tensorgraph.json` file and return an output JSON payload.
//
// - `pred_filter`: optional predicate name to restrict emitted facts (pass NULL for no filter).
// - `min_weight`: only emit facts with weight > min_weight.
// - `limit`: max number of emitted facts.
// - `out_json`: on success, set to a newly-allocated C string (free with `heyting_tensorlogic_free_string`).
int heyting_tensorlogic_run_graph_file(
    const char* graph_path,
    const char* pred_filter,
    float min_weight,
    int limit,
    char** out_json);

// Same as `heyting_tensorlogic_run_graph_file`, but accepts raw JSON text.
int heyting_tensorlogic_run_graph_json(
    const char* graph_json,
    const char* pred_filter,
    float min_weight,
    int limit,
    char** out_json);

// Convenience: verify a CAB kernel bundle (TT0 cert(s) + manifest + SHA256SUMS) and then run a
// tensor graph that is listed in `cert/manifest.json` (`tensor_graphs[].path`).
//
// - `bundle_dir`: bundle root, e.g. `dist/cab_kernel_bundle_<ts>_<rev>`.
// - `graph_rel_path`: relative path inside the bundle, e.g. `tensor_graph/reachability.tensorgraph.json`.
//
// This dynamically loads the verifier library from:
// - Linux:   `<bundle_dir>/verifier/libheyting_tt0_verifier.so`
// - macOS:   `<bundle_dir>/verifier/libheyting_tt0_verifier.dylib`
int heyting_tensorlogic_run_verified(
    const char* bundle_dir,
    const char* graph_rel_path,
    const char* pred_filter,
    float min_weight,
    int limit,
    char** out_json);

// Free a string returned via `out_json`.
void heyting_tensorlogic_free_string(char* s);

// Returns a pointer to the last error message for the current thread.
// The pointer remains valid until the next API call on the same thread.
const char* heyting_tensorlogic_last_error_message(void);

// Clears the per-thread error buffer.
void heyting_tensorlogic_clear_error(void);

// Library version string.
const char* heyting_tensorlogic_version(void);

#ifdef __cplusplus
}
#endif
