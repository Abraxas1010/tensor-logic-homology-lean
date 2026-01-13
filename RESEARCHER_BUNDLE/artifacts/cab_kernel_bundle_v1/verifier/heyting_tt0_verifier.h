#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Return codes for `heyting_tt0_*` entrypoints.
// 0 = OK; nonzero = failure (inspect `heyting_tt0_last_error_message()`).
#define HEYTING_TT0_OK 0
#define HEYTING_TT0_ERR 1

// Verify a TT0 certificate given file paths.
int heyting_tt0_verify_cert_files(const char* cab_path, const char* kernel_path, const char* cert_path);

// Verify a TT0 certificate given raw JSON strings.
int heyting_tt0_verify_cert_json(const char* cab_json, const char* kernel_json, const char* cert_json);

// Verify a full CAB kernel bundle directory (TT0 certs + manifest + SHA256SUMS).
// `bundle_dir` should point at the bundle root, e.g. `dist/cab_kernel_bundle_<ts>_<rev>`.
int heyting_tt0_verify_bundle_dir(const char* bundle_dir);

// Returns a pointer to the last error message for the current thread.
// The pointer remains valid until the next API call on the same thread.
const char* heyting_tt0_last_error_message(void);

// Clears the per-thread error buffer.
void heyting_tt0_clear_error(void);

// Library version string.
const char* heyting_tt0_version(void);

#ifdef __cplusplus
}
#endif
