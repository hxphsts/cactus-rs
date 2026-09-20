#ifndef NEEDLE_H
#define NEEDLE_H

#ifndef NEEDLE_API
#define NEEDLE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* One process-global, non-thread-safe model. Negative returns indicate failure. */
NEEDLE_API int needle_init(
    const char* system_prompt,
    const char* tools_json,
    const char* tool_index_path
);

NEEDLE_API int needle_complete(
    const char* input,
    int max_new_tokens,
    char* out,
    int out_capacity
);

/* A null output returns the model's embedding dimension without computing. */
NEEDLE_API int needle_embed(
    const char* input,
    float* out,
    int out_capacity
);

NEEDLE_API void needle_reset(void);

NEEDLE_API int needle_load(
    const unsigned char* cact,
    unsigned long long n
);

#ifdef __cplusplus
}
#endif
#endif
