#ifndef CACTUS_H
#define CACTUS_H

#ifndef CACTUS_API
#define CACTUS_API __attribute__((visibility("default")))
#endif

#define CACTUS_TEXT 1
#define CACTUS_SPEECH 2
#define CACTUS_VISION 4

#ifdef __cplusplus
extern "C" {
#endif

CACTUS_API int cactus_load(
    const unsigned char* cact,
    unsigned long long n
);

CACTUS_API int cactus_models(void);

CACTUS_API const char* cactus_last_error(void);

CACTUS_API int cactus_init(
    const char* system_prompt,
    const char* tools_json,
    const char* tool_index_path
);

CACTUS_API int cactus_complete(
    const char* input,
    const float* pcm,
    int samples,
    int max_new_tokens,
    char* out,
    int out_capacity
);

CACTUS_API void cactus_set_audio(
    const char* language,
    const char* keywords,
    int word_timestamps
);

CACTUS_API void cactus_set_threads(int threads);

CACTUS_API void cactus_reset(void);

CACTUS_API int cactus_transcribe(
    const float* pcm,
    int samples,
    const char* language,
    const char* keywords,
    int word_timestamps,
    char* out,
    int out_capacity
);

CACTUS_API int cactus_stream_transcribe_process(
    const float* pcm,
    int samples,
    const char* language,
    const char* keywords,
    char* out,
    int out_capacity
);

CACTUS_API int cactus_stream_transcribe_stop(
    char* out,
    int out_capacity
);

CACTUS_API int cactus_describe(
    const unsigned char* rgb,
    int width,
    int height,
    const char* options,
    const char* language,
    const char* keywords,
    char* out,
    int out_capacity
);

CACTUS_API int cactus_embed(
    const char* input,
    const float* pcm,
    int samples,
    float* out,
    int out_capacity
);

#ifdef __cplusplus
}
#endif

#endif
