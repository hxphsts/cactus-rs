/*
 * One function from recent LLVM libc++, for systems whose libc++ is older.
 *
 * The Needle engine archive is compiled against a libc++ in which the byte hash behind
 * std::hash<std::string> lives in the library, as std::__1::__hash_memory(const void*, size_t).
 * libc++ 18 and 20 were checked and do not export it, so distributions that package them
 * (Ubuntu 24.04 among them) fail to link the archive with exactly one undefined symbol.
 *
 * This file supplies that symbol. It is weak, so a libc++ that has the real one wins without a
 * duplicate-symbol error. Any deterministic hash is a correct implementation: every caller in the
 * process reaches the same function, and the values never leave the process.
 */

#include <stddef.h>
#include <stdint.h>

/* The Itanium mangling spells size_t as `m` (unsigned long) or `j` (unsigned int). */
#if defined(__LP64__)
#define CACTUS_HASH_MEMORY "_ZNSt3__113__hash_memoryEPKvm"
#else
#define CACTUS_HASH_MEMORY "_ZNSt3__113__hash_memoryEPKvj"
#endif

size_t cactus_libcxx_hash_memory(const void* data, size_t size) __asm__(CACTUS_HASH_MEMORY);

__attribute__((weak, visibility("default")))
size_t cactus_libcxx_hash_memory(const void* data, size_t size) {
    const unsigned char* bytes = (const unsigned char*)data;

    /* FNV-1a, then a finalizer so that short keys still spread across the high bits. */
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    for (size_t i = 0; i < size; i++) {
        hash = (hash ^ bytes[i]) * UINT64_C(0x100000001b3);
    }
    hash ^= hash >> 33;
    hash *= UINT64_C(0xff51afd7ed558ccd);
    hash ^= hash >> 33;

    return (size_t)hash;
}
