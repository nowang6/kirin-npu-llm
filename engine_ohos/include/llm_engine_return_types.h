/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: llm engine return types
 */
#ifndef LLM_ENGINE_RETURN_TYPES_H
#define LLM_ENGINE_RETURN_TYPES_H
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t LLMEngine_Status;
static const LLMEngine_Status LLMEngine_SUCCESS = 0;
static const LLMEngine_Status LLMEngine_FAILURE = 1;
static const LLMEngine_Status LLMEngine_UNINITIALIZED = 2;
static const LLMEngine_Status LLMEngine_INVALID_PARAM = 3;
static const LLMEngine_Status LLMEngine_TIMEOUT = 4;
static const LLMEngine_Status LLMEngine_UNSUPPORTED = 5;
static const LLMEngine_Status LLMEngine_MEMORY_EXCEPTION = 6;
static const LLMEngine_Status LLMEngine_INVALID_API = 7;
static const LLMEngine_Status LLMEngine_INVALID_POINTER = 8;
static const LLMEngine_Status LLMEngine_CALC_EXCEPTION = 9;
static const LLMEngine_Status LLMEngine_FILE_NOT_EXIST = 10;
static const LLMEngine_Status LLMEngine_COMM_EXCEPTION = 11;
static const LLMEngine_Status LLMEngine_DATA_OVERFLOW = 12;
static const LLMEngine_Status LLMEngine_CONNECT_EXCEPTION = 19;

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_RETURN_TYPES_H
