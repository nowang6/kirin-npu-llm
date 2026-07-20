/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: llm engine context base
 */
#ifndef LLM_ENGINE_CONTEXT_BASE_H
#define LLM_ENGINE_CONTEXT_BASE_H

#include "llm_engine_c_api_export.h"
#include "llm_engine_return_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LLMEngine_Context LLMEngine_Context;

LLMEngine_Context* LLMEngine_Context_CreateFromContextJson(const char* jsonStr);

void LLMEngine_Context_Destroy(LLMEngine_Context** ctx);

LLMEngine_Status LLMEngine_Context_GetTotalTimeMs(
    const LLMEngine_Context* ctx, double* ms);
LLMEngine_Status LLMEngine_Context_GetPrefillTimeMs(
    const LLMEngine_Context* ctx, double* ms);
LLMEngine_Status LLMEngine_Context_GetDecodeTimeMs(
    const LLMEngine_Context* ctx, double* ms);
LLMEngine_Status LLMEngine_Context_GetInputTokenCount(
    const LLMEngine_Context* ctx, uint64_t* count);
LLMEngine_Status LLMEngine_Context_GetOutputTokenCount(
    const LLMEngine_Context* ctx, uint64_t* count);
LLMEngine_Status LLMEngine_Context_GetDecodeNum(const LLMEngine_Context* ctx,
    uint64_t* num);

LLMEngine_Status LLMEngine_Context_GetGenerateStatus(
    const LLMEngine_Context* ctx);

typedef void(*callbackFuncType)(const LLMEngine_Context*);
LLMEngine_Status LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc(
    LLMEngine_Context* ctx, callbackFuncType func);
LLMEngine_Status LLMEngine_Context_SetOnAllTokensGenerateDoneFunc(
    LLMEngine_Context* ctx, callbackFuncType func);
LLMEngine_Status LLMEngine_Context_SetOnGenerateAsyncFailed(
    LLMEngine_Context* ctx, callbackFuncType func);

LLMEngine_Status LLMEngine_Context_GetOneGenerationLen(
    const LLMEngine_Context* ctx, uint32_t* len);
LLMEngine_Status LLMEngine_Context_GetOneGeneration(
    const LLMEngine_Context* ctx, char* generation, uint32_t len);

LLMEngine_Status LLMEngine_Context_GetAllGenerationLen(
    const LLMEngine_Context* ctx, uint32_t* len);
LLMEngine_Status LLMEngine_Context_GetAllGeneration(
    const LLMEngine_Context* ctx, char* generation, uint32_t len);

LLMEngine_Status LLMEngine_Context_GetAllTokenGenerationLen(
    const LLMEngine_Context* ctx, uint32_t* len);

LLMEngine_Status LLMEngine_Context_GetAllTokenGeneration(
    const LLMEngine_Context* ctx, int32_t* genToken, uint32_t genTokenLen);

LLMEngine_Status LLMEngine_Context_GetOneTokenGeneration(
    const LLMEngine_Context* ctx, int32_t* genToken);
    
#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_CONTEXT_BASE_H