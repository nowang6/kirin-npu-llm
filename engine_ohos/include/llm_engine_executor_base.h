/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: llm engine executor
 */
#ifndef LLM_ENGINE_EXECUTOR_BASE_H
#define LLM_ENGINE_EXECUTOR_BASE_H

#include "llm_engine_c_api_export.h"
#include "llm_engine_return_types.h"
#include "llm_engine_context_base.h"
#include "common/lm_engine_model_info.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LLMEngine_Executor LLMEngine_Executor;

LLMEngine_Executor* LLMEngine_Executor_CreateFromJson(const char* jsonStr);

typedef struct LLMEngine_Prompt LLMEngine_Prompt;

LLMEngine_Prompt* LLMEngine_Prompt_Create(void);

void LLMEngine_Prompt_Destroy(LLMEngine_Prompt** prompt);

LLMEngine_Status LLMEngine_Executor_LLM_Generate(
    LLMEngine_Executor* executor, LLMEngine_Context* ctx, LLMEngine_Prompt* prompt);

LLMEngine_Status LLMEngine_Executor_LLM_GenerateAsync(
    LLMEngine_Executor* executor, LLMEngine_Context* ctx, LLMEngine_Prompt* prompt);

LLMEngine_Status LLMEngine_Prompt_SetText(
    LLMEngine_Prompt* prompt, const char* text);

LLMEngine_Status LLMEngine_Prompt_SetTokenIds(
    LLMEngine_Prompt* prompt, uint32_t* inputTokens, uint32_t tokenNum);

LLMEngine_Status LLMEngine_Executor_Deinit(LLMEngine_Executor* executor);

void LLMEngine_Executor_Destroy(LLMEngine_Executor** executor);


#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_EXECUTOR_H
