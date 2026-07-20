/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: lm engine model info
 */
#ifndef LM_ENGINE_MODEL_INFO_H
#define LM_ENGINE_MODEL_INFO_H

#include <cstdint>
#include <cstddef>

#define LMENGINE_C_API_EXPORT __attribute__((__visibility__("default")))

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t LMEngine_Status;
static const LMEngine_Status LMEngine_SUCCESS = 0;
static const LMEngine_Status LMEngine_FAILURE = 1;
typedef struct LM_Buffer {
    void* data;
    size_t size;
} LM_Buffer;

typedef struct LMEngine_ModelInfo LMEngine_ModelInfo;
LMENGINE_C_API_EXPORT LMEngine_ModelInfo* LMEngine_ModelInfo_Create(void);
LMENGINE_C_API_EXPORT void LMEngine_ModelInfo_Destroy(LMEngine_ModelInfo** modelInfo);

LMENGINE_C_API_EXPORT LMEngine_Status LMEngine_ModelInfo_SetModelBuffer(
    LMEngine_ModelInfo* modelInfo, LM_Buffer* modelBuffer);

LMENGINE_C_API_EXPORT LMEngine_Status LMEngine_ModelInfo_SetModelPath(
    LMEngine_ModelInfo* modelInfo, const char* modelPath);

LMENGINE_C_API_EXPORT LMEngine_Status LMEngine_ModelInfo_SetWeightDir(
    LMEngine_ModelInfo* modelInfo, const char* weightDir);

LMENGINE_C_API_EXPORT LMEngine_Status LMEngine_ModelInfo_SetUserData(
    LMEngine_ModelInfo* modelInfo, void* userData, size_t userDataSize);

#ifdef __cplusplus
}
#endif

#endif // LM_ENGINE_MODEL_INFO_H