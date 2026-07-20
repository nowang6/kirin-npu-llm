/*
 * HarmonyOS command-line LLM inference tool.
 * Engine: HiAI LLM Engine (libhiai_llm_engine.so)
 * CLI semantics aligned with llm_mnn.cpp
 */

#include "cann_llm_engine_context.h"
#include "cann_llm_engine_executor.h"
#include "common/lm_engine_model_info.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

constexpr const char* kImEnd = "<|im_end|>";
constexpr const char* kSystemPrompt =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

bool g_generateAsyncFailed = false;
bool g_waitFlag = false;

LLMEngine_Context* g_context = nullptr;
LLMEngine_Executor* g_executor = nullptr;
std::string g_contextJson;

struct RunStats {
    uint64_t inputTokens = 0;
    uint64_t outputTokens = 0;
    double prefillMs = 0;
    double decodeMs = 0;
};

struct PromptResult {
    size_t index = 0;
    std::string prompt;
    std::string answer;
    RunStats stats;
};

struct EvalOptions {
    std::string modelDir;
    std::string promptFile;
    std::string jsonOutPath;
    int maxTokensOverride = -1;
};

static std::string ReadFileToString(const std::string& path)
{
    std::ifstream ifs(path);
    if (!ifs) {
        return {};
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

static std::string PatchMaxGenTokens(std::string json, int maxTokens)
{
    const std::string key = "\"max_gen_tokens\"";
    auto pos = json.find(key);
    if (pos == std::string::npos) {
        return json;
    }
    pos = json.find(':', pos);
    if (pos == std::string::npos) {
        return json;
    }
    ++pos;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }
    size_t end = pos;
    while (end < json.size() && std::isdigit(static_cast<unsigned char>(json[end]))) {
        ++end;
    }
    json.replace(pos, end - pos, std::to_string(maxTokens));
    return json;
}

static std::string FormatPromptWithChatTemplate(const std::string& userInput)
{
    std::string out;
    out += "<|im_start|>system\n";
    out += kSystemPrompt;
    out += kImEnd;
    out += "\n";
    out += "<|im_start|>user\n";
    out += userInput;
    out += kImEnd;
    out += "\n";
    out += "<|im_start|>assistant\n";
    return out;
}

static std::string FormatPromptWithHistory(
    const std::vector<std::pair<std::string, std::string>>& messages)
{
    std::string out;
    for (const auto& msg : messages) {
        out += "<|im_start|>";
        out += msg.first;
        out += "\n";
        out += msg.second;
        out += kImEnd;
        out += "\n";
    }
    out += "<|im_start|>assistant\n";
    return out;
}

static bool CollectStats(const LLMEngine_Context* ctx, RunStats* stats)
{
    if (ctx == nullptr || stats == nullptr) {
        return false;
    }
    LLMEngine_Status ret;
    ret = LLMEngine_Context_GetInputTokenCount(ctx, &stats->inputTokens);
    if (ret != LLMEngine_SUCCESS) {
        return false;
    }
    ret = LLMEngine_Context_GetOutputTokenCount(ctx, &stats->outputTokens);
    if (ret != LLMEngine_SUCCESS) {
        return false;
    }
    ret = LLMEngine_Context_GetPrefillTimeMs(ctx, &stats->prefillMs);
    if (ret != LLMEngine_SUCCESS) {
        return false;
    }
    ret = LLMEngine_Context_GetDecodeTimeMs(ctx, &stats->decodeMs);
    return ret == LLMEngine_SUCCESS;
}

static std::string GetAllGenerationString(const LLMEngine_Context* ctx)
{
    uint32_t len = 0;
    if (LLMEngine_Context_GetAllGenerationLen(ctx, &len) != LLMEngine_SUCCESS) {
        return {};
    }
    std::string result(len, '\0');
    if (LLMEngine_Context_GetAllGeneration(ctx, result.data(), len) != LLMEngine_SUCCESS) {
        return {};
    }
    return result;
}

static bool LLMEngineCreateContextFromJson(const std::string& jsonStr)
{
    if (g_context != nullptr) {
        LLMEngine_Context_Destroy(&g_context);
        g_context = nullptr;
    }
    g_context = LLMEngine_Context_CreateFromContextJson(jsonStr.c_str());
    if (g_context == nullptr) {
        std::cerr << "Failed to create context from JSON\n";
        return false;
    }
    return true;
}

static bool LLMEngineCreateExecutorFromJson(const std::string& jsonStr)
{
    if (g_executor != nullptr) {
        LLMEngine_Executor_Deinit(g_executor);
        LLMEngine_Executor_Destroy(&g_executor);
        g_executor = nullptr;
    }
    g_executor = LLMEngine_Executor_CreateFromJson(jsonStr.c_str());
    if (g_executor == nullptr) {
        std::cerr << "Failed to create executor from JSON\n";
        return false;
    }
    return true;
}

static bool LLMEngineInit(const std::string& contextPath, const std::string& executorPath,
                          int maxTokensOverride)
{
    g_contextJson = ReadFileToString(contextPath);
    if (g_contextJson.empty()) {
        std::cerr << "Failed to read context json: " << contextPath << "\n";
        return false;
    }
    if (maxTokensOverride > 0) {
        g_contextJson = PatchMaxGenTokens(g_contextJson, maxTokensOverride);
    }

    std::string executorJson = ReadFileToString(executorPath);
    if (executorJson.empty()) {
        std::cerr << "Failed to read executor json: " << executorPath << "\n";
        return false;
    }

    if (!LLMEngineCreateContextFromJson(g_contextJson)) {
        return false;
    }
    if (!LLMEngineCreateExecutorFromJson(executorJson)) {
        return false;
    }
    std::cout << "LLM Engine init done.\n";
    return true;
}

static void LLMEngineDeInit()
{
    if (g_executor != nullptr) {
        LLMEngine_Executor_Deinit(g_executor);
        LLMEngine_Executor_Destroy(&g_executor);
        g_executor = nullptr;
    }
    if (g_context != nullptr) {
        LLMEngine_Context_Destroy(&g_context);
        g_context = nullptr;
    }
}

static bool LLMEngineResetContext(int maxTokensOverride)
{
    std::string json = g_contextJson;
    if (maxTokensOverride > 0) {
        json = PatchMaxGenTokens(json, maxTokensOverride);
    }
    return LLMEngineCreateContextFromJson(json);
}

static bool RunGenerateSync(const std::string& formattedPrompt, bool printOutput)
{
    LLMEngine_Prompt* prompt = LLMEngine_Prompt_Create();
    if (prompt == nullptr) {
        std::cerr << "LLMEngine_Prompt_Create failed\n";
        return false;
    }

    LLMEngine_Status ret = LLMEngine_Prompt_SetText(prompt, formattedPrompt.c_str());
    if (ret != LLMEngine_SUCCESS) {
        std::cerr << "LLMEngine_Prompt_SetText failed\n";
        LLMEngine_Prompt_Destroy(&prompt);
        return false;
    }

    ret = LLMEngine_Executor_LLM_Generate(g_executor, g_context, prompt);
    if (ret != LLMEngine_SUCCESS) {
        std::cerr << "LLMEngine_Executor_LLM_Generate failed\n";
        LLMEngine_Prompt_Destroy(&prompt);
        return false;
    }

    if (printOutput) {
        std::string generation = GetAllGenerationString(g_context);
        std::cout << generation << std::flush;
    }

    LLMEngine_Prompt_Destroy(&prompt);
    return true;
}

static bool RunGenerateAsync(const std::string& formattedPrompt)
{
    auto onAllDone = [](const LLMEngine_Context* ctx) {
        g_waitFlag = false;
        (void)ctx;
    };

    auto onSomeToken = [](const LLMEngine_Context* ctx) {
        uint32_t len = 0;
        if (LLMEngine_Context_GetOneGenerationLen(ctx, &len) != LLMEngine_SUCCESS) {
            return;
        }
        std::string chunk(len, '\0');
        if (LLMEngine_Context_GetOneGeneration(ctx, chunk.data(), len) != LLMEngine_SUCCESS) {
            return;
        }
        std::cout << chunk << std::flush;
    };

    auto onFailed = [](const LLMEngine_Context* ctx) {
        g_waitFlag = false;
        g_generateAsyncFailed = true;
        if (ctx != nullptr) {
            std::cerr << "\nGenerate async failed, status: "
                      << LLMEngine_Context_GetGenerateStatus(ctx) << "\n";
        }
    };

    LLMEngine_Status ret =
        LLMEngine_Context_SetOnAllTokensGenerateDoneFunc(g_context, onAllDone);
    if (ret != LLMEngine_SUCCESS) {
        std::cerr << "SetOnAllTokensGenerateDoneFunc failed\n";
        return false;
    }
    ret = LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc(g_context, onSomeToken);
    if (ret != LLMEngine_SUCCESS) {
        std::cerr << "SetOnSomeTokenGenerateDoneFunc failed\n";
        return false;
    }
    ret = LLMEngine_Context_SetOnGenerateAsyncFailed(g_context, onFailed);
    if (ret != LLMEngine_SUCCESS) {
        std::cerr << "SetOnGenerateAsyncFailed failed\n";
        return false;
    }

    LLMEngine_Prompt* prompt = LLMEngine_Prompt_Create();
    if (prompt == nullptr) {
        std::cerr << "LLMEngine_Prompt_Create failed\n";
        return false;
    }

    ret = LLMEngine_Prompt_SetText(prompt, formattedPrompt.c_str());
    if (ret != LLMEngine_SUCCESS) {
        std::cerr << "LLMEngine_Prompt_SetText failed\n";
        LLMEngine_Prompt_Destroy(&prompt);
        return false;
    }

    g_generateAsyncFailed = false;
    g_waitFlag = true;

    ret = LLMEngine_Executor_LLM_GenerateAsync(g_executor, g_context, prompt);
    if (ret != LLMEngine_SUCCESS) {
        std::cerr << "LLMEngine_Executor_LLM_GenerateAsync failed\n";
        LLMEngine_Prompt_Destroy(&prompt);
        return false;
    }

    while (g_waitFlag) {
        usleep(100000);
    }

    LLMEngine_Prompt_Destroy(&prompt);

    if (g_generateAsyncFailed) {
        return false;
    }
    std::cout << std::endl;
    return true;
}

static void PrintPerRunStats(const LLMEngine_Context* ctx)
{
    RunStats stats;
    if (!CollectStats(ctx, &stats)) {
        return;
    }
    std::cerr << "  input tokens: " << stats.inputTokens
              << ", output tokens: " << stats.outputTokens
              << ", prefill: " << stats.prefillMs << " ms"
              << ", decode: " << stats.decodeMs << " ms";
    if (stats.outputTokens > 0) {
        std::cerr << ", decode/token: " << (stats.decodeMs / stats.outputTokens) << " ms";
    }
    std::cerr << std::endl;
}

static void PrintBenchmarkSummary(uint64_t promptTokens, uint64_t decodeTokens,
                                double prefillS, double decodeS)
{
    std::cout << "\n#################################\n";
    std::cout << "prompt tokens num = " << promptTokens << "\n";
    std::cout << "decode tokens num = " << decodeTokens << "\n";
    std::cout << "prefill time = " << prefillS << " s\n";
    std::cout << " decode time = " << decodeS << " s\n";
    if (prefillS > 0) {
        std::cout << "prefill speed = " << (promptTokens / prefillS) << " tok/s\n";
    }
    if (decodeS > 0) {
        std::cout << " decode speed = " << (decodeTokens / decodeS) << " tok/s\n";
    }
    std::cout << "##################################\n";
}

static std::string JsonEscape(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20) {
                char buf[7];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += static_cast<char>(c);
            }
            break;
        }
    }
    return out;
}

static void WriteJsonStringField(std::ostream& os, int indent, const char* key,
                                 const std::string& value, bool trailingComma)
{
    os << std::string(static_cast<size_t>(indent), ' ') << '"' << key << "\": \""
       << JsonEscape(value) << '"';
    if (trailingComma) {
        os << ',';
    }
    os << '\n';
}

static void WriteJsonNumberField(std::ostream& os, int indent, const char* key, double value,
                                 bool trailingComma)
{
    os << std::string(static_cast<size_t>(indent), ' ') << '"' << key << "\": "
       << std::setprecision(10) << value;
    if (trailingComma) {
        os << ',';
    }
    os << '\n';
}

static bool WriteJsonResults(const EvalOptions& opts, const std::vector<PromptResult>& results,
                             uint64_t totalInputTokens, uint64_t totalOutputTokens,
                             double totalPrefillMs, double totalDecodeMs)
{
    std::ofstream ofs(opts.jsonOutPath);
    if (!ofs) {
        std::cerr << "Failed to open JSON output: " << opts.jsonOutPath << "\n";
        return false;
    }

    const double prefillS = totalPrefillMs / 1000.0;
    const double decodeS = totalDecodeMs / 1000.0;

    ofs << std::fixed;
    ofs << "{\n";
    WriteJsonStringField(ofs, 2, "model_dir", opts.modelDir, true);
    WriteJsonStringField(ofs, 2, "prompt_file", opts.promptFile, true);
    if (opts.maxTokensOverride > 0) {
        ofs << "  \"max_tokens\": " << opts.maxTokensOverride << ",\n";
    }
    ofs << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        const bool last = (i + 1 == results.size());
        const double decodePerToken =
            r.stats.outputTokens > 0 ? (r.stats.decodeMs / r.stats.outputTokens) : 0.0;

        ofs << "    {\n";
        ofs << "      \"index\": " << r.index << ",\n";
        WriteJsonStringField(ofs, 6, "prompt", r.prompt, true);
        WriteJsonStringField(ofs, 6, "answer", r.answer, true);
        ofs << "      \"input_tokens\": " << r.stats.inputTokens << ",\n";
        ofs << "      \"output_tokens\": " << r.stats.outputTokens << ",\n";
        WriteJsonNumberField(ofs, 6, "prefill_ms", r.stats.prefillMs, true);
        WriteJsonNumberField(ofs, 6, "decode_ms", r.stats.decodeMs, true);
        WriteJsonNumberField(ofs, 6, "decode_ms_per_token", decodePerToken, false);
        ofs << "    }";
        if (!last) {
            ofs << ',';
        }
        ofs << '\n';
    }
    ofs << "  ],\n";
    ofs << "  \"summary\": {\n";
    ofs << "    \"total_input_tokens\": " << totalInputTokens << ",\n";
    ofs << "    \"total_output_tokens\": " << totalOutputTokens << ",\n";
    WriteJsonNumberField(ofs, 4, "total_prefill_ms", totalPrefillMs, true);
    WriteJsonNumberField(ofs, 4, "total_decode_ms", totalDecodeMs, true);
    if (prefillS > 0) {
        WriteJsonNumberField(ofs, 4, "prefill_speed_tok_s", totalInputTokens / prefillS, true);
    } else {
        ofs << "    \"prefill_speed_tok_s\": null,\n";
    }
    if (decodeS > 0) {
        WriteJsonNumberField(ofs, 4, "decode_speed_tok_s", totalOutputTokens / decodeS, false);
    } else {
        ofs << "    \"decode_speed_tok_s\": null\n";
    }
    ofs << "  }\n";
    ofs << "}\n";

    if (!ofs) {
        std::cerr << "Failed to write JSON output: " << opts.jsonOutPath << "\n";
        return false;
    }
    std::cerr << "Wrote " << results.size() << " result(s) to " << opts.jsonOutPath << "\n";
    return true;
}

static void chat(int maxTokensOverride)
{
    std::vector<std::pair<std::string, std::string>> messages;
    messages.emplace_back("system", kSystemPrompt);

    while (true) {
        std::cout << "\nUser: ";
        std::string userStr;
        if (!std::getline(std::cin, userStr)) {
            break;
        }
        if (userStr == "/exit") {
            return;
        }
        if (userStr == "/reset") {
            if (!LLMEngineResetContext(maxTokensOverride)) {
                std::cerr << "Reset context failed\n";
                return;
            }
            messages.clear();
            messages.emplace_back("system", kSystemPrompt);
            std::cout << "\nA: reset done." << std::endl;
            continue;
        }

        messages.emplace_back("user", userStr);
        const std::string formatted = FormatPromptWithHistory(messages);

        std::cout << "\nA: " << std::flush;
        if (!RunGenerateAsync(formatted)) {
            std::cerr << "Generation failed\n";
            messages.pop_back();
            continue;
        }

        const std::string assistantStr = GetAllGenerationString(g_context);
        messages.emplace_back("assistant", assistantStr);
        PrintPerRunStats(g_context);
    }
}

static int benchmark(const EvalOptions& opts, const std::vector<std::string>& prompts)
{
    const bool jsonOut = !opts.jsonOutPath.empty();

    uint64_t promptLen = 0;
    uint64_t decodeLen = 0;
    double prefillTimeMs = 0;
    double decodeTimeMs = 0;
    std::vector<PromptResult> results;
    size_t resultIndex = 0;

    for (const auto& prompt : prompts) {
        if (!prompt.empty() && prompt[0] == '#') {
            continue;
        }

        const std::string formatted = FormatPromptWithChatTemplate(prompt);
        if (!jsonOut) {
            std::cout << "\n--- prompt ---\n" << prompt << "\n--- answer ---\n";
        } else {
            std::cerr << "[" << (resultIndex + 1) << "] generating...\n";
        }

        if (!RunGenerateSync(formatted, !jsonOut)) {
            std::cerr << "Generation failed for prompt\n";
            return 1;
        }
        if (!jsonOut) {
            std::cout << std::endl;
        }

        RunStats stats;
        if (!CollectStats(g_context, &stats)) {
            std::cerr << "Failed to collect stats\n";
            return 1;
        }

        if (jsonOut) {
            PromptResult row;
            row.index = resultIndex++;
            row.prompt = prompt;
            row.answer = GetAllGenerationString(g_context);
            row.stats = stats;
            results.push_back(std::move(row));
        }

        promptLen += stats.inputTokens;
        decodeLen += stats.outputTokens;
        prefillTimeMs += stats.prefillMs;
        decodeTimeMs += stats.decodeMs;
        PrintPerRunStats(g_context);

        if (!LLMEngineResetContext(opts.maxTokensOverride > 0 ? opts.maxTokensOverride : -1)) {
            std::cerr << "Reset context between prompts failed\n";
            return 1;
        }
    }

    if (jsonOut) {
        if (!WriteJsonResults(opts, results, promptLen, decodeLen, prefillTimeMs, decodeTimeMs)) {
            return 1;
        }
    } else {
        PrintBenchmarkSummary(promptLen, decodeLen, prefillTimeMs / 1000.0,
                              decodeTimeMs / 1000.0);
    }
    return 0;
}

static int eval(const EvalOptions& opts)
{
    std::cout << "prompt file is " << opts.promptFile << std::endl;
    if (!opts.jsonOutPath.empty()) {
        std::cout << "json output is " << opts.jsonOutPath << std::endl;
    }

    std::ifstream promptFs(opts.promptFile);
    if (!promptFs) {
        std::cerr << "Failed to open prompt file: " << opts.promptFile << "\n";
        return 1;
    }

    std::vector<std::string> prompts;
    std::string line;
    while (std::getline(promptFs, line)) {
        if (line.empty()) {
            continue;
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        prompts.push_back(line);
    }
    promptFs.close();

    if (prompts.empty()) {
        std::cerr << "No prompts in file\n";
        return 1;
    }

    if (opts.maxTokensOverride > 0 && !LLMEngineResetContext(opts.maxTokensOverride)) {
        std::cerr << "Failed to apply max_tokens to context\n";
        return 1;
    }

    return benchmark(opts, prompts);
}

static bool FileExists(const std::string& path)
{
    std::ifstream f(path);
    return f.good();
}

static bool IsAllDigits(const char* s)
{
    if (s == nullptr || *s == '\0') {
        return false;
    }
    for (const char* p = s; *p != '\0'; ++p) {
        if (!std::isdigit(static_cast<unsigned char>(*p))) {
            return false;
        }
    }
    return true;
}

static bool ParseEvalArgs(int argc, const char* argv[], EvalOptions* opts)
{
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json-out" || arg == "-o") {
            if (i + 1 >= argc) {
                std::cerr << arg << " requires a file path\n";
                return false;
            }
            opts->jsonOutPath = argv[++i];
            continue;
        }
        if (IsAllDigits(argv[i])) {
            std::istringstream iss(argv[i]);
            iss >> opts->maxTokensOverride;
            continue;
        }
        if (opts->promptFile.empty()) {
            opts->promptFile = arg;
            continue;
        }
        std::cerr << "Unexpected argument: " << arg << "\n";
        return false;
    }

    if (opts->promptFile.empty()) {
        std::cerr << "Missing prompt file for batch mode\n";
        return false;
    }
    return true;
}

static void PrintUsage(const char* prog)
{
    std::cout << "Usage: " << prog << " <model_dir> [prompt.txt] [max_tokens] [--json-out FILE]\n"
              << "  model_dir must contain context.json and executor.json\n"
              << "  Run from the directory that contains model_dir (e.g. repo root).\n"
              << "  Interactive chat if prompt.txt is omitted.\n"
              << "  Batch mode: --json-out (or -o) writes structured results to a JSON file.\n"
              << "  Commands in chat: /exit, /reset\n";
}

} // namespace

int main(int argc, const char* argv[])
{
    if (argc < 2) {
        PrintUsage(argv[0]);
        return 0;
    }

    const std::string modelDir = argv[1];
    const std::string contextPath = modelDir + "/context.json";
    const std::string executorPath = modelDir + "/executor.json";

    if (!FileExists(contextPath) || !FileExists(executorPath)) {
        std::cerr << "Missing " << contextPath << " or " << executorPath << "\n";
        return 1;
    }

    EvalOptions evalOpts;
    evalOpts.modelDir = modelDir;

    int maxTokensOverride = -1;
    if (argc >= 3) {
        if (!ParseEvalArgs(argc, argv, &evalOpts)) {
            PrintUsage(argv[0]);
            return 1;
        }
        maxTokensOverride = evalOpts.maxTokensOverride;
    }

    std::cout << "model dir is " << modelDir << std::endl;

    if (!LLMEngineInit(contextPath, executorPath, maxTokensOverride)) {
        return 1;
    }

    int ret = 0;
    if (argc < 3) {
        chat(maxTokensOverride);
    } else {
        ret = eval(evalOpts);
    }

    LLMEngineDeInit();
    return ret;
}
