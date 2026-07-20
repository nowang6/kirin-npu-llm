/*
 * Tool-calling accuracy evaluation for HarmonyOS LLM (HiAI LLM Engine).
 * Evaluation logic aligned with llm_eval.py; inference aligned with llm_bin.cpp.
 */

#include "cann_llm_engine_context.h"
#include "cann_llm_engine_executor.h"
#include "common/lm_engine_model_info.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* kImEnd = "<|im_end|>";
constexpr const char* kDefaultDataset = "dataset10.jsonl";
constexpr const char* kDefaultToolsFile = "tools_functions.json";

// [\s\S] matches any char including newline (aligned with Python re.DOTALL)
const std::regex kToolCallPattern(R"(<tool_call>\s*([\s\S]*?)\s*</tool_call>)");

LLMEngine_Context* g_context = nullptr;
LLMEngine_Executor* g_executor = nullptr;
std::string g_contextJson;

// ---------------------------------------------------------------------------
// Minimal JSON (dataset / tool_call / arguments comparison)
// ---------------------------------------------------------------------------

enum class JsonType { Null, Bool, Number, String, Array, Object };

struct JsonValue {
    JsonType type = JsonType::Null;
    bool boolVal = false;
    double numberVal = 0;
    bool numberIsInt = true;
    std::string strVal;
    std::vector<JsonValue> arrVal;
    std::map<std::string, JsonValue> objVal;
};

class JsonParser {
public:
    explicit JsonParser(const std::string& s) : src_(s) {}

    bool Parse(JsonValue* out)
    {
        if (out == nullptr) {
            return false;
        }
        SkipWs();
        if (!ParseValue(out)) {
            return false;
        }
        SkipWs();
        return pos_ == src_.size();
    }

private:
    const std::string& src_;
    size_t pos_ = 0;

    char Peek() const
    {
        return pos_ < src_.size() ? src_[pos_] : '\0';
    }

    char Get()
    {
        return pos_ < src_.size() ? src_[pos_++] : '\0';
    }

    void SkipWs()
    {
        while (pos_ < src_.size() && std::isspace(static_cast<unsigned char>(src_[pos_]))) {
            ++pos_;
        }
    }

    bool Match(const char* lit)
    {
        const size_t n = std::strlen(lit);
        if (pos_ + n > src_.size() || src_.compare(pos_, n, lit) != 0) {
            return false;
        }
        pos_ += n;
        return true;
    }

    bool ParseValue(JsonValue* out)
    {
        SkipWs();
        const char c = Peek();
        if (c == '"') {
            return ParseString(out);
        }
        if (c == '{') {
            return ParseObject(out);
        }
        if (c == '[') {
            return ParseArray(out);
        }
        if (c == 't' || c == 'f') {
            return ParseBool(out);
        }
        if (c == 'n') {
            return ParseNull(out);
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            return ParseNumber(out);
        }
        return false;
    }

    bool ParseNull(JsonValue* out)
    {
        if (!Match("null")) {
            return false;
        }
        *out = JsonValue{};
        out->type = JsonType::Null;
        return true;
    }

    bool ParseBool(JsonValue* out)
    {
        if (Match("true")) {
            out->type = JsonType::Bool;
            out->boolVal = true;
            return true;
        }
        if (Match("false")) {
            out->type = JsonType::Bool;
            out->boolVal = false;
            return true;
        }
        return false;
    }

    bool ParseNumber(JsonValue* out)
    {
        const size_t start = pos_;
        if (Peek() == '-') {
            Get();
        }
        if (Peek() == '0') {
            Get();
        } else {
            if (!std::isdigit(static_cast<unsigned char>(Peek()))) {
                return false;
            }
            while (std::isdigit(static_cast<unsigned char>(Peek()))) {
                Get();
            }
        }
        out->numberIsInt = true;
        if (Peek() == '.') {
            out->numberIsInt = false;
            Get();
            if (!std::isdigit(static_cast<unsigned char>(Peek()))) {
                return false;
            }
            while (std::isdigit(static_cast<unsigned char>(Peek()))) {
                Get();
            }
        }
        if (Peek() == 'e' || Peek() == 'E') {
            out->numberIsInt = false;
            Get();
            if (Peek() == '+' || Peek() == '-') {
                Get();
            }
            if (!std::isdigit(static_cast<unsigned char>(Peek()))) {
                return false;
            }
            while (std::isdigit(static_cast<unsigned char>(Peek()))) {
                Get();
            }
        }
        const std::string numStr = src_.substr(start, pos_ - start);
        char* end = nullptr;
        out->numberVal = std::strtod(numStr.c_str(), &end);
        out->type = JsonType::Number;
        return true;
    }

    bool ParseString(JsonValue* out)
    {
        if (Get() != '"') {
            return false;
        }
        std::string s;
        while (pos_ < src_.size()) {
            char c = Get();
            if (c == '"') {
                out->type = JsonType::String;
                out->strVal = std::move(s);
                return true;
            }
            if (c == '\\') {
                if (pos_ >= src_.size()) {
                    return false;
                }
                c = Get();
                switch (c) {
                case '"':
                case '\\':
                case '/':
                    s += c;
                    break;
                case 'b':
                    s += '\b';
                    break;
                case 'f':
                    s += '\f';
                    break;
                case 'n':
                    s += '\n';
                    break;
                case 'r':
                    s += '\r';
                    break;
                case 't':
                    s += '\t';
                    break;
                case 'u': {
                    if (pos_ + 4 > src_.size()) {
                        return false;
                    }
                    unsigned int code = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char h = Get();
                        code <<= 4;
                        if (h >= '0' && h <= '9') {
                            code |= static_cast<unsigned>(h - '0');
                        } else if (h >= 'a' && h <= 'f') {
                            code |= static_cast<unsigned>(h - 'a' + 10);
                        } else if (h >= 'A' && h <= 'F') {
                            code |= static_cast<unsigned>(h - 'A' + 10);
                        } else {
                            return false;
                        }
                    }
                    if (code <= 0x7F) {
                        s += static_cast<char>(code);
                    } else if (code <= 0x7FF) {
                        s += static_cast<char>(0xC0 | ((code >> 6) & 0x1F));
                        s += static_cast<char>(0x80 | (code & 0x3F));
                    } else {
                        s += static_cast<char>(0xE0 | ((code >> 12) & 0x0F));
                        s += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                        s += static_cast<char>(0x80 | (code & 0x3F));
                    }
                    break;
                }
                default:
                    return false;
                }
            } else {
                s += c;
            }
        }
        return false;
    }

    bool ParseArray(JsonValue* out)
    {
        if (Get() != '[') {
            return false;
        }
        SkipWs();
        out->type = JsonType::Array;
        out->arrVal.clear();
        if (Peek() == ']') {
            Get();
            return true;
        }
        for (;;) {
            JsonValue item;
            if (!ParseValue(&item)) {
                return false;
            }
            out->arrVal.push_back(std::move(item));
            SkipWs();
            if (Peek() == ']') {
                Get();
                return true;
            }
            if (Get() != ',') {
                return false;
            }
            SkipWs();
        }
    }

    bool ParseObject(JsonValue* out)
    {
        if (Get() != '{') {
            return false;
        }
        SkipWs();
        out->type = JsonType::Object;
        out->objVal.clear();
        if (Peek() == '}') {
            Get();
            return true;
        }
        for (;;) {
            if (Peek() != '"') {
                return false;
            }
            JsonValue keyVal;
            if (!ParseString(&keyVal)) {
                return false;
            }
            SkipWs();
            if (Get() != ':') {
                return false;
            }
            SkipWs();
            JsonValue val;
            if (!ParseValue(&val)) {
                return false;
            }
            out->objVal[keyVal.strVal] = std::move(val);
            SkipWs();
            if (Peek() == '}') {
                Get();
                return true;
            }
            if (Get() != ',') {
                return false;
            }
            SkipWs();
        }
    }
};

static bool ParseJson(const std::string& s, JsonValue* out)
{
    JsonParser parser(s);
    return parser.Parse(out);
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

static void AppendCanonical(const JsonValue& v, std::string* out);

static void AppendCanonicalObject(const JsonValue& v, std::string* out)
{
    *out += '{';
    bool first = true;
    for (const auto& kv : v.objVal) {
        if (!first) {
            *out += ", ";
        }
        first = false;
        *out += '"';
        *out += JsonEscape(kv.first);
        *out += "\": ";
        AppendCanonical(kv.second, out);
    }
    *out += '}';
}

static void AppendCanonicalArray(const JsonValue& v, std::string* out)
{
    *out += '[';
    for (size_t i = 0; i < v.arrVal.size(); ++i) {
        if (i > 0) {
            *out += ", ";
        }
        AppendCanonical(v.arrVal[i], out);
    }
    *out += ']';
}

static void AppendCanonicalNumber(const JsonValue& v, std::string* out)
{
    if (v.numberIsInt && std::floor(v.numberVal) == v.numberVal &&
        v.numberVal >= static_cast<double>(-9007199254740991LL) &&
        v.numberVal <= static_cast<double>(9007199254740991LL)) {
        *out += std::to_string(static_cast<long long>(v.numberVal));
        return;
    }
    std::ostringstream oss;
    oss << std::setprecision(15) << v.numberVal;
    std::string s = oss.str();
    if (s.find('.') != std::string::npos) {
        while (!s.empty() && s.back() == '0') {
            s.pop_back();
        }
        if (!s.empty() && s.back() == '.') {
            s.pop_back();
        }
    }
    *out += s;
}

static void AppendCanonical(const JsonValue& v, std::string* out)
{
    switch (v.type) {
    case JsonType::Null:
        *out += "null";
        break;
    case JsonType::Bool:
        *out += v.boolVal ? "true" : "false";
        break;
    case JsonType::Number:
        AppendCanonicalNumber(v, out);
        break;
    case JsonType::String:
        *out += '"';
        *out += JsonEscape(v.strVal);
        *out += '"';
        break;
    case JsonType::Array:
        AppendCanonicalArray(v, out);
        break;
    case JsonType::Object:
        AppendCanonicalObject(v, out);
        break;
    }
}

static std::string CanonicalJson(const JsonValue& v)
{
    std::string out;
    AppendCanonical(v, &out);
    return out;
}

static bool JsonObjectGetString(const JsonValue& obj, const char* key, std::string* out)
{
    if (obj.type != JsonType::Object) {
        return false;
    }
    const auto it = obj.objVal.find(key);
    if (it == obj.objVal.end() || it->second.type != JsonType::String) {
        return false;
    }
    *out = it->second.strVal;
    return true;
}

static const JsonValue* JsonObjectGet(const JsonValue& obj, const char* key)
{
    if (obj.type != JsonType::Object) {
        return nullptr;
    }
    const auto it = obj.objVal.find(key);
    return it == obj.objVal.end() ? nullptr : &it->second;
}

// ---------------------------------------------------------------------------
// Tool call types & evaluation (aligned with llm_eval.py)
// ---------------------------------------------------------------------------

struct ToolCall {
    std::string name;
    JsonValue arguments;
};

struct EvalSample {
    std::string extraSystem;
    std::string userContent;
    std::vector<ToolCall> expected;
};

struct MatchStats {
    int nameHits = 0;
    int fullHits = 0;
    int expectedCount = 0;
    int predictedCount = 0;
};

struct SampleResult {
    bool toolCorrect = false;
    bool allCorrect = false;
    int nameHits = 0;
    int fullHits = 0;
    int expectedCount = 0;
    int predictedCount = 0;
    std::vector<ToolCall> predicted;
    std::vector<ToolCall> expected;
    std::string response;
    std::string userContent;
};

struct EvalOptions {
    std::string modelDir;
    std::string datasetFile = kDefaultDataset;
    std::string toolsFile = kDefaultToolsFile;
    std::string jsonOutPath;
    int maxTokensOverride = -1;
    bool quiet = false;
};

struct RunStats {
    uint64_t inputTokens = 0;
    uint64_t outputTokens = 0;
    double prefillMs = 0;
    double decodeMs = 0;
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

static std::string Trim(const std::string& s)
{
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

static std::string BuildSystemPrompt(const std::string& toolsJson)
{
    std::string prompt = R"(# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
)";
    prompt += toolsJson;
    prompt += R"(</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>)";
    return prompt;
}

static std::string FormatEvalPrompt(const std::string& systemContent, const std::string& userContent)
{
    std::string out;
    out += "<|im_start|>system\n";
    out += systemContent;
    out += kImEnd;
    out += "\n";
    out += "<|im_start|>user\n";
    out += userContent;
    out += kImEnd;
    out += "\n";
    out += "<|im_start|>assistant\n";
    return out;
}

static bool ParseArgumentsValue(const JsonValue& raw, JsonValue* argsOut)
{
    if (raw.type == JsonType::Object) {
        *argsOut = raw;
        return true;
    }
    if (raw.type == JsonType::String) {
        JsonValue parsed;
        if (!ParseJson(raw.strVal, &parsed)) {
            *argsOut = JsonValue{};
            argsOut->type = JsonType::Object;
            argsOut->objVal.clear();
            return true;
        }
        if (parsed.type == JsonType::Object) {
            *argsOut = std::move(parsed);
        } else {
            *argsOut = JsonValue{};
            argsOut->type = JsonType::Object;
            argsOut->objVal.clear();
        }
        return true;
    }
    *argsOut = JsonValue{};
    argsOut->type = JsonType::Object;
    argsOut->objVal.clear();
    return true;
}

static bool ToolCallFromJson(const JsonValue& obj, ToolCall* call)
{
    if (obj.type != JsonType::Object) {
        return false;
    }
    std::string name;
    if (!JsonObjectGetString(obj, "name", &name) || name.empty()) {
        return false;
    }
    const JsonValue* argsRaw = JsonObjectGet(obj, "arguments");
    JsonValue args;
    if (argsRaw != nullptr) {
        ParseArgumentsValue(*argsRaw, &args);
    } else {
        args.type = JsonType::Object;
        args.objVal.clear();
    }
    call->name = std::move(name);
    call->arguments = std::move(args);
    return true;
}

static std::vector<ToolCall> ParseToolCalls(const std::string& response)
{
    std::vector<ToolCall> calls;
    const std::sregex_iterator end;
    for (std::sregex_iterator it(response.begin(), response.end(), kToolCallPattern);
         it != end; ++it) {
        const std::string raw = Trim((*it)[1].str());
        if (raw.empty()) {
            continue;
        }
        JsonValue obj;
        if (!ParseJson(raw, &obj)) {
            continue;
        }
        ToolCall call;
        if (ToolCallFromJson(obj, &call)) {
            calls.push_back(std::move(call));
        }
    }
    return calls;
}

static std::vector<ToolCall> NormalizeExpected(const JsonValue& answerArr)
{
    std::vector<ToolCall> calls;
    if (answerArr.type != JsonType::Array) {
        return calls;
    }
    for (const auto& item : answerArr.arrVal) {
        const JsonValue* fn = JsonObjectGet(item, "function");
        if (fn == nullptr) {
            continue;
        }
        std::string name;
        if (!JsonObjectGetString(*fn, "name", &name)) {
            continue;
        }
        const JsonValue* argsRaw = JsonObjectGet(*fn, "arguments");
        JsonValue args;
        if (argsRaw != nullptr) {
            ParseArgumentsValue(*argsRaw, &args);
        } else {
            args.type = JsonType::Object;
            args.objVal.clear();
        }
        ToolCall call;
        call.name = std::move(name);
        call.arguments = std::move(args);
        calls.push_back(std::move(call));
    }
    return calls;
}

static bool ArgsEqual(const JsonValue& a, const JsonValue& b)
{
    return CanonicalJson(a) == CanonicalJson(b);
}

static MatchStats MatchToolCalls(const std::vector<ToolCall>& predicted,
                                 const std::vector<ToolCall>& expected)
{
    MatchStats stats;
    stats.expectedCount = static_cast<int>(expected.size());
    stats.predictedCount = static_cast<int>(predicted.size());

    std::vector<ToolCall> predRemaining = predicted;
    for (const auto& gold : expected) {
        auto it = std::find_if(predRemaining.begin(), predRemaining.end(),
                               [&](const ToolCall& p) { return p.name == gold.name; });
        if (it == predRemaining.end()) {
            continue;
        }
        const ToolCall pred = *it;
        predRemaining.erase(it);
        stats.nameHits += 1;
        if (ArgsEqual(pred.arguments, gold.arguments)) {
            stats.fullHits += 1;
        }
    }
    return stats;
}

static std::string ToolCallsToJson(const std::vector<ToolCall>& calls)
{
    std::string out = "[";
    for (size_t i = 0; i < calls.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += "{\"name\": \"";
        out += JsonEscape(calls[i].name);
        out += "\", \"arguments\": ";
        out += CanonicalJson(calls[i].arguments);
        out += "}";
    }
    out += "]";
    return out;
}

static bool LoadDataset(const std::string& path, std::vector<EvalSample>* samples)
{
    std::ifstream ifs(path);
    if (!ifs) {
        std::cerr << "Failed to open dataset: " << path << "\n";
        return false;
    }
    std::string line;
    size_t lineNo = 0;
    while (std::getline(ifs, line)) {
        ++lineNo;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        line = Trim(line);
        if (line.empty()) {
            continue;
        }
        JsonValue root;
        if (!ParseJson(line, &root)) {
            std::cerr << "JSON parse error at line " << lineNo << " in " << path << "\n";
            return false;
        }
        const JsonValue* messages = JsonObjectGet(root, "messages");
        const JsonValue* answer = JsonObjectGet(root, "answer");
        if (messages == nullptr || answer == nullptr) {
            std::cerr << "Missing messages/answer at line " << lineNo << "\n";
            return false;
        }
        EvalSample sample;
        if (messages->type == JsonType::Array) {
            for (const auto& msg : messages->arrVal) {
                std::string role;
                std::string content;
                if (!JsonObjectGetString(msg, "role", &role)) {
                    continue;
                }
                JsonObjectGetString(msg, "content", &content);
                if (role == "system" && !content.empty()) {
                    sample.extraSystem = content;
                } else if (role == "user") {
                    sample.userContent = content;
                }
            }
        }
        sample.expected = NormalizeExpected(*answer);
        samples->push_back(std::move(sample));
    }
    return !samples->empty();
}

// ---------------------------------------------------------------------------
// HiAI LLM Engine (from llm_bin.cpp)
// ---------------------------------------------------------------------------

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

    const std::string executorJson = ReadFileToString(executorPath);
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

static bool RunGenerateSync(const std::string& formattedPrompt)
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
    LLMEngine_Prompt_Destroy(&prompt);
    if (ret != LLMEngine_SUCCESS) {
        std::cerr << "LLMEngine_Executor_LLM_Generate failed\n";
        return false;
    }
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

static SampleResult EvaluateSample(const EvalSample& sample, const std::string& baseSystemPrompt,
                                   const EvalOptions& opts)
{
    std::string systemContent = baseSystemPrompt;
    if (!sample.extraSystem.empty()) {
        systemContent += "\n\n";
        systemContent += sample.extraSystem;
    }

    const std::string formatted = FormatEvalPrompt(systemContent, sample.userContent);
    SampleResult result;
    result.userContent = sample.userContent;
    result.expected = sample.expected;

    if (!RunGenerateSync(formatted)) {
        std::cerr << "Generation failed\n";
        return result;
    }

    result.response = GetAllGenerationString(g_context);
    result.predicted = ParseToolCalls(result.response);

    const MatchStats match = MatchToolCalls(result.predicted, result.expected);
    result.nameHits = match.nameHits;
    result.fullHits = match.fullHits;
    result.expectedCount = match.expectedCount;
    result.predictedCount = match.predictedCount;

    const bool countMatch = (match.predictedCount == match.expectedCount);
    result.toolCorrect = (match.nameHits == match.expectedCount && countMatch);
    result.allCorrect = (match.fullHits == match.expectedCount && countMatch);

    if (!opts.quiet) {
        std::cout << "\n============================================================\n";
        std::cout << "用户: " << result.userContent << "\n";
        std::cout << "期望工具数: " << result.expectedCount
                  << ", 预测工具数: " << result.predictedCount << "\n";
        std::cout << "工具正确: " << (result.toolCorrect ? "✓" : "✗") << "\n";
        std::cout << "全部正确: " << (result.allCorrect ? "✓" : "✗") << "\n";
        std::cout << "期望: " << ToolCallsToJson(result.expected) << "\n";
        std::cout << "预测: " << ToolCallsToJson(result.predicted) << "\n";
        if (!result.allCorrect) {
            const size_t previewLen = std::min(result.response.size(), size_t{500});
            std::cout << "模型输出片段: " << result.response.substr(0, previewLen) << "\n";
        }
    }

    PrintPerRunStats(g_context);
    return result;
}

static void PrintSummary(const std::vector<SampleResult>& results)
{
    const int n = static_cast<int>(results.size());
    int toolCorrect = 0;
    int allCorrect = 0;
    for (const auto& r : results) {
        if (r.toolCorrect) {
            ++toolCorrect;
        }
        if (r.allCorrect) {
            ++allCorrect;
        }
    }

    std::cout << "\n============================================================\n";
    std::cout << "评测汇总\n";
    std::cout << "============================================================\n";
    std::cout << "样本数: " << n << "\n";
    if (n > 0) {
        std::cout << "工具正确: " << toolCorrect << "/" << n << " = "
                  << std::fixed << std::setprecision(1)
                  << (100.0 * toolCorrect / n) << "%\n";
        std::cout << "全部正确: " << allCorrect << "/" << n << " = "
                  << (100.0 * allCorrect / n) << "%\n";
    }
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

static void WriteJsonBoolField(std::ostream& os, int indent, const char* key, bool value,
                               bool trailingComma)
{
    os << std::string(static_cast<size_t>(indent), ' ') << '"' << key << "\": "
       << (value ? "true" : "false");
    if (trailingComma) {
        os << ',';
    }
    os << '\n';
}

static bool WriteJsonResults(const EvalOptions& opts, const std::vector<SampleResult>& results)
{
    std::ofstream ofs(opts.jsonOutPath);
    if (!ofs) {
        std::cerr << "Failed to open JSON output: " << opts.jsonOutPath << "\n";
        return false;
    }

    int toolCorrect = 0;
    int allCorrect = 0;
    for (const auto& r : results) {
        if (r.toolCorrect) {
            ++toolCorrect;
        }
        if (r.allCorrect) {
            ++allCorrect;
        }
    }

    ofs << std::fixed;
    ofs << "{\n";
    WriteJsonStringField(ofs, 2, "model_dir", opts.modelDir, true);
    WriteJsonStringField(ofs, 2, "dataset_file", opts.datasetFile, true);
    WriteJsonStringField(ofs, 2, "tools_file", opts.toolsFile, true);
    if (opts.maxTokensOverride > 0) {
        ofs << "  \"max_tokens\": " << opts.maxTokensOverride << ",\n";
    }
    ofs << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        const bool last = (i + 1 == results.size());
        ofs << "    {\n";
        ofs << "      \"index\": " << i << ",\n";
        WriteJsonStringField(ofs, 6, "user", r.userContent, true);
        WriteJsonBoolField(ofs, 6, "tool_correct", r.toolCorrect, true);
        WriteJsonBoolField(ofs, 6, "all_correct", r.allCorrect, true);
        ofs << "      \"name_hits\": " << r.nameHits << ",\n";
        ofs << "      \"full_hits\": " << r.fullHits << ",\n";
        ofs << "      \"expected_count\": " << r.expectedCount << ",\n";
        ofs << "      \"predicted_count\": " << r.predictedCount << ",\n";
        WriteJsonStringField(ofs, 6, "expected", ToolCallsToJson(r.expected), true);
        WriteJsonStringField(ofs, 6, "predicted", ToolCallsToJson(r.predicted), true);
        WriteJsonStringField(ofs, 6, "response", r.response, false);
        ofs << "    }";
        if (!last) {
            ofs << ',';
        }
        ofs << '\n';
    }
    ofs << "  ],\n";
    ofs << "  \"summary\": {\n";
    ofs << "    \"sample_count\": " << results.size() << ",\n";
    ofs << "    \"tool_correct\": " << toolCorrect << ",\n";
    ofs << "    \"all_correct\": " << allCorrect << ",\n";
    if (!results.empty()) {
        ofs << "    \"tool_correct_rate\": " << std::setprecision(4)
            << (static_cast<double>(toolCorrect) / results.size()) << ",\n";
        ofs << "    \"all_correct_rate\": " << (static_cast<double>(allCorrect) / results.size())
            << "\n";
    } else {
        ofs << "    \"tool_correct_rate\": null,\n";
        ofs << "    \"all_correct_rate\": null\n";
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
        if (arg == "--tools" || arg == "--tools-file") {
            if (i + 1 >= argc) {
                std::cerr << arg << " requires a file path\n";
                return false;
            }
            opts->toolsFile = argv[++i];
            continue;
        }
        if (arg == "--dataset" || arg == "-d") {
            if (i + 1 >= argc) {
                std::cerr << arg << " requires a file path\n";
                return false;
            }
            opts->datasetFile = argv[++i];
            continue;
        }
        if (arg == "--quiet" || arg == "-q") {
            opts->quiet = true;
            continue;
        }
        if (IsAllDigits(argv[i])) {
            std::istringstream iss(argv[i]);
            iss >> opts->maxTokensOverride;
            continue;
        }
        if (opts->datasetFile == kDefaultDataset) {
            opts->datasetFile = arg;
            continue;
        }
        std::cerr << "Unexpected argument: " << arg << "\n";
        return false;
    }
    return true;
}

static void PrintUsage(const char* prog)
{
    std::cout << "Usage: " << prog
              << " <model_dir> [dataset.jsonl] [max_tokens] [options]\n"
              << "  Tool-calling accuracy evaluation (aligned with llm_eval.py).\n"
              << "  model_dir must contain context.json and executor.json.\n"
              << "  Run from the directory that contains model_dir and dataset files.\n"
              << "\n"
              << "Options:\n"
              << "  -d, --dataset FILE     Dataset jsonl (default: " << kDefaultDataset << ")\n"
              << "  --tools FILE           Tools JSON (default: " << kDefaultToolsFile << ")\n"
              << "  -o, --json-out FILE    Write structured results to JSON\n"
              << "  -q, --quiet            Suppress per-sample details\n"
              << "\n"
              << "Examples:\n"
              << "  " << prog << " qwen25\n"
              << "  " << prog << " qwen25 dataset10.jsonl 2048 -o eval_results.json\n";
}

} // namespace

int main(int argc, const char* argv[])
{
    if (argc < 2) {
        PrintUsage(argv[0]);
        return 0;
    }

    EvalOptions opts;
    if (!ParseEvalArgs(argc, argv, &opts)) {
        PrintUsage(argv[0]);
        return 1;
    }

    opts.modelDir = argv[1];
    const std::string contextPath = opts.modelDir + "/context.json";
    const std::string executorPath = opts.modelDir + "/executor.json";

    if (!FileExists(contextPath) || !FileExists(executorPath)) {
        std::cerr << "Missing " << contextPath << " or " << executorPath << "\n";
        return 1;
    }
    if (!FileExists(opts.datasetFile)) {
        std::cerr << "Missing dataset file: " << opts.datasetFile << "\n";
        return 1;
    }
    if (!FileExists(opts.toolsFile)) {
        std::cerr << "Missing tools file: " << opts.toolsFile << "\n";
        return 1;
    }

    const std::string toolsJson = Trim(ReadFileToString(opts.toolsFile));
    if (toolsJson.empty()) {
        std::cerr << "Empty tools file: " << opts.toolsFile << "\n";
        return 1;
    }
    const std::string baseSystemPrompt = BuildSystemPrompt(toolsJson);

    std::vector<EvalSample> samples;
    if (!LoadDataset(opts.datasetFile, &samples)) {
        return 1;
    }

    std::cout << "model dir is " << opts.modelDir << std::endl;
    std::cout << "加载数据集: " << opts.datasetFile << " (" << samples.size() << " 条)\n";

    if (!LLMEngineInit(contextPath, executorPath, opts.maxTokensOverride)) {
        return 1;
    }

    std::vector<SampleResult> results;
    results.reserve(samples.size());

    for (size_t i = 0; i < samples.size(); ++i) {
        if (!opts.quiet) {
            std::cout << "\n>>> 评测样本 " << (i + 1) << "/" << samples.size() << "\n";
        } else {
            std::cerr << "[" << (i + 1) << "/" << samples.size() << "] evaluating...\n";
        }

        results.push_back(EvaluateSample(samples[i], baseSystemPrompt, opts));

        if (!LLMEngineResetContext(opts.maxTokensOverride > 0 ? opts.maxTokensOverride : -1)) {
            std::cerr << "Reset context between samples failed\n";
            LLMEngineDeInit();
            return 1;
        }
    }

    PrintSummary(results);

    if (!opts.jsonOutPath.empty()) {
        if (!WriteJsonResults(opts, results)) {
            LLMEngineDeInit();
            return 1;
        }
    }

    LLMEngineDeInit();
    return 0;
}
