// ORT GenAI mods using common headers
#include <string>
#include <memory>
#include <ort_genai.h>

using namespace onnx::genai;


class OrtGenAIText2TextPipeline: public Text2TextPipeline {
public:
    OrtGenAIText2TextPipeline(const std::string& modelPath) : modelPath(modelPath) {
        genConfig = OgaConfig::CreateGenerationConfig(modelPath.c_str()).get();
    };

    GenerationConfig* get_generation_config() override {
        return genConfig;
    };

    void set_generation_config(const GenerationConfig& config) override {
    };

    GenerationResult generate(const std::string& tokens) override {
        return GenerationResult();
    };


private:
    const std::string modelPath;
    GenerationConfig* genConfig;
};