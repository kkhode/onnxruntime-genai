// ORT GenAI mods using common headers
#include <string>
#include <memory>
#include <ort_genai.h>
#include "pipelines\text2text_pipeline.hpp"
#include "pipelines\pipeline_factory.hpp"

using namespace onnx::genai;


namespace ort::genai {

class OrtGenAIText2TextPipeline : public Text2TextPipeline {
public:
    OrtGenAIText2TextPipeline(const std::filesystem::path& models_path):
    modelPath(models_path) {
        this->ogaConfig = OgaConfig::Create(modelPath.string().c_str());
        (*this->ogaConfig).PushToGenerationConfig(&genConfig);

        auto model = OgaModel::Create(*(this->ogaConfig));
        auto params = OgaGeneratorParams::Create(*model);
        this->tokenizer = OgaTokenizer::Create(*model);
        this->tokenizerStream = OgaTokenizerStream::Create(*tokenizer);
        this->generator = OgaGenerator::Create(*model, *params);
    };

    GenerationConfig get_generation_config() const override {
        return this->genConfig;
    };

    void set_generation_config(const GenerationConfig& config) override {
        this->genConfig = config;
        (*this->ogaConfig).PullFromGenerationConfig(config);
    };

    std::map<std::string, std::any> get_device_config() const override {
        return this->deviceConfig;
    };

    void set_device_config(std::map<std::string, std::any> device_config) override {
        this->deviceConfig = device_config;
        std::string provider;
        std::map<std::string, const char*> providerOptions;
        for (auto config: device_config) {
            if (config.first == "ep") {
                provider = std::any_cast<const char*>(config.second);
            }
            else {
                providerOptions[config.first] = std::any_cast<const char*>(config.second);
            }
        }
        if (provider == "follow_config") {
            return;
        }
        else {
            (*this->ogaConfig).ClearProviders();
            (*this->ogaConfig).AppendProvider(provider.c_str());
            for (auto option : providerOptions) {
                (*this->ogaConfig).SetProviderOption(provider.c_str(), option.first.c_str(), option.second);
            }
        }
    };

    GenerationResult operator()(const std::string& input) override {
        GenerationResult result;
        auto sequences = OgaSequences::Create();
        const std::string prompt = "<|user|>\n" + input + "<|end|>\n<|assistant|>";
        tokenizer->Encode(prompt.c_str(), *sequences);
        generator->AppendTokenSequences(*sequences);

        try {
            while (!generator->IsDone()) {
                generator->GenerateNextToken();
                const auto num_tokens = generator->GetSequenceCount(0);
                const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
                result.text += tokenizerStream->Decode(new_token);
            }
        } catch (const std::exception& e) {
            std::cout << "Session Terminated: " << e.what() << std::endl;
        }

        return result;
    };

private:
    const std::filesystem::path& modelPath;
    GenerationConfig genConfig;
    std::map<std::string, std::any> deviceConfig;

    std::unique_ptr<OgaConfig> ogaConfig;
    std::unique_ptr<OgaTokenizer> tokenizer;
    std::unique_ptr<OgaTokenizerStream> tokenizerStream;
    std::unique_ptr<OgaGenerator> generator;
};

}  // namespace ort::genai

namespace {
    struct Registrar {
        Registrar() {
            PipelineFactory::GetInstance().Register(
                "onnxruntime_genai",
                [](const std::filesystem::path& models_path) {
                    return std::make_shared<ort::genai::OrtGenAIText2TextPipeline>(models_path);
                }
            );
        }
    };
    static Registrar registrar;
}
