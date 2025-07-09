// ORT GenAI mods using common headers
#include <string>
#include <memory>
#include <ort_genai.h>

using namespace onnx::genai;


namespace ort::genai {

class OrtGenAIText2TextPipeline : public Text2TextPipeline {
public:
    OrtGenAIText2TextPipeline(const std::string& modelPath) : modelPath(modelPath) {
        auto config = OgaConfig::CreateGenerationConfig(modelPath.c_str(), &genConfig);
        ogaConfig = config.get();
        auto model = OgaModel::Create(*config);
        auto params = OgaGeneratorParams::Create(*model);
        this->tokenizer = OgaTokenizer::Create(*model);
        this->tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);
        this->generator = OgaGenerator::Create(*model, *params);
    };

    GenerationConfig get_generation_config() override {
        return this->genConfig;
    };

    void set_generation_config(const GenerationConfig& config) override{
        this->genConfig = config;
        ogaConfig->PullFromGenerationConfig(config);
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
                result.text += tokenizer_stream->Decode(new_token);
            }
        } catch (const std::exception& e) {
            std::cout << "Session Terminated: " << e.what() << std::endl;
        }

        return result;
    };

private:
    const std::string modelPath;
    GenerationConfig genConfig;
    OgaConfig* ogaConfig;
    std::unique_ptr<OgaTokenizer> tokenizer;
    std::unique_ptr<OgaTokenizerStream> tokenizer_stream;
    std::unique_ptr<OgaGenerator> generator;
};

std::shared_ptr<onnx::genai::Text2TextPipeline> create_text2text_pipeline(const std::string models_path) {
  return std::make_shared<OrtGenAIText2TextPipeline>(models_path);
}

}  // namespace ort::genai