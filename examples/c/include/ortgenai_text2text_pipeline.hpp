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
        (*this->ogaConfig).Overlay(export_genconfig2json().c_str());
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

    std::string export_genconfig2json() {
        std::ostringstream oss;
        oss << std::boolalpha << this->genConfig.sampling_config.do_sample;
        std::string doSample = oss.str();
        oss.str("");
        oss.clear();
        oss << std::boolalpha << (this->genConfig.beam_search_config.stop_criteria == GenerationConfig::BeamSearchConfig::StopCriteria::EARLY ? true : false);
        std::string stopCriteria = oss.str();

        std::string json = "{\n";
        json += "    \"search\": {\n";
        json += "        \"max_length\": " + std::to_string(this->genConfig.max_length) + ",\n";
        json += "        \"min_length\": " + std::to_string(this->genConfig.min_new_tokens) + ",\n";
        json += "        \"do_sample\": " + doSample + ",\n";
        json += "        \"random_seed\": " + std::to_string(this->genConfig.sampling_config.rng_seed) + ",\n";
        json += "        \"temperature\": " + std::to_string(this->genConfig.sampling_config.temperature) + ",\n";
        json += "        \"top_k\": " + std::to_string(this->genConfig.sampling_config.top_k) + ",\n";
        json += "        \"top_p\": " + std::to_string(this->genConfig.sampling_config.top_p) + ",\n";
        json += "        \"repetition_penalty\": " + std::to_string(this->genConfig.sampling_config.repetition_penalty) + ",\n";
        json += "        \"num_beams\": " + std::to_string(this->genConfig.beam_search_config.num_beams) + ",\n";
        json += "        \"diversity_penalty\": " + std::to_string(this->genConfig.beam_search_config.diversity_penalty) + ",\n";
        json += "        \"length_penalty\": " + std::to_string(this->genConfig.beam_search_config.length_penalty) + ",\n";
        json += "        \"num_return_sequences\": " + std::to_string(this->genConfig.beam_search_config.num_return_sequences) + ",\n";
        json += "        \"no_repeat_ngram_size\": " + std::to_string(this->genConfig.beam_search_config.no_repeat_ngram_size) + ",\n";
        json += "        \"early_stopping\": " + stopCriteria + "\n";
        json += "    },\n";
        json += "    \"model\": {\n";
        json += "        \"eos_token_id\": [";
        for (int64_t tid : this->genConfig.eos_token_ids) {
          json += std::to_string(tid) + ",";
        }
        if (json.back() == ',') {
          json.pop_back();
        }
        json += "]\n";
        json += "    }\n";
        json += "}\n";
        std::cout << json << std::endl;
        return json;
    }

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
