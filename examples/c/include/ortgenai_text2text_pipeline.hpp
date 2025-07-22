// ORT GenAI mods using common headers
#include <string>

#include <ort_genai.h>
#include "pipelines\text2text_pipeline.hpp"
#include "pipelines\pipeline_factory.hpp"

using namespace onnx::genai;
using namespace onnx::genai::Text2Text;

namespace ort::genai {

class OrtGenAIText2TextGenerator : public Generator {
public:
    OrtGenAIText2TextGenerator(const std::unique_ptr<OgaModel>& model) {
        tokenizer = OgaTokenizer::Create(*model);
        tokenizerStream = OgaTokenizerStream::Create(*tokenizer);
        generator = OgaGenerator::Create(*model, *OgaGeneratorParams::Create(*model));
    }

    void AppendInputTokens(const std::string& tokens) override {
        const std::string prompt = "<|user|>\n" + tokens + "<|end|>\n<|assistant|>";
        auto sequences = OgaSequences::Create();
        tokenizer->Encode(prompt.c_str(), *sequences);
        generator->AppendTokenSequences(*sequences);
    }

    std::string GenerateNextToken() const override {
        generator->GenerateNextToken();
        return tokenizerStream->Decode(generator->GetSequenceData(0)[generator->GetSequenceCount(0) - 1]);
    };

    bool IsDone() const override {
        return generator->IsDone();
    };

private:
    std::unique_ptr<OgaTokenizer> tokenizer;
    std::unique_ptr<OgaTokenizerStream> tokenizerStream;
    std::unique_ptr<OgaGenerator> generator;
};

class OrtGenAIText2TextPipeline : public Pipeline {
public:
    OrtGenAIText2TextPipeline(const std::filesystem::path& models_path) {
        ogaConfig = OgaConfig::Create(models_path.string().c_str());
        ogaModel = OgaModel::Create(*ogaConfig);

        // Populate genConfig and device once mechanism to bubble up config info to user plumbed in
        generator = std::make_optional(OrtGenAIText2TextGenerator(ogaModel));
    };

    GenerationConfig get_generation_config() const override {
        return genConfig;
    };

    void set_generation_config(const GenerationConfig& config) override {
        genConfig = config;
        (*ogaConfig).Overlay(export_genconfig2json().c_str());
    };

    const std::vector<Device> get_supported_devices() const override {
        std::vector<Device> devices(3);
        devices[0].identifier = "CPU";
        devices[0].config["ep"] = "CPU";
        devices[1].identifier = "GPU";
        devices[1].config["ep"] = "cuda";
        devices[1].config["enable_cuda_graph"] = "0";
        devices[2].identifier = "NPU";
        devices[2].config["ep"] = "OpenVINO";
        devices[2].config["device_type"] = "NPU";
        return devices;
    }

    Device get_device() const override {
        return device;
    };

    void set_device(const Device& device) override {
        this->device = device;
        std::string provider;
        std::map<std::string, const char*> providerOptions;
        for (auto config: device.config) {
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
            (*ogaConfig).ClearProviders();
            (*ogaConfig).AppendProvider(provider.c_str());
            for (auto option : providerOptions) {
                (*ogaConfig).SetProviderOption(provider.c_str(), option.first.c_str(), option.second);
            }
        }
    };

    GenerationResult operator()(const GenerationInput& input) override {
        GenerationResult result;

        (*generator).AppendInputTokens(input.text);
        try {
            while (!(*generator).IsDone()) {
                result.text += (*generator).GenerateNextToken();
            }
        }
        catch (const std::exception& e) {
            std::cout << "Session Terminated: " << e.what() << std::endl;
        }

        return result;
    };


private:
    GenerationConfig genConfig;
    Device device;
    std::optional<OrtGenAIText2TextGenerator> generator;

    std::unique_ptr<OgaConfig> ogaConfig;
    std::unique_ptr<OgaModel> ogaModel;

    std::string export_genconfig2json() {
        std::ostringstream oss;
        oss << std::boolalpha << genConfig.sampling_config.do_sample;
        std::string doSample = oss.str();
        oss.str("");
        oss.clear();
        oss << std::boolalpha << (genConfig.beam_search_config.stop_criteria == GenerationConfig::BeamSearchConfig::StopCriteria::EARLY ? true : false);
        std::string stopCriteria = oss.str();

        std::string json = "{\n";
        json += "    \"search\": {\n";
        json += "        \"max_length\": " + std::to_string(genConfig.max_length) + ",\n";
        json += "        \"min_length\": " + std::to_string(genConfig.min_new_tokens) + ",\n";
        json += "        \"do_sample\": " + doSample + ",\n";
        json += "        \"random_seed\": " + std::to_string(genConfig.sampling_config.rng_seed) + ",\n";
        json += "        \"temperature\": " + std::to_string(genConfig.sampling_config.temperature) + ",\n";
        json += "        \"top_k\": " + std::to_string(genConfig.sampling_config.top_k) + ",\n";
        json += "        \"top_p\": " + std::to_string(genConfig.sampling_config.top_p) + ",\n";
        json += "        \"repetition_penalty\": " + std::to_string(genConfig.sampling_config.repetition_penalty) + ",\n";
        json += "        \"num_beams\": " + std::to_string(genConfig.beam_search_config.num_beams) + ",\n";
        json += "        \"diversity_penalty\": " + std::to_string(genConfig.beam_search_config.diversity_penalty) + ",\n";
        json += "        \"length_penalty\": " + std::to_string(genConfig.beam_search_config.length_penalty) + ",\n";
        json += "        \"num_return_sequences\": " + std::to_string(genConfig.beam_search_config.num_return_sequences) + ",\n";
        json += "        \"no_repeat_ngram_size\": " + std::to_string(genConfig.beam_search_config.no_repeat_ngram_size) + ",\n";
        json += "        \"early_stopping\": " + stopCriteria + "\n";
        json += "    },\n";
        json += "    \"model\": {\n";
        json += "        \"eos_token_id\": [";
        for (int64_t tid : genConfig.eos_token_ids) {
        json += std::to_string(tid) + ",";
        }
        if (json.back() == ',') {
        json.pop_back();
        }
        json += "]\n";
        json += "    }\n";
        json += "}\n";
        return json;
    }
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
