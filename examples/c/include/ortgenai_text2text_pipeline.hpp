// ORT GenAI mods using common headers
#include <string>
#include <memory>
#include <ort_genai.h>

using namespace onnx::genai;

static TerminateSession catch_terminate;

void signalHandlerWrapper(int signum) {
  catch_terminate.signalHandler(signum);
}

class OrtGenAIText2TextPipeline: public Text2TextPipeline {
public:
    OrtGenAIText2TextPipeline(const std::string& modelPath) : modelPath(modelPath) {
       this->genConfig = new onnx::genai::GenerationConfig();
       OgaConfig::CreateGenerationConfig(modelPath.c_str(), &(this->genConfig));
       auto model = OgaModel::Create(modelPath.c_str());
       auto params = OgaGeneratorParams::Create(*model);

       this->tokenizer = OgaTokenizer::Create(*model);
       this->tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);
       this->generator = OgaGenerator::Create(*model, *params);
    };

    GenerationConfig* get_generation_config() override {
        return genConfig;
    };

    void set_generation_config(const GenerationConfig& config) override {
    };

    GenerationResult generate(const std::string& tokens) override {
        std::thread th(std::bind(&TerminateSession::Generator_SetTerminate_Call, &catch_terminate, generator.get()));

        // Define System Prompt
        const std::string system_prompt = std::string("<|system|>\n") + "You are a helpful AI and give elaborative answers" + "<|end|>";
        bool include_system_prompt = true;

        while (true) {
          signal(SIGINT, signalHandlerWrapper);
            std::string text;
            std::cout << "Prompt: (Use quit() to exit) Or (To terminate current output generation, press Ctrl+C)" << std::endl;
            // Clear Any cin error flags because of SIGINT
            std::cin.clear();
            std::getline(std::cin, text);

            if (text == "quit()") {
              break;  // Exit the loop
            }

            const std::string prompt = "<|user|>\n" + text + "<|end|>\n<|assistant|>";

            bool is_first_token = true;
            Timing timing;
            timing.RecordStartTimestamp();

            auto sequences = OgaSequences::Create();
            if (include_system_prompt) {
              std::string combined = system_prompt + prompt;
              tokenizer->Encode(combined.c_str(), *sequences);
              include_system_prompt = false;
            } else {
              tokenizer->Encode(prompt.c_str(), *sequences);
            }

            std::cout << "Generating response..." << std::endl;
            generator->AppendTokenSequences(*sequences);

            try {
              while (!generator->IsDone()) {
                generator->GenerateNextToken();

                if (is_first_token) {
                  timing.RecordFirstTokenTimestamp();
                  is_first_token = false;
                }

                const auto num_tokens = generator->GetSequenceCount(0);
                const auto new_token = generator->GetSequenceData(0)[num_tokens - 1];
                std::cout << tokenizer_stream->Decode(new_token) << std::flush;
              }
            } catch (const std::exception& e) {
              std::cout << "Session Terminated: " << e.what() << std::endl;
            }

            timing.RecordEndTimestamp();
            const int prompt_tokens_length = sequences->SequenceCount(0);
            const int new_tokens_length = generator->GetSequenceCount(0) - prompt_tokens_length;
            timing.Log(prompt_tokens_length, new_tokens_length);

            if (th.joinable()) {
              th.join();  // Join the thread if it's still running
            }

            for (int i = 0; i < 3; ++i)
              std::cout << std::endl;
        }

        return GenerationResult();
    };


private:
    const std::string modelPath;
    GenerationConfig* genConfig;
    std::unique_ptr<OgaTokenizer> tokenizer;
    std::unique_ptr<OgaTokenizerStream> tokenizer_stream;
    std::unique_ptr<OgaGenerator> generator;
};