#include <filesystem>
#include <iostream>
#include <limits>
#include <string>

#include "pipelines\pipeline_factory.hpp"  // Use the factory instead of the direct OV GenAI header
#include "ortgenai_text2text_pipeline.hpp"


std::map<std::string, std::any> setOrtGenAIConfig(const std::string& device) {
  std::map<std::string, std::any> device_config;
  if (device == "CONFIG") {
    device_config["ep"] = "follow_config";
  }
  else if (device == "CPU") {
    device_config["ep"] = "CPU";
  }
  else if (device == "GPU") {
    device_config["ep"] = "cuda";
    device_config["enable_cuda_graph"] = "0";
  }
  else if (device == "NPU") {
    device_config["ep"] = "OpenVINO";
    device_config["device_type"] = "NPU";
  }
  return device_config;
}


int main(int argc, char* argv[]) try {
  if (argc < 3 || argc > 4) {
    throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <BACKEND> <MODEL_DIR> [DEVICE]");
  }

  std::string backend_name = argv[1];
  std::string models_path = argv[2];
  std::string device = (argc == 4) ? argv[3] : "CONFIG";  // CPU, GPU, NPU can be used as well
  std::string prompt;

  if (backend_name == "onnxruntime_genai") {
    link_onnxruntime_genai_backend();
  }

  // Create the pipeline using the backend name provided as an argument
  auto text2text_pipeline = PipelineFactory::GetInstance().Create(backend_name, models_path);

  std::cout << "Successfully created '" << backend_name << "' backend." << std::endl;

  // --- The rest of your logic remains the same ---

  auto generation_config = text2text_pipeline->get_generation_config();
  generation_config.sampling_config.do_sample = true;
  generation_config.sampling_config.temperature = 0.7;
  generation_config.sampling_config.top_k = 40;
  generation_config.sampling_config.rng_seed = 42;
  text2text_pipeline->set_generation_config(generation_config);

  std::map<std::string, std::any> device_config = setOrtGenAIConfig(device);
  text2text_pipeline->set_device_config(device_config);

  std::cout << "question:\n";
  while (std::getline(std::cin, prompt)) {
    if (prompt == "exit") break;
    auto results = (*text2text_pipeline)(prompt);
    std::cout << results.text << std::endl;
    std::cout << "\n----------\n"
              << "question:\n";
  }

} catch (const std::exception& error) {
  try {
    std::cerr << "Error: " << error.what() << '\n';
  } catch (const std::ios_base::failure&) {
  }
  return EXIT_FAILURE;
} catch (...) {
  try {
    std::cerr << "Non-exception object thrown\n";
  } catch (const std::ios_base::failure&) {
  }
  return EXIT_FAILURE;
}
