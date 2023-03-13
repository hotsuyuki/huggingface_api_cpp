#pragma once

#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <curlpp/cURLpp.hpp>
#include <curlpp/Easy.hpp>
#include <curlpp/Exception.hpp>
#include <curlpp/Infos.hpp>
#include <curlpp/Options.hpp>
#include <nlohmann/json.hpp>

#include "huggingface_api_cpp/inference/args.h"
#include "huggingface_api_cpp/inference/options.h"

namespace huggingface_api_cpp::inference {

class HfInference {
 public:
  HfInference(const std::string& api_key = "") : api_key_(api_key)  {}

  void setApiKey(const std::string& api_key) {
    api_key_ = api_key;
  }

  void setOutputFilePath(const std::filesystem::path& output_directory_path) {
    output_file_path_ = output_directory_path;
  }

  /////////////////////////////////
  // Natural Language Processing //
  /////////////////////////////////

  std::string fillMask(const Args& args, const FillMaskArgs& other_args,
                       const Options& options = Options()) const {
    const ExtendedOptions extended_options(options);
    return request(args, other_args, extended_options);
  };

  std::string summarization(const Args& args, const SummarizationArgs& other_args,
                            const Options& options = Options()) const {
    const ExtendedOptions extended_options(options);
    return request(args, other_args, extended_options);
  };

  std::string questionAnswer(const Args& args, const QuestionAnswerArgs& other_args,
                             const Options& options = Options()) const {
    const ExtendedOptions extended_options(options);
    return request(args, other_args, extended_options);
  };

  std::string tableQuestionAnswer(const Args& args, const TableQuestionAnswerArgs& other_args,
                                  const Options& options = Options()) const {
    const ExtendedOptions extended_options(options);
    return request(args, other_args, extended_options);
  };

  std::string textClassification(const Args& args, const TextClassificationArgs& other_args,
                                 const Options& options = Options()) const {
    const ExtendedOptions extended_options(options);
    return request(args, other_args, extended_options);
  };

  std::string textGeneration(const Args& args, const TextGenerationArgs& other_args,
                             const Options& options = Options()) const {
    const ExtendedOptions extended_options(options);
    return request(args, other_args, extended_options);
  }

  std::string tokenClassification(const Args& args, const TokenClassificationArgs& other_args,
                                  const Options& options = Options()) const {
    const ExtendedOptions extended_options(options);
    return request(args, other_args, extended_options);
  }

  std::string translation(const Args& args, const TranslationArgs& other_args,
                          const Options& options = Options()) const {
    const ExtendedOptions extended_options(options);
    return request(args, other_args, extended_options);
  }

  std::string zeroShotClassification(const Args& args, const ZeroShotClassificationArgs& other_args,
                                     const Options& options = Options()) const {
    const ExtendedOptions extended_options(options);
    return request(args, other_args, extended_options);
  }

  std::string conversational(const Args& args, const ConversationalArgs& other_args,
                             const Options& options = Options()) const {
    const ExtendedOptions extended_options(options);
    return request(args, other_args, extended_options);
  }

  //////////////////////
  // Audio Processing //
  //////////////////////

  std::string automaticSpeechRecognition(const Args& args, const AutomaticSpeechRecognitionArgs& other_args,
                                         const Options& options = Options()) const {
    ExtendedOptions extended_options(options);
    extended_options.binary = true;
    return request(args, other_args, extended_options, other_args.data);
  }

  std::string audioClassification(const Args& args, const AudioClassificationArgs& other_args,
                                  const Options& options = Options()) const {
    ExtendedOptions extended_options(options);
    extended_options.binary = true;
    return request(args, other_args, extended_options, other_args.data);
  }

  /////////////////////
  // Computer Vision //
  /////////////////////

  std::string imageClassification(const Args& args, const ImageClassificationArgs& other_args,
                                  const Options& options = Options()) const {
    ExtendedOptions extended_options(options);
    extended_options.binary = true;
    return request(args, other_args, extended_options, other_args.data);
  }

  std::string objectDetection(const Args& args, const ObjectDetectionArgs& other_args,
                              const Options& options = Options()) const {
    ExtendedOptions extended_options(options);
    extended_options.binary = true;
    return request(args, other_args, extended_options, other_args.data);
  }

  std::string imageSegmentation(const Args& args, const ImageSegmentationArgs& other_args,
                                const Options& options = Options()) const {
    ExtendedOptions extended_options(options);
    extended_options.binary = true;
    return request(args, other_args, extended_options, other_args.data);
  }

  std::string textToImage(const Args& args, const TextToImageArgs& other_args,
                          const Options& options = Options()) const {
    ExtendedOptions extended_options(options);
    extended_options.blob = true;
    return request(args, other_args, extended_options);
  }

 private:
  template <typename T>
  std::string request(const Args& args, const T& other_args, const ExtendedOptions& extended_options,
                      const std::filesystem::path& input_file_path = std::filesystem::path()) const {
    std::future<std::string> output_string_ftr = std::async(std::launch::async, [&]() mutable {
      try {
        curlpp::Cleanup curlpp_cleanup;
        curlpp::Easy curlpp_request;

        // Headers.
        std::list<std::string> headers;
        if (!api_key_.empty()) {
          headers.push_back("Authorization: Bearer " + api_key_);
        }
        if (!extended_options.binary) {
          headers.push_back("Content-Type: application/json");
        }
        if (extended_options.binary && extended_options.wait_for_model) {
          headers.push_back("X-Wait-For-Model: true");
        }
        curlpp_request.setOpt(new curlpp::options::HttpHeader(headers));

        // URL.
        curlpp_request.setOpt(new curlpp::options::Url("https://api-inference.huggingface.co/models/" + args.model));

        // Body.
        const std::string body = extended_options.binary ? MakeBodyFromFile(input_file_path)
                                                         : MakeBodyFromJson(other_args, extended_options);
        curlpp_request.setOpt(new curlpp::options::PostFields(body));
        curlpp_request.setOpt(new curlpp::options::PostFieldSize(body.size()));

        /*
        extended_options.binary ? (std::cout << "body.size() = " << body.size() << std::endl << std::endl)
                                : (std::cout << "body = " << body << std::endl << std::endl);
        curlpp_request.setOpt(new curlpp::options::Verbose(true));
        */

        // Output.
        std::ofstream output_file_stream;
        std::ostringstream output_string_stream;
        if (extended_options.blob) {
          std::filesystem::create_directories(output_file_path_.parent_path());
          output_file_stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
          output_file_stream.open(output_file_path_, std::ios::out | std::ios::binary);
          curlpp_request.setOpt(new curlpp::options::WriteStream(&output_file_stream));
        } else {
          curlpp_request.setOpt(new curlpp::options::WriteStream(&output_string_stream));
        }

        // Performs the curlpp request.
        curlpp_request.perform();

        // If the output type is file, then performs the post process.
        if (extended_options.blob) {
          output_file_stream.close();
          const nlohmann::json output_file_path_json{
            {"output_file_path", output_file_path_.string()},
          };
          output_string_stream << output_file_path_json.dump();
        }

        // If the response code is a 503 error, then retries again with waiting for the model to be ready.
        const long response_code = curlpp::infos::ResponseCode::get(curlpp_request);
        if (extended_options.retry_on_error && response_code == 503 && !extended_options.wait_for_model) {
          std::cerr << "Received " << response_code << ", retry on error..." << std::endl;
          ExtendedOptions new_extended_options = extended_options;
          new_extended_options.wait_for_model = true;
          return request(args, other_args, new_extended_options, input_file_path);
        }

        return output_string_stream.str();
      }
      catch (const std::fstream::failure& e) {
        const nlohmann::json std_fstream_failure_json{
            {"std_fstream_failure", "Exception opening/reading/writing/closing file."},
        };
        return std_fstream_failure_json.dump();
      }
      catch(const curlpp::RuntimeError& e) {
        const nlohmann::json curlpp_runtime_error_json{
            {"curlpp_runtime_error", e.what()},
        };
        return curlpp_runtime_error_json.dump();
      }
      catch(const curlpp::LogicError& e) {
        const nlohmann::json curlpp_logic_error_json{
            {"curlpp_logic_error", e.what()},
        };
        return curlpp_logic_error_json.dump();
      }
    });

    output_string_ftr.wait();

    return output_string_ftr.get();
  }

  template <typename T>
  std::string MakeBodyFromJson(const T& other_args, const ExtendedOptions& extended_options) const {
    // Composes a JSON object.
    nlohmann::json body_json = other_args;
    body_json["options"] = extended_options;

    // Converts to `std::string` type.
    const std::string body = body_json.dump();

    return body;
  }

  std::string MakeBodyFromFile(const std::filesystem::path& input_file_path) const {
    // Opens the input file.
    std::ifstream input_file_stream;
    input_file_stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    input_file_stream.open(input_file_path, std::ios::in | std::ios::binary);

    // Gets the input file size.
    input_file_stream.seekg(0, std::ios::end);
    const std::size_t input_file_size = input_file_stream.tellg();
    input_file_stream.seekg(0, std::ios::beg);

    // Reads the input file contents.
    std::vector<char> body_chars(input_file_size / sizeof(char));
    input_file_stream.read(static_cast<char*>(&body_chars[0]), input_file_size);
    input_file_stream.close();

    // Converts to `std::string` type.
    const std::string body(body_chars.begin(), body_chars.end());

    return body;
  }
  
  std::string api_key_;
  std::filesystem::path output_file_path_;
};

}  // namespace huggingface_api_cpp::inference
