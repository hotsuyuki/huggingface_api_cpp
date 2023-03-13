#pragma once

#include <nlohmann/json.hpp>

namespace huggingface_api_cpp::inference {

struct Options {
  bool retry_on_error = true;
  bool use_cache = true;
  bool use_gpu = false;
  bool wait_for_model = false;
};

struct ExtendedOptions : public Options {
  ExtendedOptions(const Options& options) : Options(options) {}
  bool binary = false;  // Whether the input is a file or not.
  bool blob = false;    // Whether the output_string_ftr is a file or not.
};

void to_json(nlohmann::json& json, const ExtendedOptions& extended_options) {
  json = nlohmann::json{
    {"retry_on_error", extended_options.retry_on_error},
    {"use_cache", extended_options.use_cache},
    {"use_gpu", extended_options.use_gpu},
    {"wait_for_model", extended_options.wait_for_model},
    {"binary", extended_options.binary},
    {"blob", extended_options.blob}
  };
}

}  // namespace huggingface_api_cpp::inference
