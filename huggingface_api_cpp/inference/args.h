#pragma once

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace huggingface_api_cpp::inference {

struct Args {
  std::string model;
};

/////////////////////////////////
// Natural Language Processing //
/////////////////////////////////

struct FillMaskArgs {
  std::string inputs = "";
};

void to_json(nlohmann::json& json, const FillMaskArgs& other_args) {
  json = nlohmann::json{
    {"inputs", other_args.inputs},
  };
}

struct SummarizationArgs {
  struct Parameters {
    std::optional<int> max_length_opt = std::nullopt;
    std::optional<float> max_time_opt = std::nullopt;
    std::optional<int> min_length_opt = std::nullopt;
    std::optional<float> repetition_penalty_opt = std::nullopt;
    std::optional<float> temperature_opt = 1.0;
    std::optional<int> top_k_opt = std::nullopt;
    std::optional<float> top_p_opt = std::nullopt;
  };

  std::string inputs = "";
  std::optional<Parameters> parameters_opt = std::nullopt;
};

void to_json(nlohmann::json& json, const SummarizationArgs::Parameters& parameters) {
  json = nlohmann::json{};

  if (parameters.max_length_opt.has_value()) {
    json["max_length"] = parameters.max_length_opt.value();
  }
  if (parameters.max_time_opt.has_value()) {
    json["max_time"] = parameters.max_time_opt.value();
  }
  if (parameters.min_length_opt.has_value()) {
    json["min_length"] = parameters.min_length_opt.value();
  }
  if (parameters.repetition_penalty_opt.has_value()) {
    json["repetition_penalty"] = parameters.repetition_penalty_opt.value();
  }
  if (parameters.temperature_opt.has_value()) {
    json["temperature"] = parameters.temperature_opt.value();
  }
  if (parameters.top_k_opt.has_value()) {
    json["top_k"] = parameters.top_k_opt.value();
  }
  if (parameters.top_p_opt.has_value()) {
    json["top_p"] = parameters.top_p_opt.value();
  }
}

void to_json(nlohmann::json& json, const SummarizationArgs& other_args) {
  json = nlohmann::json{
    {"inputs", other_args.inputs},
  };

  if (other_args.parameters_opt.has_value()) {
    json["parameters"] = other_args.parameters_opt.value();
  }
}

struct QuestionAnswerArgs {
  struct Inputs {
    std::string question = "";
    std::string context = "";
  };

  Inputs inputs = {};
};

void to_json(nlohmann::json& json, const QuestionAnswerArgs::Inputs& parameters) {
  json = nlohmann::json{
    {"question", parameters.question},
    {"context", parameters.context},
  };
}

void to_json(nlohmann::json& json, const QuestionAnswerArgs& other_args) {
  json = nlohmann::json{
    {"inputs", other_args.inputs},
  };
}

struct TableQuestionAnswerArgs {
  struct Inputs {
    std::string query = "";
    std::unordered_map<std::string, std::vector<std::string>> table = {};
  };

  Inputs inputs;
};

void to_json(nlohmann::json& json, const TableQuestionAnswerArgs::Inputs& parameters) {
  json = nlohmann::json{
    {"query", parameters.query},
    {"table", parameters.table},
  };
}

void to_json(nlohmann::json& json, const TableQuestionAnswerArgs& other_args) {
  json = nlohmann::json{
    {"inputs", other_args.inputs},
  };
}

struct TextClassificationArgs {
  std::string inputs = "";
};

void to_json(nlohmann::json& json, const TextClassificationArgs& other_args) {
  json = nlohmann::json{
    {"inputs", other_args.inputs},
  };
}

struct TextGenerationArgs {
  struct Parameters {
    std::optional<bool> do_sample_opt = true;
    std::optional<int> max_new_tokens_opt = std::nullopt;
    std::optional<float> max_time_opt = std::nullopt;
    std::optional<int> num_return_sequences_opt = 1;
    std::optional<float> repetition_penalty_opt = std::nullopt;
    std::optional<bool> return_full_text_opt = true;
    std::optional<float> temperature_opt = 1.0;
    std::optional<int> top_k_opt = std::nullopt;
    std::optional<float> top_p_opt = std::nullopt;
  };

  std::string inputs = "";
  std::optional<Parameters> parameters_opt = std::nullopt;
};

void to_json(nlohmann::json& json, const TextGenerationArgs::Parameters& parameters) {
  json = nlohmann::json{};

  if (parameters.do_sample_opt.has_value()) {
    json["do_sample"] = parameters.do_sample_opt.value();
  }
  if (parameters.max_new_tokens_opt.has_value()) {
    json["max_new_tokens"] = parameters.max_new_tokens_opt.value();
  }
  if (parameters.max_time_opt.has_value()) {
    json["max_time"] = parameters.max_time_opt.value();
  }
  if (parameters.num_return_sequences_opt.has_value()) {
    json["num_return_sequences"] = parameters.num_return_sequences_opt.value();
  }
  if (parameters.repetition_penalty_opt.has_value()) {
    json["repetition_penalty"] = parameters.repetition_penalty_opt.value();
  }
  if (parameters.return_full_text_opt.has_value()) {
    json["return_full_text"] = parameters.return_full_text_opt.value();
  }
  if (parameters.temperature_opt.has_value()) {
    json["temperature"] = parameters.temperature_opt.value();
  }
  if (parameters.top_k_opt.has_value()) {
    json["top_k"] = parameters.top_k_opt.value();
  }
  if (parameters.top_p_opt.has_value()) {
    json["top_p"] = parameters.top_p_opt.value();
  }
}

void to_json(nlohmann::json& json, const TextGenerationArgs& other_args) {
  json = nlohmann::json{
    {"inputs", other_args.inputs},
  };

  if (other_args.parameters_opt.has_value()) {
    json["parameters"] = other_args.parameters_opt.value();
  }
}

struct TokenClassificationArgs {
  struct Parameters {
    enum class AggregationStrategy {
      kNone,
      kSimple,
      kFirst,
      kAverage,
      kMax,
    };

    static std::map<AggregationStrategy, std::string> AggregationStrategyMapper() {
      return {
        {AggregationStrategy::kNone, "none"},
        {AggregationStrategy::kSimple, "simple"},
        {AggregationStrategy::kFirst, "first"},
        {AggregationStrategy::kAverage, "average"},
        {AggregationStrategy::kMax, "max"},
      };
    }

    std::optional<AggregationStrategy> aggregation_strategy_opt = AggregationStrategy::kSimple;
  };

  std::string inputs = "";
  std::optional<Parameters> parameters_opt = std::nullopt;
};

void to_json(nlohmann::json& json, const TokenClassificationArgs::Parameters& parameters) {
  json = nlohmann::json{};

  if (parameters.aggregation_strategy_opt.has_value()) {
    const auto aggregation_strategy_mapper = TokenClassificationArgs::Parameters::AggregationStrategyMapper();
    json["aggregation_strategy"] = aggregation_strategy_mapper.at(parameters.aggregation_strategy_opt.value());
  }
}

void to_json(nlohmann::json& json, const TokenClassificationArgs& other_args) {
  json = nlohmann::json{
    {"inputs", other_args.inputs},
  };

  if (other_args.parameters_opt.has_value()) {
    json["parameters"] = other_args.parameters_opt.value();
  }
}

struct TranslationArgs {
  std::string inputs = "";
};

void to_json(nlohmann::json& json, const TranslationArgs& other_args) {
  json = nlohmann::json{
    {"inputs", other_args.inputs},
  };
}

struct ZeroShotClassificationArgs {
  struct Parameters {
    std::vector<std::string> candidate_labels = {};
    std::optional<bool> multi_label_opt = false;
  };

  std::vector<std::string> inputs = {};
  std::optional<Parameters> parameters_opt = std::nullopt;
};

void to_json(nlohmann::json& json, const ZeroShotClassificationArgs::Parameters& parameters) {
  json = nlohmann::json{
    {"candidate_labels", parameters.candidate_labels},
  };

  if (parameters.multi_label_opt.has_value()) {
    json["multi_label"] = parameters.multi_label_opt.value();
  }
}

void to_json(nlohmann::json& json, const ZeroShotClassificationArgs& other_args) {
  json = nlohmann::json{
    {"inputs", other_args.inputs},
  };

  if (other_args.parameters_opt.has_value()) {
    json["parameters"] = other_args.parameters_opt.value();
  }
}

struct ConversationalArgs {
  struct Inputs {
    std::vector<std::string> past_user_inputs = {};
    std::vector<std::string> generated_responses = {};
    std::string text = "";
  };

  struct Parameters {
    std::optional<int> max_length_opt = std::nullopt;
    std::optional<float> max_time_opt = std::nullopt;
    std::optional<int> min_length_opt = std::nullopt;
    std::optional<float> repetition_penalty_opt = std::nullopt;
    std::optional<float> temperature_opt = 1.0;
    std::optional<int> top_k_opt = std::nullopt;
    std::optional<float> top_p_opt = std::nullopt;
  };

  Inputs inputs = {};
  std::optional<Parameters> parameters_opt = std::nullopt;
};

void to_json(nlohmann::json& json, const ConversationalArgs::Inputs& parameters) {
  json = nlohmann::json{
    {"past_user_inputs", parameters.past_user_inputs},
    {"generated_responses", parameters.generated_responses},
    {"text", parameters.text},
  };
}

void to_json(nlohmann::json& json, const ConversationalArgs::Parameters& parameters) {
  json = nlohmann::json{};

  if (parameters.max_length_opt.has_value()) {
    json["max_length"] = parameters.max_length_opt.value();
  }
  if (parameters.max_time_opt.has_value()) {
    json["max_time"] = parameters.max_time_opt.value();
  }
  if (parameters.min_length_opt.has_value()) {
    json["min_length"] = parameters.min_length_opt.value();
  }
  if (parameters.repetition_penalty_opt.has_value()) {
    json["repetition_penalty"] = parameters.repetition_penalty_opt.value();
  }
  if (parameters.temperature_opt.has_value()) {
    json["temperature"] = parameters.temperature_opt.value();
  }
  if (parameters.top_k_opt.has_value()) {
    json["top_k"] = parameters.top_k_opt.value();
  }
  if (parameters.top_p_opt.has_value()) {
    json["top_p"] = parameters.top_p_opt.value();
  }
}

void to_json(nlohmann::json& json, const ConversationalArgs& other_args) {
  json = nlohmann::json{
    {"inputs", other_args.inputs},
  };

  if (other_args.parameters_opt.has_value()) {
    json["parameters"] = other_args.parameters_opt.value();
  }
}

//////////////////////
// Audio Processing //
//////////////////////

struct AutomaticSpeechRecognitionArgs {
  std::filesystem::path data;
};

void to_json(nlohmann::json& json, const AutomaticSpeechRecognitionArgs& other_args) {
  // Needs to be defined for compilation, but no need to implement `to_json()` for `AutomaticSpeechRecognitionArgs`
  // because it will never be called during the runtime.
}

struct AudioClassificationArgs {
  std::filesystem::path data;
};

void to_json(nlohmann::json& json, const AudioClassificationArgs& other_args) {
  // Needs to be defined for compilation, but no need to implement `to_json()` for `AutomaticSpeechRecognitionArgs`
  // because it will never be called during the runtime.
}

/////////////////////
// Computer Vision //
/////////////////////

struct ImageClassificationArgs {
  std::filesystem::path data;
};

void to_json(nlohmann::json& json, const ImageClassificationArgs& other_args) {
  // Needs to be defined for compilation, but no need to implement `to_json()` for `AutomaticSpeechRecognitionArgs`
  // because it will never be called during the runtime.
}

struct ObjectDetectionArgs {
  std::filesystem::path data;
};

void to_json(nlohmann::json& json, const ObjectDetectionArgs& other_args) {
  // Needs to be defined for compilation, but no need to implement `to_json()` for `AutomaticSpeechRecognitionArgs`
  // because it will never be called during the runtime.
}

struct ImageSegmentationArgs {
  std::filesystem::path data;
};

void to_json(nlohmann::json& json, const ImageSegmentationArgs& other_args) {
  // Needs to be defined for compilation, but no need to implement `to_json()` for `AutomaticSpeechRecognitionArgs`
  // because it will never be called during the runtime.
}

struct TextToImageArgs {
  std::string inputs = "";
  std::optional<std::string> negative_prompt_opt = std::nullopt;
};

void to_json(nlohmann::json& json, const TextToImageArgs& other_args) {
  json = nlohmann::json{
    {"inputs", other_args.inputs},
  };

  if (other_args.negative_prompt_opt.has_value()) {
    json["negative_prompt"] = other_args.negative_prompt_opt.value();
  }
}

}  // namespace huggingface_api_cpp::inference
