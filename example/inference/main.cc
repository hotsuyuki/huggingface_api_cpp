#include <filesystem>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "huggingface_api_cpp/inference.h"

using namespace huggingface_api_cpp::inference;

int main(const int argc, const char* argv[]) {
  assert(2 <= argc);
  const std::filesystem::path directory_path = argv[1];
  const std::string api_key = (3 <= argc) ? argv[2] : "";

  HfInference hf_inference(api_key);

  constexpr bool kRunFillMask = false;
  constexpr bool kRunSummarization = false;
  constexpr bool kRunQuestionAnswer = false;
  constexpr bool kRunTableQuestionAnswer = false;
  constexpr bool kRunTextClassification = false;
  constexpr bool kRunTextGeneration = false;
  constexpr bool kRunTokenClassification = false;
  constexpr bool kRunTranslation = false;
  constexpr bool kRunZeroShotClassification = false;
  constexpr bool kRunConversational = false;

  constexpr bool kRunAutomaticSpeechRecognition = false;
  constexpr bool kRunAudioClassification = false;

  constexpr bool kRunImageClassification = false;
  constexpr bool kRunObjectDetection = false;
  constexpr bool kRunImageSegmentation = false;
  constexpr bool kRunTextToImage = true;

  /////////////////////////////////
  // Natural Language Processing //
  /////////////////////////////////

  if (kRunFillMask) {
    const std::string output_string = hf_inference.fillMask(
      {.model = "bert-base-uncased"},
      {.inputs = "[MASK] world!"}
    );

    std::cout << nlohmann::json::parse(output_string).dump(2) << std::endl << std::endl;
    /*
    Output:
    [
      {
        "score": 0.2910905182361603,
        "sequence": "the world!",
        "token": 1996,
        "token_str": "the"
      },
      {
        "score": 0.18091173470020294,
        "sequence": "my world!",
        "token": 2026,
        "token_str": "my"
      },
      ...
      {
        "score": 0.015912111848592758,
        "sequence": "a world!",
        "token": 1037,
        "token_str": "a"
      }
    ]
    */
  }

  if (kRunSummarization) {
    const std::string output_string = hf_inference.summarization(
      {.model = "facebook/bart-large-cnn"},
      {
        .inputs = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930.",
        .parameters_opt = SummarizationArgs::Parameters{
          .max_length_opt = 100
        }
      }
    );

    std::cout << nlohmann::json::parse(output_string).dump(2) << std::endl << std::endl;
    /*
    Output:
    [
      {
        "summary_text": "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world."
      }
    ]
    */
  }

  if (kRunQuestionAnswer) {
    const std::string output_string = hf_inference.questionAnswer(
      {.model = "deepset/roberta-base-squad2"},
      {
        .inputs = QuestionAnswerArgs::Inputs{
          .question = "What is the capital of France?",
          .context = "The capital of France is Paris."
        }
      }
    );

    std::cout << nlohmann::json::parse(output_string).dump(2) << std::endl << std::endl;
    /*
    Output:
    {
      "answer": "Paris",
      "end": 30,
      "score": 0.9703431725502014,
      "start": 25
    }
    */
  }

  if (kRunTableQuestionAnswer) {
    // TODO: Use `nlohmann::ordered_json` instead of `nlohmann::json` to preserve the order of each element.
    const std::string output_string = hf_inference.tableQuestionAnswer(
      {.model = "google/tapas-base-finetuned-wtq"},
      {
        .inputs = TableQuestionAnswerArgs::Inputs{
          .query = "How many stars does the transformers repository have?",
          .table = {
            {"Repository", {"Transformers", "Datasets", "Tokenizers"}},
            {"Stars", {"36542", "4512", "3934"}},
            {"Contributors", {"651", "77", "34"}},
            {"Programming language", {"Python", "Python", "Rust, Python and NodeJS"}},
          }
        }
      }
    );

    std::cout << nlohmann::json::parse(output_string).dump(2) << std::endl << std::endl;
    /*
    Output:
    {
      "aggregator": "AVERAGE",
      "answer": "AVERAGE > 36542",
      "cells": [
        "36542"
      ],
      "coordinates": [
        [
          0,
          3
        ]
      ]
    }
    */
  }

  if (kRunTextClassification) {
    const std::string output_string = hf_inference.textClassification(
      {.model = "distilbert-base-uncased-finetuned-sst-2-english"},
      {.inputs = "I like you. I love you."}
    );

    std::cout << nlohmann::json::parse(output_string).dump(2) << std::endl << std::endl;
    /*
    Output:
    [
      [
        {
          "label": "POSITIVE",
          "score": 0.9998763799667358
        },
        {
          "label": "NEGATIVE",
          "score": 0.00012365408474579453
        }
      ]
    ]
    */
  }

  if (kRunTextGeneration) {
    const std::string output_string = hf_inference.textGeneration(
      {.model = "gpt2"},
      {.inputs = "The answer to the universe is"}
    );

    std::cout << nlohmann::json::parse(output_string).dump(2) << std::endl << std::endl;
    /*
    Output:
    [
      {
        "generated_text": "The answer to the universe is one in which we have no choice in the matter in which we live. And this is only possible through reason. If you are to understand physics, it is only necessary to understand the very nature of the matter in one"
      }
    ]
    */
  }

  if (kRunTokenClassification) {
    const std::string output_string = hf_inference.tokenClassification(
      {.model = "dbmdz/bert-large-cased-finetuned-conll03-english"},
      {
        .inputs = "My name is Sarah Jessica Parker but you can call me Jessica",
        .parameters_opt = TokenClassificationArgs::Parameters{
          .aggregation_strategy_opt = TokenClassificationArgs::Parameters::AggregationStrategy::kSimple
        }
      }
    );

    std::cout << nlohmann::json::parse(output_string).dump(2) << std::endl << std::endl;
    /*
    Output:
    [
      {
        "end": 31,
        "entity_group": "PER",
        "score": 0.9991335868835449,
        "start": 11,
        "word": "Sarah Jessica Parker"
      },
      {
        "end": 59,
        "entity_group": "PER",
        "score": 0.9979913234710693,
        "start": 52,
        "word": "Jessica"
      }
    ]
    */
  }

  if (kRunTranslation) {
    const std::string output_string = hf_inference.translation(
      {.model = "t5-base"},
      {.inputs = "My name is Wolfgang and I live in Berlin"}
    );

    std::cout << nlohmann::json::parse(output_string).dump(2) << std::endl << std::endl;
    /*
    Output:
    [
      {
        "translation_text": "Mein Name ist Wolfgang und ich lebe in Berlin"
      }
    ]
    */
  }

  if (kRunZeroShotClassification) {
    const std::string output_string = hf_inference.zeroShotClassification(
      {.model = "facebook/bart-large-mnli"},
      {
        .inputs = {
          "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"
        },
        .parameters_opt = ZeroShotClassificationArgs::Parameters{
          .candidate_labels = {
            "refund",
            "legal",
            "faq"
          }
        }
      }
    );

    std::cout << nlohmann::json::parse(output_string).dump(2) << std::endl << std::endl;
    /*
    Output:
    [
      {
        "labels": [
          "refund",
          "faq",
          "legal"
        ],
        "scores": [
          0.8777875304222107,
          0.10522652417421341,
          0.01698593609035015
        ],
        "sequence": "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"
      }
    ]
    */
  }

  if (kRunConversational) {
    const std::string output_string = hf_inference.conversational(
      {.model = "microsoft/DialoGPT-large"},
      {
        .inputs = ConversationalArgs::Inputs{
          .past_user_inputs = {
            "Which movie is the best?",
          },
          .generated_responses = {
            "It is Die Hard for sure.",
          },
          .text = "Can you explain why?"
        }
      }
    );

    std::cout << nlohmann::json::parse(output_string).dump(2) << std::endl << std::endl;
    /*
    Output:
    {
      "conversation": {
        "generated_responses": [
          "It is Die Hard for sure.",
          "It's the best movie ever."
        ],
        "past_user_inputs": [
          "Which movie is the best?",
          "Can you explain why?"
        ]
      },
      "generated_text": "It's the best movie ever.",
      "warnings": [
        "Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation."
      ]
    }
    */
  }

  //////////////////////
  // Audio Processing //
  //////////////////////

  if (kRunAutomaticSpeechRecognition) {
    const std::string output_string = hf_inference.automaticSpeechRecognition(
      {.model = "facebook/wav2vec2-large-960h-lv60-self"},
      {.data = directory_path / "test" / "sample1.flac"}
    );

    std::cout << nlohmann::json::parse(output_string).dump(2) << std::endl << std::endl;
    /*
    Output:
    {
      "text": "GOING ALONG SLUSHY COUNTRY ROADS AND SPEAKING TO DAMP AUDIENCES IN DRAUGHTY SCHOOLROOMS DAY AFTER DAY FOR A FORTNIGHT HE'LL HAVE TO PUT IN AN APPEARANCE AT SOME PLACE OF WORSHIP ON SUNDAY MORNING AND HE CAN COME TO US IMMEDIATELY AFTERWARDS"
    }
    */
  }

  if (kRunAudioClassification) {
    const std::string output_string = hf_inference.audioClassification(
      {.model = "superb/hubert-large-superb-er"},
      {.data = directory_path / "test" / "sample1.flac"}
    );

    std::cout << nlohmann::json::parse(output_string).dump(2) << std::endl << std::endl;
    /*
    Output:
    [
      {
        "label": "neu",
        "score": 0.5927600264549255
      },
      {
        "label": "hap",
        "score": 0.2002566158771515
      },
      ...
      {
        "label": "sad",
        "score": 0.07902464270591736
      }
    ]
    */
  }

  /////////////////////
  // Computer Vision //
  /////////////////////

  if (kRunImageClassification) {
    const std::string output_string = hf_inference.imageClassification(
      {.model = "google/vit-base-patch16-224"},
      {.data = directory_path / "test" / "cheetah.png"}
    );

    std::cout << nlohmann::json::parse(output_string).dump(2) << std::endl << std::endl;
    /*
    Output:
    [
      {
        "label": "cheetah, chetah, Acinonyx jubatus",
        "score": 0.9981812238693237
      },
      {
        "label": "leopard, Panthera pardus",
        "score": 0.000663304585032165
      },
      ...
      {
        "label": "tiger, Panthera tigris",
        "score": 5.714610961149447e-05
      }
    ]
    */
  }

  if (kRunObjectDetection) {
    const std::string output_string = hf_inference.objectDetection(
      {.model = "facebook/detr-resnet-50"},
      {.data = directory_path / "test" / "cats.png"}
    );

    std::cout << nlohmann::json::parse(output_string).dump(2) << std::endl << std::endl;
    /*
    Output:
    [
      {
        "box": {
          "xmax": 629,
          "xmin": 506,
          "ymax": 405,
          "ymin": 233
        },
        "label": "cat",
        "score": 0.9957962036132813
      },
      {
        "box": {
          "xmax": 616,
          "xmin": 532,
          "ymax": 203,
          "ymin": 33
        },
        "label": "cat",
        "score": 0.9948530793190002
      },
      ...
      {
        "box": {
          "xmax": 472,
          "xmin": 363,
          "ymax": 195,
          "ymin": 25
        },
        "label": "cat",
        "score": 0.9991069436073303
      }
    ]
    */
  }

  if (kRunImageSegmentation) {
    const std::string output_string = hf_inference.imageSegmentation(
      {.model = "facebook/detr-resnet-50-panoptic"},
      {.data = directory_path / "test" / "cats.png"}
    );

    std::cout << nlohmann::json::parse(output_string).dump(2) << std::endl << std::endl;
    /*
    Output:
    [
      {
        "label": "LABEL_199",
        "mask": "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",
        "score": 0.986684
      },
      {
        "label": "cat",
        "mask": "iVBORw0KGgoAAAANSUhEUgAAAoAAAAGkCAAAAACEc8tCAAAEj0lEQVR4nO3d3W6bQBCAUbvq+7+ye5FaitsA+z8wnCNVvalcsv4yCzaJHw8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKDXM/oATuVlQVb7FX0AZ/L6+4d1Un/Df8VU+iW+00u9JKeTeQK+Pv4q/Ndm4FqZA3x7lSQluxh3CJATSxzgt5l2PAMNwCCJA6yhvyh3CXC/MP2FuUuAu43pL85tAtypTH+B7hPgZmf6i/Q7+gAWen1/k0N253CnAN83G2jvRO4V4EN9Z3Ojc0DOSICEEiChBEgoAf7HDakrCZBQiQNsnGQG4FKJA+QKBEgoARJKgIQSIKEESCgBEipxgG68uoK8AervEvIGyCUIkFBZb8m3AV+ECUiolBPQ+LuOjBNQfxdysgn42U7LrXnqu5aY2y8rKqk8wP7+3JC61PLl7ink4GDHDD8BLjV2uefvf5vHO+y/FuBSA5d79dnXc85/KsClBi13nlN//a014io4T30s1xeg9OjUs+NkzM8OvFj7BMyYH8u1Big/hmgKUH2M0hCg/Bin/m6Y1P25BlmtdsVT56e/9SonYO7+WK8uwOT9GYDrVQWYvD8CZLwlv5UBGKAmwOQDUH8RKgJM3h8hbMFvBmCI8gANQCYoDjB7fwZgDFswoUoDzD4ACVIYoP6YwxZMqKJz71vMv/KrkI/lcPHSp2T9btFf+1JIsMfx6t0kv8fjaDF2FkKDzQ6X7kb9PR6PrQU5WgUFtlryC6eu52NZihZBgm2adx3+ocAmXoYZxTdrk90ArWkNq9ViL0ArynQ7AeqP+ZwDEmo7QAOQBUxAQm0GaADW8kJgCxOQUFsBGoC1DMAmJiChBDiIAdhGgISa8nGtf6eB80gObe0cjfH89HCFDzVtD1vyfWAHbjQ0wI6Pspz/BHZ2+Nx/AAE2Ghhg4XOQ6Od63l/KRQ//FLYuQuat6Q+PfNUn8Pl14Fc9/FMY9gHQVc/Ctwf37N3b9vNfWaCQaDHoZRj50WZIgPKjlXdCCDUiQAOQZgMC1B/ttgMs7Up/dHAOSKidAMtGmwFIDxOQUHsBGm5MtzsBFchstmBCCZBQR7vsGe5lJrH+CehHj+hgCybUUYA2WKY6nIAKZKYBW7CTQNodB/g0A5lnxEWIEUgzV8GEKgnQHsw0RRNQgcxiCyZUWYBGIJMUTkCf6socpVuwApmi+BxQgcxQfhGiQCaouApWIOPVvAyjQIarfIHFL+pmrOpo9hJUILWq3wkRGSPVvxW3U6DTQGp5L5hQDQEagYzTeEq3VZozROrYggnVPLI2ZqARSJWeYBJ97CBROoP5r0EBUqU7mM8E9Ued7osQydGj/yrYb06gw9iP6tIiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJzMH0a9dQbRt5R4AAAAAElFTkSuQmCC",
        "score": 0.999631
      },
      ...
      {
        "label": "cat",
        "mask": "iVBORw0KGgoAAAANSUhEUgAAAoAAAAGkCAAAAACEc8tCAAAFNklEQVR4nO3dzW7jOBCFUaXR7//K7kX+HHdsk1RJRbPO2cxigoyBfLiUZCezbQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA0O6S/QKI9if7BfS4KHA5b9kvoNlHe6/zgmmR//N8D+vp6/javvxXTKD0I/jy4x/PvozVpAf4SWE1/c1+AV8uDw5XdS5rmgVs5RJwLTMFeG/nLlf/Qn+LmSlAJ21BUwX4a4EXWa5srgBtYDmTBfhfgTf75xJwNfM8hvlgA2uZbQEfM4DLea0AWY4ASSVAUgmQVC8VoHuQ9bxUgKwnPUCrVlt6gB0FanVB+QFSmgBJJUBSCZBUAiSVAEk1QYCtT1c8hVnRBAFS2esEaACX9DoBsiQBkkqApJohQFd3hc0QIIUJkFQzBOh30QubIcAmLhTXNEGABrCyCQKksvwA2wbQCbyo9AAdwLVlB6i/4pID1F91uQHqr7zMi/uO/NyDrCpxAc0f2Ucw5eUFaADZEgPUH9uWF2Bff+5BlpXz/wkxf3xIWcDe/gzguk5eQNPHTycGKD7+d1qA8uM35wSoPu44PkDx8cCxAYqPJw59DKM/njnuEVtcfR4DLuywBbR+tDhkXWLjM4ArO2IBjR/N4u+C5UeH8AXUHz1iF1B9dApdQP3RKzJA/dEtMED90S8uQP0xICxA/TEiKkD9MSQoQP0xJuSN1kPz81bw0iIWUH8M88eJSBUQoOs/xu0P8Nj+nMCL2/thBPPHLpNfAxrA1U0eIKvbGeDBJ7ABXN6+APXHTrsCdAfCXnsC1B+7zXwT4gQuYOYAKWBHgEefwAawgvEAXQESYN4j2ACWMBygA5gIowE6gAkx6xFsAIuYNED9VTEYoDeBiTHlAuqvjrEAfQyfIFMuIHUMBWgAiTLfAuqvlJEAPYQmzHQLaABrGQjQABJntgU0gMXMFiDF9AfoBCaQBSRVd4AeQhPJApJKgKQSIKkESCoBkkqApBIgqQRIKgGSqjdA7wQTygKSqjPAmAH0ji+fchbwTYK8ywhQfXxxDUiqhAANIN8sIKmyAjSDbNtmAUnWF2Dg+yAmkG3rDND7cEQ7/wj+mD4xs23b9rfjayOa+Tx59ce2bacvoCs/fprsLtgwVnNugN8D6OMIbNvWFaB1Il57gAH9WT1unXoE21BunXsNeF3g73Oo0WKaT8WoMt7ev9PbvW/omK6l9ed93jIpsJTJngNSzXwBugospe3AOzcKh3Ah8y0gpcwYoEO4kBkDpBABkqopQGciR2kJUH8cxhFMqp7fCTmHp4ClTLeA+qsldwHVVt7xAT74PWD90dLAw7tgEbHHdNeA1LL30zAGkF32LqCH1OziCCZVW4APDloTyB6NC6hAjuEIJlVAgCaQce2PUR505lkMo0IC7Po+cCXqGtA5zBA3IaRqD9DfNOUAFpBUAiRVT4DOYMJ1LeCDAsXJkL4j+H5mHsMwpPMa0K0wsbqDGv1w9GXkP8by+psYKPDy/EsoaqCI3gIvLV9EUQPPATuvA297dbvCldAH0ZeWuCwgV4YCvL+B5o0+578Vp1GuDAa45xxVIN9GS7pb0e039EFqHhkuoPVhzNO902BpsQH++t1sIPft+PnfhtX8l940BwAAQI5/ot6GZotGOwEAAAAASUVORK5CYII=",
        "score": 0.998007
      }
    ]
    */
  }

  if (kRunTextToImage) {
    const std::filesystem::path output_file_path = directory_path / "test" / "blob.png";
    hf_inference.setOutputFilePath(output_file_path);

    const std::string output_string = hf_inference.textToImage(
      {.model = "stabilityai/stable-diffusion-2"},
      {
        .inputs = "award winning high resolution photo of a giant tortoise/((ladybird)) hybrid, [trending on artstation]",
        .negative_prompt_opt = "blurry"
      }
    );

    std::cout << nlohmann::json::parse(output_string).dump(2) << std::endl << std::endl;
    /*
    Output:
    A generated file will be saved in `output_file_path`.
    */
  }

  return EXIT_SUCCESS;
}
