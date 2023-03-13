[![test](https://github.com/hotsuyuki/huggingface_api_cpp/actions/workflows/test.yml/badge.svg)](https://github.com/hotsuyuki/huggingface_api_cpp/actions/workflows/test.yml)

# huggingface_api_cpp

A collection of C++ libraries to interact with the Hugging Face API (C++ version of [huggingface/huggingface.js](https://github.com/huggingface/huggingface.js)).

# Prerequisite

* [Bazel](https://github.com/bazelbuild/bazel) (or [Bazelisk](https://github.com/bazelbuild/bazelisk) for convenience)

# Usage example

For more details, see [./example/inference/main.cc](./example/inference/main.cc).

Command:
```Shell
$ cd /path/to/huggingface_api_cpp/
$ bazel run //example/inference:main -- /path/to/huggingface_api_cpp/huggingface_api_cpp/inference/ YOUR_API_KEY
```

Code:
```C++
#include "huggingface_api_cpp/inference.h"

using namespace huggingface_api_cpp::inference;

int main(const int argc, const char* argv[]) {
  const std::filesystem::path directory_path = "/path/to/huggingface_api_cpp/huggingface_api_cpp/inference/";
  const std::string api_key = "YOUR_API_KEY";

  HfInference hf_inference(api_key);

  /////////////////////////////////
  // Natural Language Processing //
  /////////////////////////////////

  const std::string output_string = hf_inference.translation(
    {.model = "t5-base"},
    {.inputs = "My name is Wolfgang and I live in Berlin"}
  );
  /*
  Output:
  [
    {
      "translation_text": "Mein Name ist Wolfgang und ich lebe in Berlin"
    }
  ]
  */

  //////////////////////
  // Audio Processing //
  //////////////////////

  const std::string output_string = hf_inference.automaticSpeechRecognition(
    {.model = "facebook/wav2vec2-large-960h-lv60-self"},
    {.data = directory_path / "test" / "sample1.flac"}
  );
  /*
  Output:
  {
    "text": "GOING ALONG SLUSHY COUNTRY ROADS AND SPEAKING TO DAMP AUDIENCES IN DRAUGHTY SCHOOLROOMS DAY AFTER DAY FOR A FORTNIGHT HE'LL HAVE TO PUT IN AN APPEARANCE AT SOME PLACE OF WORSHIP ON SUNDAY MORNING AND HE CAN COME TO US IMMEDIATELY AFTERWARDS"
  }
  */

  /////////////////////
  // Computer Vision //
  /////////////////////

  const std::filesystem::path output_file_path = directory_path / "test" / "blob.png";
  hf_inference.setOutputFilePath(output_file_path);

  const std::string output_string = hf_inference.textToImage(
    {.model = "stabilityai/stable-diffusion-2"},
    {
      .inputs = "award winning high resolution photo of a giant tortoise/((ladybird)) hybrid, [trending on artstation]",
      .negative_prompt_opt = "blurry"
    }
  );
  /*
  Output:
  A generated file will be saved in `output_file_path`.
  */

  return EXIT_SUCCESS;
}
```

The generated image by `HfInference::textToImage()` can be found in [./huggingface_api_cpp/inference/test/blob.png](./huggingface_api_cpp/inference/test/blob.png).

![./huggingface_api_cpp/inference/test/blob.png](./huggingface_api_cpp/inference/test/blob.png)
