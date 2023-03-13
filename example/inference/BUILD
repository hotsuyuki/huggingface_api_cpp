load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
  name = "main",
  srcs = ["main.cc"],
  deps = [
    "//huggingface_api_cpp:inference",
    "@json//:json",
  ],
)
