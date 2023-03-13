load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
  name = "hf_inference",
  hdrs = [
    "args.h",
    "hf_inference.h",
    "options.h",
  ],
  deps = [
    "@curlpp//:curlpp",
    "@json//:json",
  ],
  visibility = ["//visibility:public"],
)
