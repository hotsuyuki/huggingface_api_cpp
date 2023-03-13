load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
  name = "curlpp",
  hdrs = glob([
    "include/curlpp/**/*.hpp",
    "include/curlpp/**/*.inl",
    "include/utilspp/**/*.hpp",
    "include/utilspp/**/*.inl",
  ]),
  srcs = glob(["src/curlpp/**/*.cpp"]),
  linkopts = [
    "-l curl",  # libcurl (URL data transfer library)
  ],
  strip_include_prefix = "include/",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "json",
  hdrs = glob(["include/nlohmann/**/*.hpp"]),
  strip_include_prefix = "include/",
  visibility = ["//visibility:public"],
)
