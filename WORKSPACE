load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
  name = "com_google_googletest",
  url = "https://github.com/google/googletest/archive/release-1.11.0.zip",
  sha256 = "353571c2440176ded91c2de6d6cd88ddd41401d14692ec1f99e35d013feda55a",
  strip_prefix = "googletest-release-1.11.0",
)

http_archive(
  name = "curlpp",
  url = "https://github.com/jpbarrette/curlpp/archive/refs/tags/v0.8.1.zip",
  sha256 = "67bb923bee565d1076baa6a758d299594ff0d8fd26fc5e02b83c5f5b5764ccee",
  build_file = "@//:BUILD",
  strip_prefix = "curlpp-0.8.1",
)

http_archive(
  name = "json",
  url = "https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.zip",
  sha256 = "95651d7d1fcf2e5c3163c3d37df6d6b3e9e5027299e6bd050d157322ceda9ac9",
  build_file = "@//:BUILD",
  strip_prefix = "json-3.11.2",
)
