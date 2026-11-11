# Sources and licenses

UniMI code is licensed under Apache-2.0; see [LICENSE](LICENSE).

- Base model: [Qwen/Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B),
  the pretrained Base model without instruction tuning. Its model card states
  Apache-2.0. Download the base weights separately and retain their notices.
- SPA teacher: [Qwen/Qwen3-VL-Reranker-8B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B).
  Teacher weights are not included. The supplied SPA JSONL contains its scores.
- Product and interaction data: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/).
  Product text, identifiers, and images originate from that collection. The
  code license does not grant a new license to this third-party content;
  consult the source collection for its terms and attribution.

The generated images and synthetic records in `examples/` illustrate the
input format and are not products from the experimental datasets.
