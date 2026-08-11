# Third-party models, benchmarks, data, and images

This repository contains original experiment code and a research report built on third-party models and datasets. No ownership of those third-party materials is claimed.

## Qwen2-VL-2B-Instruct

- Model: [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- Upstream license: Apache License 2.0, as declared by the model repository.
- Model weights are not included in this repository or its release archive.

## POPE

- Project: [RUCAIBox/POPE](https://github.com/RUCAIBox/POPE)
- Paper: *Evaluating Object Hallucination in Large Vision-Language Models* (EMNLP 2023).
- Upstream repository license: MIT.

## RePOPE

- Project and corrected annotations: [YanNeu/RePOPE](https://github.com/YanNeu/RePOPE)
- Paper: [RePOPE: Impact of Annotation Errors on the POPE Benchmark](https://arxiv.org/abs/2504.15707)
- The final report uses the retained and corrected RePOPE labels as benchmark ground truth.

## MS COCO images and annotations

- Dataset: [COCO — Common Objects in Context](https://cocodataset.org/)
- COCO images originate from Flickr and retain their individual image licenses. COCO annotations and dataset use remain subject to the COCO terms of use.
- The final report includes 25 diagnostic images extracted from the POPE/RePOPE COCO subset. They are included solely to explain model errors and are not relicensed by this repository.

## Repository license status

No project-wide open-source license is granted by this repository. Third-party materials remain governed by their original licenses and terms.
