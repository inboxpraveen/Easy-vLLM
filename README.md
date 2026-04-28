# Easy-vLLM

**Run vLLM without memorizing every vLLM flag.**

Easy-vLLM is a simple web UI that generates ready-to-run vLLM Docker deployments. Choose your model, GPU count, memory budget, context length, quantization mode, and model config. easy-vLLM then generates Docker Compose files, optional Dockerfiles, environment files, OpenAI-compatible test clients, curl examples, and a deployment guide.

The goal is simple: make vLLM easier for everyone.

vLLM is already one of the best open-source engines for high-throughput LLM inference and serving. Easy-vLLM does not replace it. Instead, it helps more people use it correctly by removing deployment confusion, GPU memory guesswork, and command-line trial and error.

## Run it locally

```bash
# 1. Install dependencies (Python 3.10+).
pip install -r requirements.txt

# 2. Start the Flask app.
python app.py

# 3. Open the wizard.
# http://localhost:5000
```

Optional: run the test suite.

```bash
pytest -q
```

## How to use

1. **Step 1 - Model**: enter a Hugging Face model ID (e.g. `Qwen/Qwen3-8B-Instruct`) or a local path. Drag-drop the model's `config.json` for an accurate memory estimate. For MoE / multimodal models, also enter an approximate parameter count.
2. **Step 2 - Hardware & workload**: pick a GPU preset (RTX 4090, A100, H100, ...), how many GPUs you have, your expected input/output token sizes, and how many concurrent requests you need to support.
3. **Step 3 - Optimize**: pick precision and quantization (AWQ / GPTQ / FP8 / BitsAndBytes / GGUF). Toggle **Advanced** for full KV-cache, batching, image-tag, and raw-flag controls.

A live panel on the right shows you a memory-fit gauge (Good / Risky / Likely-OOM), a per-component breakdown (weights, KV cache, runtime overhead), color-coded warnings, ordered fixes, and a live preview of the `vllm serve` command.

When you're happy, click **Generate deployment** to download a zip containing:

```
easy-vllm-output/
  docker-compose.yml      # single-service vLLM deployment
  .env                    # ports, HF_TOKEN, paths, image tag
  test_client.py          # OpenAI Python client smoke test
  test_curl.sh            # curl smoke test
  README.md               # how-to run + memory verdict
  config_summary.json     # the wizard answers, for reproducibility
```

Then:

```bash
unzip easy-vllm-*.zip && cd easy-vllm-output
docker compose up -d
docker compose logs -f vllm
python test_client.py
```

## Project structure

```
.
├── app.py                       # Flask entrypoint
├── requirements.txt
├── easy_vllm/
│   ├── schemas.py               # Pydantic models
│   ├── gpu_presets.py           # GPU dropdown list
│   ├── config_parser.py         # parse HF config.json
│   ├── memory_estimator.py      # weights + KV cache + verdict
│   ├── command_builder.py       # vllm serve arg builder
│   ├── validators.py            # cross-field warnings
│   ├── docker_generator.py      # render artifact templates
│   ├── zip_exporter.py          # bundle artifacts to .zip
│   └── routes.py                # Flask routes
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── _step_model.html
│   ├── _step_hardware.html
│   ├── _step_optimization.html
│   ├── _live_panel.html
│   ├── _result.html
│   └── artifacts/               # Jinja2 templates for generated files
├── static/
│   ├── css/styles.css           # dark + light themes, animations
│   ├── js/app.js                # wizard, live estimator, drag-drop
│   └── img/logo.svg
└── tests/                       # pytest suite (29 tests)
```

## Memory estimator math

Per GPU:

- `weight_gb = (params * bytes_per_weight * 1.15) / 1024^3 / (tp_size * pp_size)`
- `kv_bytes_per_token = 2 * num_layers * ceil(kv_heads / tp) * head_dim * kv_dtype_bytes`
- `kv_cache_gb = kv_bytes_per_token * (input_tokens + output_tokens) * max_num_seqs / 1024^3`
- `required = weight_gb + kv_cache_gb + 2 GiB runtime`
- `usable = gpu_total_gb * gpu_memory_utilization`

Verdict:

| Total / Usable | Status |
|---|---|
| < 85 % | Good |
| 85 % - 100 % | Risky |
| > 100 % | Likely OOM |

The estimate is intentionally rough - vLLM profiles memory at startup and the real footprint also depends on CUDA graphs, kernels, fragmentation, and activations. It's accurate enough to catch most OOM disasters before deployment.

## What's intentionally out of scope

- Running Docker from the web app (security risk - we only generate files).
- Multi-node Ray/NCCL deployments (we surface a warning if you ask for `pipeline-parallel-size > 1`).
- Kubernetes manifests, benchmark runners, GPU auto-detection - tracked as future work.

## Why Easy-vLLM?

vLLM is one of the most powerful open-source engines for high-throughput LLM inference and serving. It gives developers access to production-grade features like OpenAI-compatible APIs, efficient GPU memory management, batching, quantization, Hugging Face model support, and Docker-based deployment. However, for many beginners, students, solo builders, and even backend developers, getting the right vLLM command can still be confusing. Users often need to understand GPU memory limits, model context length, tensor parallelism, quantization choices, Docker flags, CUDA compatibility, Hugging Face tokens, and API testing before they can run a single model successfully.

Easy-vLLM exists to make that first successful deployment much easier. Instead of forcing users to manually guess vLLM flags, Easy-vLLM provides a simple UI where they can select the model, upload the model config, choose GPU and memory settings, define input and output token limits, and generate a ready-to-run Docker Compose project. The goal is not to replace vLLM. The goal is to make vLLM more approachable, searchable, explainable, and usable for everyone who wants to self-host open-source LLMs.

## Credits and attribution

Easy-vLLM is a community helper project built around the amazing work of the vLLM project.

vLLM is the actual inference and serving engine used for high-throughput LLM serving. easy-vLLM does not replace vLLM, fork vLLM, or claim ownership of vLLM. This project simply helps users generate safer and easier deployment configurations for running vLLM with Docker, Docker Compose, GPU settings, model configuration, quantization choices, and OpenAI-compatible test clients.

All credit for the vLLM engine, serving architecture, OpenAI-compatible server, PagedAttention, CUDA/ROCm support, quantization integrations, and core inference runtime belongs to the official vLLM project and its contributors.

Official vLLM project:
https://github.com/vllm-project/vllm

Official vLLM documentation:
https://docs.vllm.ai/
