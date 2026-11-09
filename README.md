# MoE²-LoRA

Official code for **MoE²-LoRA: When MoE Models meet MoE-style Low-Rank Adaptation** (TMLR, 2026).

[Paper](https://openreview.net/forum?id=lTb6mATNkF)

MoE²-LoRA adapts frozen mixture-of-experts backbones with a shared pool of low-rank experts. A routing-conditioned projection combines pretrained router logits with a bottleneck projection of the MoE input hidden state to select adapter experts for each token.

## Installation

Use Python 3.11 or newer and a CUDA-enabled PyTorch installation matching your hardware.

```bash
git clone https://github.com/yqyati/moe-2lora.git
cd moe-2lora
pip install -e ".[eval]"
```

The training interface is based on LLaMA-Factory and retains the `llamafactory-cli` command. Run the commands below from the repository root.

## Prepare training data

```bash
python data/preprocess_metamathqa_r1.py
python data/preprocess_magicoder.py
```

These scripts download MetaMathQA-R1 and Magicoder-OSS-Instruct-75K and create the local JSON files registered in `data/dataset_info.json`. MetaMathQA-R1 preprocessing removes `<think>` blocks by default; use `--keep_think` to retain them. Datasets and model weights are downloaded separately under their respective licenses.

## Train

Three representative recipes cover global and grouped pools:

| Recipe | Backbone | Task | Total pool / groups | Rank | Top-K | Hidden bottleneck | Learning rate |
|---|---|---|---:|---:|---:|---:|---:|
| `examples/olmoe_math.yaml` | OLMoE-1B-7B | Math | 128 / 1 | 16 | 4 | 16 | 4e-4 |
| `examples/deepseek_math.yaml` | DeepSeek-V2-Lite | Math | 256 / 1 | 16 | 6 | 64 | 4e-4 |
| `examples/qwen3_code.yaml` | Qwen3-30B-A3B-Base | Code | 512 / 4 | 16 | 8 | 32 | 6e-4 |

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 \
  llamafactory-cli train examples/olmoe_math.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 \
  llamafactory-cli train examples/deepseek_math.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 \
  llamafactory-cli train examples/qwen3_code.yaml
```

Adjust GPU count, microbatch size, and gradient accumulation to your hardware. Effective batch size is `GPU count × per_device_train_batch_size × gradient_accumulation_steps`. DDP training keeps a full frozen backbone on each GPU. A local model directory can replace the Hugging Face ID:

```bash
llamafactory-cli train examples/olmoe_math.yaml model_name_or_path=/path/to/model
```

The YAML files specify the full recipe, including training duration and sample limits. Final adapter weights and routing configuration are saved under `saves/<recipe>`. Intermediate checkpointing is disabled in these compact examples; set `save_strategy=steps` and `save_steps` to enable it.

### Adapter settings

- `moe_lora_pool_share: global` shares adapter experts across layers.
- `moe_lora_n_experts` is the **total** number of adapter experts. With `moe_lora_n_groups > 1`, each group has `total / groups` experts, shared by a consecutive subset of MoE layers.
- `moe_lora_w_share: per_layer` gives each MoE layer its own routing projection.
- `moe_lora_hidden_bottleneck_dim` controls the auxiliary hidden-state channel.
- `moe_lora_alpha / moe_lora_rank` is the LoRA scaling coefficient.

The recipes freeze backbone weights and train adapter experts and routing projections. Adapter output matrices use a small normal initialization (standard deviation `1e-4`).

## Evaluate

Evaluation entry points cover GSM8K, MATH-500, HumanEval, MBPP, and general capability retention. For example:

```bash
python eval_scripts/eval_gsm8k.py \
  --base_model allenai/OLMoE-1B-7B-0924 \
  --adapter_path saves/olmoe_math --max_new_tokens 1024

python eval_scripts/eval_humaneval.py \
  --base_model Qwen/Qwen3-30B-A3B-Base \
  --adapter_path saves/qwen3_code --max_new_tokens 1024

python eval_scripts/eval_general.py \
  --base_model allenai/OLMoE-1B-7B-0924 \
  --adapter_path saves/olmoe_math --benchmark all
```

Use `--help` for each script's options, including local evaluation data and prediction output paths. Omit `--adapter_path` to evaluate the base model. Run adapter evaluation with the full backbone on one GPU; distributed evaluation uses one model replica per GPU. HumanEval and MBPP execute generated code and should run in an isolated evaluation environment.

## Code layout

```text
src/llamafactory/model/model_utils/moe_lora.py  # Routing, expert pools, injection, save/load
src/llamafactory/                             # Training framework and model integration
examples/                                    # Three MoE²-LoRA training recipes
data/                                        # Dataset preparation and registration
eval_scripts/                                # Evaluation entry points
tests/                                       # Core adapter tests
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q tests
```

## Citation

```bibtex
@article{yang2026moe2lora,
  title={MoE$^2$-LoRA: When MoE Models meet MoE-style Low-Rank Adaptation},
  author={Yang, Qingyu and He, Haonan and Li, Minglei and Ye, Jingqi and Chen, Tao and Bai, Lei and Ye, Peng},
  journal={Transactions on Machine Learning Research},
  year={2026},
  url={https://openreview.net/forum?id=lTb6mATNkF}
}
```

## Acknowledgments and license

The training framework builds on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), Hugging Face Transformers, and PEFT. Upstream copyright notices are preserved. Code is released under the [Apache License 2.0](LICENSE).
