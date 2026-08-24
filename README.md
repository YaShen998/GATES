# GATES: Cost-aware Dynamic Workflow Scheduling via Graph Attention Networks and Evolution Strategy

This repository contains the original training implementation, complete training and evaluation logs, and all saved checkpoints for the GATES paper, published at the **Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-2025)**.

GATES uses the same cost-aware dynamic workflow scheduling simulator as our earlier work, *Cost-Aware Dynamic Cloud Workflow Scheduling Using Self-attention and Evolutionary Reinforcement Learning*, awarded **Best Paper at ICSOC-2024**.

[![GATES Poster](IJCAI_2025_poster_ya.png)](IJCAI_2025_poster_ya.pdf)

---

## 📦 Environment Setup

Create and activate the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate gates
```

---

## 🚀 Quick Start

Run the main training program with a run number:

```bash
python main_01.py --run 1
```

The run number controls the training seed used by `main_01.py` (`seed = run * 100 + 1`). The default experiment configuration is stored in `config/workflow_scheduling_es_openai.yaml`.

Evaluate a final saved model:

```bash
python eval_rl_01.py \
  --gamma 1.0 \
  --wf_size S \
  --log_path logs/WorkflowScheduling-v3/20250323222324784072 \
  --model_num 2000
```

Evaluate all saved episode models in a run:

```bash
python eval_rl_02.py \
  --gamma 1.0 \
  --wf_size S \
  --log_path logs/WorkflowScheduling-v3/20250323222324784072 \
  --model_num 1
```

The original Slurm submission scripts are available as `myjob.sl` and `myjob_eval.sl`.

---

## Baseline Policies and Results

GATES, ESRL, and SPNCWS use the same workflow-scheduling simulator, training pipeline, and evolution-strategy optimizer. GATES is the default method in the repository root; the baseline policy networks and summary results are provided below:

| Method | Policy network | Summary results |
| --- | --- | --- |
| GATES | [`policy/wf_model_02.py`](policy/wf_model_02.py) | [`results/GATES_summary.csv`](results/GATES_summary.csv) |
| ESRL | [`baseline/ESRL/policy/wf_model.py`](baseline/ESRL/policy/wf_model.py) | [`results/ESRL_summary.csv`](results/ESRL_summary.csv) |
| SPNCWS | [`baseline/SPNCWS/policy/wf_model_01.py`](baseline/SPNCWS/policy/wf_model_01.py) | [`results/SPNCWS_summary.csv`](results/SPNCWS_summary.csv) |

The complete GATES checkpoints and logs are available in [`logs/`](logs/). To run a baseline with the shared training pipeline, select its corresponding `WFPolicy` module in `builder.py`.

---

## 📚 Citation

If you find this project useful for your research, please consider giving it a star and citing the following papers:

```bibtex
@inproceedings{huang2022cost,
  title={Cost-aware dynamic multi-workflow scheduling in cloud data center using evolutionary reinforcement learning},
  author={Huang, Victoria and Wang, Chen and Ma, Hui and Chen, Gang and Christopher, Kameron},
  booktitle={International Conference on Service-Oriented Computing},
  pages={449--464},
  year={2022},
  organization={Springer}
}

@inproceedings{shen2024cost,
  title={Cost-Aware Dynamic Cloud Workflow Scheduling Using Self-attention and Evolutionary Reinforcement Learning},
  author={Shen, Ya and Chen, Gang and Ma, Hui and Zhang, Mengjie},
  booktitle={International Conference on Service-Oriented Computing},
  pages={3--18},
  year={2024},
  organization={Springer}
}

@inproceedings{ijcai2025p960,
  title={GATES: Cost-aware Dynamic Workflow Scheduling via Graph Attention Networks and Evolution Strategy},
  author={Shen, Ya and Chen, Gang and Ma, Hui and Zhang, Mengjie},
  booktitle={Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}},
  publisher={International Joint Conferences on Artificial Intelligence Organization},
  pages={8635--8643},
  year={2025}
}
```

---

## 🙋‍♂️ Contact

If you have any questions or academic collaboration interests, feel free to reach out: [**Email**](mailto:ya.shen@ecs.vuw.ac.nz)

---

## 🙏 Acknowledgements

We gratefully acknowledge the prior works by [Victoria Huang](https://niwa.co.nz/people/victoria-huang), [Chen Wang](https://niwa.co.nz/people/chen-wang), and [Yifan Yang](https://scholar.google.com/citations?user=dO8kmG4AAAAJ&hl=zh-CN), whose codes laid the foundation for this simulator. This work was also supported by the [AI-SCC & Big Data Group](https://ecs.wgtn.ac.nz/Groups/AISCC/WebHome) at Victoria University of Wellington.

---

## 📝 License

This project is licensed under the Apache License 2.0. See `LICENSE` for the complete license text.
