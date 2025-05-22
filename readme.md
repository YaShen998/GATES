  
# Cost-aware Dynamic Workflow Scheduling via Deep Reinforcement Learning
This repository hosts the implementation of our several research works in Cost-aware Dynamic Workflow Scheduling:

1. *GATES: Cost-aware Dynamic Workflow Scheduling via Graph Attention Networks and Evolution Strategy,*  
  **To appear in IJCAI 2025.**

2. *Cost-Aware Dynamic Cloud Workflow Scheduling Using Self-attention and Evolutionary Reinforcement Learning,*
  **Awarded Best Paper at ICSOC 2024.**

Both works use the same simulator for policy learning, enhanced with distinct designs of deep reinforcement learning (DRL) algorithms.

## 📦 Environment Setup

You can set up the environment using the following methods:

### Using Conda (Recommended)

```bash

conda env create -f environment.yml -n your-env-name

conda activate your_env_name

````

> Make sure your Python version is `>=3.8`.


## 🚀 Quick Start

1.  Run  the  main  program:

```bash

python main.py

```
2. Test the saved model:

```bash

python eval_rl.py

```

> To run the code in a server, you need to customize `train.py` or `eval_rl.py` to a service script.
> For multiple independent runs, `main.py` should use a different random seed everytime.


## 📊 Example Output

After training and testing, results will be saved in the `logs/` directory, including total cost, SLA violations, and VM fees, which can be used for visualization and evaluation.

## 📚 Citation
If you find this project useful for your research, please consider give a star and cite the following papers in your future works:

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

@inproceedings{shen2025cost,
  title={Cost-Aware Dynamic Workflow Scheduling via Graph Attetion Networks and Evolution Strategy},
  author={Shen, Ya and Chen, Gang and Ma, Hui and Zhang, Mengjie},
  booktitle={International Joint Conference on Artificial Intelligence},
  year={2025}
}
```

## 🙋‍♂️ Contact
If you have any questions or Academic collaboration interests, feel free to reach out:

GitHub: [YaShen998](https://github.com/YaShen998), Email: ya.shen@ecs.vuw.ac.nz

## 🙏 Acknowledgements
We gratefully acknowledge the prior works by [Victoria Huang](https://niwa.co.nz/people/victoria-huang), [Chen Wang](https://niwa.co.nz/people/chen-wang), and [Yifan Yang](https://scholar.google.com/citations?user=dO8kmG4AAAAJ&hl=zh-CN), whose codes laid the foundation for this simulator. This work was also supported by the [AI-SCC & Big Data Group](https://ecs.wgtn.ac.nz/Groups/AISCC/WebHome) at Victoria University of Wellington.

## 📝 License
This project is licensed under the MIT License.
