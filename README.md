# Federated Early-Stopping for Hindering Memorizing Heterogeneous Label Noise
**Journal:** Accepted by Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence

**Authors:** Bixiao Zeng, Xiaodong Yang, Yiqiang Chen, Zhiqi Shen, Hanchao Yu, Yingwei Zhang

**Url:** https://doi.org/10.24963/ijcai.2024/599

**Cite:** Zeng, B., Yang, X., Chen, Y., Shen, Z., Yu, H., & Zhang, Y. (2024). FedES: Federated Early-Stopping for Hindering Memorizing Heterogeneous Label Noise. International Joint Conference on Artificial Intelligence.

**Getting Started**
---
```bash
cd code/FedES
```

```bash
python build_dataset_fed.py --noise_mode norm --mean 0.5 --std 0.2 --num_clients 20 --dataset cifar10
```

```bash
python fednoisy/algorithms/FedES/main.py --noise_mode norm --mean 0.5 --std 0.2 --num_clients 20 --dataset cifar10
```

