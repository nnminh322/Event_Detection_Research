# Lifelong Event Detection via Optimal Transport
Source code for the ACL Rolling Review submission LEDOT.


## Data & Model Preparation

We preprocess the data similar to [Lifelong Event Detection with Knowledge Transfer](https://aclanthology.org/2021.emnlp-main.428/) (Yu et al., EMNLP 2021), run the following commands to prepare data:
```bash
python prepare_inputs.py
python prepare_stream_instances.py
```

## Training and Testing

To start training on MAVEN, run:
```bash
sh sh/maven.sh
```

## Requirements:
- transformer == 4.23.1
- torch == 1.9.1
- torchmeta == 1.8.0
- numpy == 1.21.6
- tqdm == 4.64.1
- scikit-learn
- cvxpy
