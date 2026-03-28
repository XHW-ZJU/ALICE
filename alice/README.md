# ALICE System
## Mapping AAV capsid sequences to functions through in silico function-guided evolution

✨ALICE offers a novel design paradigm for designing protein complexes: AAV capsids, helping researchers to explore the potential AAV capsid space more accurately. It provides new ideas for AI to design protein complexes under limited data conditions. More importantly, it accelerates the development of new serotype capsids in gene therapy.

## Note
🔗To replicate the sequence screening outcomes detailed in our paper, please follow these steps: 

📥First, download the pertinent data from Google Drive. 

📁Next, navigate to the "step5_elite_ascendancy" folder within the ALICE project directory. 

☺️Execute the scripts sequentially, numbered from 1 to 8, to acquire the eight sequences featured in the study for subsequent wet-lab validation. 

Should you encounter any inquiries or challenges, feel free to post a message in the issue section; we are committed to providing a prompt response.

## Required python packages
- python 3.8-3.10
- pytorch torch-deploy-1.8 (GPU 3090, CUDA 11.2)
- torch = 1.10.1+cu113
- torchvision = 0.11.2+cu113
- rdkit (2021.09.4)
- bio = 1.3.3
- biopython = 1.79
- sklearn = 1.0.2
- numpy = 1.21.5
- pandas = 0.24.2
- scipy = 1.7.1
- openbabel = 3.1.1
- argparse
- jupyter-client==7.4.9
- jupyter-core==4.12.0
- jupyterlab-pygments==0.2.2
- kiwisolver==1.4.4
- scikit-learn==1.0.2
- scipy==1.7.3
- tokenizers==0.13.3
- traitlets==5.9.0
- transformers==4.30.2
- xgboost==1.6.2
- zipp==3.15.0
- huggingface-hub==0.14.1
- bleach==6.0.0
- safetensors==0.3.1

(for details see the file of 'environment.yml')

**News:**    
```yaml
Key Innovations of ALICE
```
ALICE (AI-driven Ligand-Informed Capsid Engineering) represents a groundbreaking approach to AAV capsid engineering, offering several key innovations:
- [x] 1. Direct in silico design of multifunctional capsids: ALICE achieves direct computational engineering of AAV capsids with multiple functions, independent of in vivo selection processes. While AI has been used to construct capsid libraries with enhanced fitness or diversity, previous approaches still relied on unpredictable in vivo selection. ALICE pioneers the direct in silico design of multifunctional capsids, potentially accelerating AAV-based gene therapy development more efficiently.
- [x] 2. Function-guided evolutionary approach: ALICE overcomes the challenge of fusing multiple functions into a single sequence, a limitation often faced by conventional generative language models. By employing a novel function-guided evolutionary approach, ALICE successfully integrates multiple functions into a single capsid sequence, bridging the gap in AI-driven design of multifunctional AAV capsids and providing insights for other multifunctional protein complexes.
- [x] 3. Interpretable computational mapping of capsid sequence to function: ALICE demonstrates an interpretable approach to mapping capsid sequences to multiple functions. By illustrating multi-level relationships between capsid sequences and functions, ALICE enables knowledge-driven AAV design. This approach offers insights into the interpretability, controllability, and safety of viral capsid protein engineering, addressing crucial bio-security concerns in the field.

------

## Usage 
### Please access and download the relevant data from the following Google Drive link:

  > [ALICE Data Download](https://drive.google.com/drive/folders/1vHAZe2p-mq58dbaS-pdC5AtfR12eYMDX?usp=sharing)

    https://drive.google.com/drive/folders/1vHAZe2p-mq58dbaS-pdC5AtfR12eYMDX?usp=sharing

Upon downloading, ensure that the acquired "input", "output", and "checkpoint" directories are placed at the same hierarchical level as the "step1_pretrain", "step2_semantic_tuning", and other directories within the ALICE project folder structure. This directory organization is crucial for maintaining the correct relative paths and enabling seamless execution of the ALICE pipeline.


### Step 1: Pretrain Module (directory: step1_pretrain)
This module is used for training and inferencing the RoBERTa pretrain model on UniProt datasets. It provides a solid foundation for understanding protein sequences and their underlying patterns.
> [RoBERTa Pretraining](https://github.com/StarLight1212/AAV.ALICE/tree/main/ALICE/step1_pretrain)
#### Key components:
- [x] Data preprocessing of UniProt datasets
- [x] RoBERTa model configuration and initialization
- [x] Training pipeline with masked language modeling objective

### Step 2: Semantic Tuning Module (directory: step2_semantic_tuning)
This module focuses on training and inferencing the RoBERTa-based SeqGAN (Sequence Generative Adversarial Network) model using AAV capsid sequences. It aims to capture the semantic properties of functional capsid sequences.
> [Semantic Tuning](https://github.com/StarLight1212/AAV.ALICE/tree/main/ALICE/step2_semantic_tuning)
#### Key components:
- [x] SeqGAN architecture implementation
- [x] Training pipeline for the generator and discriminator
- [x] Inference scripts for generating novel capsid sequences

### Step 3: Ranking and Filtration Process (directory: step3_rankfiltpro)
This module is responsible for ranking and filtering the capsid sequences generated by the RoBERTa-based SeqGAN. It focuses on identifying sequences with high production fitness and high binding affinity to the targets Ly6a and Ly6c1.
> [RankFiltPro](https://github.com/StarLight1212/AAV.ALICE/tree/main/ALICE/step3_rankfiltpro)
#### Key components:
- [x] Filtration criteria implementation

### Step 4: Function-guided Evolution (FE) (directory: step4_function_guided_evolution)
This module evolves and fuses multiple functions into the capsid sequences using contrastive and heuristic strategies. It aims to optimize the sequences for desired functional properties.
> [Function-guided Evolution](https://github.com/StarLight1212/AAV.ALICE/tree/main/ALICE/step4_function_guided_evolution)
#### Key components:
- [x] Multi-objective optimization framework
- [x] Contrastive learning object function
- [x] Heuristic search algorithms
- [x] Evolution pipeline for iterative improvement

### Step 5: Elite Ascendancy (directory: step5_elite_ascendancy)
This module selects the best capsid sequences with multiple desired functions for in vivo testing. It represents the final stage of the computational pipeline before experimental validation.
> [Elite Ascendancy](https://github.com/StarLight1212/AAV.ALICE/tree/main/ALICE/step5_elite_ascendancy)
#### Key components:
- [x] Evaluate the sequence's production fitness and binding affinity to the targets Ly6a and Ly6c1.
- [x] Elite Ascendancy Score Calculation (EA Score)
- [x] Output generation of elite sequences for testing

### Evaluation (directory: evaluation)
This module is applied to predict (train and inference) the binding ability of Ly6a, Ly6c1, and production fitness of AAV capsid sequences. It provides crucial feedback for the entire pipeline.
> [Evaluation](https://github.com/StarLight1212/AAV.ALICE/tree/main/ALICE/evaluation)
#### Key components:
- [x] Model architectures for binding and fitness prediction
- [x] Training pipelines for each prediction task

Each module in this pipeline builds upon the previous one, creating a comprehensive framework for designing and optimizing AAV capsid sequences with desired functional properties. The FE module serves as a critical component for fusing and generating the capsid sequences with multiple functions throughout the process.

### Statistics in Wet-lab (directory: statistical_code)
The statistics and analysis codes for both in vivo experiments and mechanism assays are compiled in this collection.

## License  
This repo is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, and scientific publications. Permission is granted to use ALICE given that you agree to my licensing terms. Regarding the request for commercial use, please contact us via email to help you obtain the authorization letter.  


