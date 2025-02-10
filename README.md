<h1 align="center">Synergistic Drug Combination Prediction with Interactive Permutation—Agnostic Molecular Representation Learning</h1>

The code is from our new paper in the field of synergistic drug combination prediction, entitled'' Synergistic Drug Combination Prediction with Interactive
Permutation—Agnostic Molecular Representation Learning (PerMolSyner) ''. PerMolSyner is a powerful tool designed for drug combination prediction task, which leverages advanced algorithms to improve efficiency and accuracy.
## Highlight
-  To our knowledge, this is the first work to consider the consistency issue in synergistic drug prediction methods, whereby interactive representation learning methods along a single-direction in drug combinations cannot encode multi-scale molecule-inter interaction semantics and lead to inconsistent predictions under different orders of pairs of drugs-cell lines.
 -  We design a novel synergistic drug prediction model entitled PerMolSyner, which introduces a bidirectional attention block to comprehensively model intricate interactive semantics with different interaction orders while integrating pretrain-finetune mechanism to learn consistent feature representations of the same drug-drug-cell line on different input orders, thereby enhancing the model’s generalizability.
 - Extensive experiments on three benchmark datasets demonstrate PerMolSyner’s superiority over SOTA base-lines.
## Model Architecture
Figure 1: The overall architecture of PerMolSyner, which consists of two critical modules, i.e., (a) a molecule interaction representation module and (b) a pretrain-fineture module. The bidirectional attention block in the molecule interaction representation module comprises an encoder-decoder structure with different directions, i.e., the Drug A-drug B direction and the Drug B-drug A direction, as illustrated in Figure 1(c).

![image](https://github.com/AlexCostra/PerMolSyner/blob/main/Utils/Fig1.png)

## Requirements
- Python >= 3.6
- Pytorch >= 1.7

## Folder Specification

- **main_PerMolSyner.py:** Please excuate this file to run our model.
- **Model:** It includes our model of PerMolSyner (model.py).
- **Data** This folder includes samples.csv containing four rows of data. i.e., molecule SMILES Strings of Drug A, molecule SMILES Strings of Drug B, cell line ID and their synergistic relationships(1:synergistic relation and 0: non-synergistic relation).
- **Utils:**  This folder includes utils_test.py responsible for processing sequence data of drug moleculers and samples of synergistic drug combinations.
## Run the Code
  To excecute PerMolSyner, please run the following command. The contents include model train and model test:

```bash
cd PerMolSyner
python main_PerMolSyner.py
``` 
## Acknowledgement
We sincerely thank Weiyu Shi for providing code. Please contact us with Email: standyshi@qq.com
<u><p><b><i><font size="6">If you are interested in Natural Language Processing and Large Language Models, feel free to contact us by Email: zhangyijia@dlmu.edu.cn </font></i></b></p>

