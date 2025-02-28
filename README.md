
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

