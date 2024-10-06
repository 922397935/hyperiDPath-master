# hyperiDPath
This is the PyTorch implementation for our paper:
Hypergraph interpretable deep learning-based path-reasoning framework for drug repurposing

# Introduction
hyperiDPath is a hypergraph interpretable deep learning-based path-reasoning framework for drug discovery and repurposing. This framework focuses on higher-order interactions among various biomolecules, including transcription factors, genes, proteins, and chemicals.

# Environment Requirement
The code has been tested running under Python 3.9. 

# Installation
1. To install the required packages for running hyperiDPath, please use the following   command first. If you meet any problems when installing pytorch, please refer to [pytorch official website](https://pytorch.org/)
```bash
pip install -r requirements.txt
```

2. You may need to download the following files to run hyperiDPath
* Download the the [data folder](https://drive.google.com/file/d/1YrL1frHNCHzfIkL-U_KEDO1zkmE9R_eS/view?usp=drive_link)
* Download the [result folder](https://drive.google.com/file/d/1B-VlSjDT3BE4gUoYro4IpUBAzUaXzab7/view?usp=drive_link)

# Example to Run the Codes
## Train 
```bash
python train.py --config config/config.json 
```
## Test 
When the training is finished, you will get a file that records the parameters for the best model, remember its location (such as `saved/models/hyperiDPath/0319_180500/model_best.pth`) and use it for testing.
```bash
python test.py --config config/config.json --resume saved/models/hyperiDPath/0319_180500/model_best.pth
```
## Inference
1. Data. To make an inference on the new drug-disease pair, you need to prepare a csv file named `test.csv` under the folder `data/test` with the following fields, where the drug(node) is denoted by its drugbank ID and disease(hyperedge) is denoted by its C0XX code. Note that if your input drugs or diseases cannot be found in our dataset, the corresponding pairs will be ignored.
```
node_1,hyperedge_1
DB00001,C0040038
```
2. Pre-trained model. You can use your own pre-trained model or use our prepared one [`model_best.pth`](https://drive.google.com/file/d/1WeG75vYUbNlP96kc6IHvbAedTRGG57i5/view?usp=sharing) and put the `config.json` and `model_best.pth` to the folder `data/test`.
3. Run. We provide an argument `K` in the `inference_config.json` to control the output of the number of top-k critical paths identified by iDPath. Please use the following command to run the inference.
```bash
python inference.py --resume data/test/model_best.pth --config config/inference_config.json
```
4. Result. After the inference is done, you will get a file named `result.csv` under the folder `saved/models/hyperiDPath/xxxx_xxxxxx` (where `xxxx_xxxxxx` is your runing time as the runing id). The `result.csv` contains the predicted probability of therapeutic effect and top-k critical paths of your input drug-disease pairs.

# Dataset
Datasets used in the paper:
* [RegNetwork](http://www.regnetworkweb.org/home.jsp)
* [STRING](https://version-10-5.string-db.org/cgi/download.pl?species_text=Homo+sapiens)
* [STITCH](http://stitch.embl.de/cgi/download.pl?UserId=zHfYv4tsZAzR&sessionId=tPYL1GXyX6xd&species_text=Homo+sapiens)
* [DrugBank](https://go.drugbank.com/releases/latest#full)
* [TTD](http://db.idrblab.net/ttd/)
* [DisGeNet](https://www.disgenet.org/home/)

  
# License
Distributed under the GPL-2.0 License License. See `LICENSE` for more information.

# Acknowledge
Thanks to the author of iDPath for open-sourcing the code.
# Contact
Bai Libing - LiBingBai10@gmail.com