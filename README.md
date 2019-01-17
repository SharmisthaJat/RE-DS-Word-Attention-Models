# RE-DS-Word-Attention-Models
Paper Link: http://www.akbc.ws/2017/papers/15_paper.pdf 
Abstract: Distant Supervision (DS) is a popular technique for developing relation extractors starting with limited supervision. We note that most of the sentences in the distant supervision setting are very long and may benefit from word attention for better sentence representation. Our contributions in this paper are threefold. Firstly, we propose two novel word attention models for distantly-supervised relation extraction: (1) a Bi-GRU based word attention model (BGWA), (2) an entity-centric attention model (EA), and a combination model which combines multiple complementary models using weighted voting method for improved relation extraction. Secondly, we introduce GDS, a new distant supervision dataset for relation extraction. GDS removes test data noise present in all previous distant-supervision benchmark datasets, making credible automatic evaluation possible. Thirdly, through extensive experiments on multiple real-world datasets, we demonstrate effectiveness of the proposed methods

Citation: Please cite article https://arxiv.org/abs/1804.06987 , if you find this code useful. 

## Software required

python 2.7, 
pytorch 0.1.12, 
cuda 8.0, 
numpy 1.12.1, 
sklearn 0.18.2 

## Running the Model Files

The folder `Codes/Models/` has the files for the 3 models: 
1. **BGWA.py** : Bi-GRU based word attention model
2. **EA.py** : Entity-centric attention model
3. **PCNN.py** : Piecewise convolutional neural model
4. **ENSEMBLE.py** : Ensemble model for relation extraction

files 1,2,3 can be run in the following way:
```
python2.7 <file> <data directory> <train file name> <test file name> <dev file name> <word embedding file name>
```
The command has 5 arguments
1. <data directory> : The name of the directory containing the processed files
2. <train file name> : The name (not the path) of the processed train file
3. <test file name> : The name (not the path) of the processed test file
4. <dev file name> : The name (not the path) of the processed dev file
5. <word embedding file name> : The name (not the path) of the processed word embedding file

File 4 can be run using the following command

```
python2.7 ENSEMBLE.py <path_to_dataset_ensemble_files>

```
e.g: Codes/Models$ python2.7 ENSEMBLE.py ../../Data/Ensemble_Data/gids/ 

Please extract the preprocessed files for riedel dataset before using ensemble model by running the following command

Codes/Models$ unzip ../../Data/Ensemble_Data/riedel2010/preprocessed_dataset.zip

## Preprocessing the files

The folder `Code/Preprocess/` has the files for preprocessing the data. 
For the **Reidel2010** dataset, just run the file `preprocess.sh` to get the output files in the same folder. There will be some intermediate files, but the final processed files will have the following name:
1. train_final.p : The processed train files
2. test_final.p : The processed test files
3. dev_final.p : The processed dev files

For the **GIDS** dataset, just run the file `preprocess_GIDS.sh` to get the output files in the same folder. There will be some intermediate files, but the final processed files will have the following name:
1. train_final.p : The processed train files
2. test_final.p : The processed test files
3. dev_final.p : The processed dev files

For the two datasets used in the paper you can find pre-processed files at 'Data/Ensemble_Data/gids/preprocessed_dataset/' and 'Data/Ensemble_Data/riedel2010/preprocessed_dataset.zip' (extract in same folder)

# Saved Models and PR files

Find all the models and saved pr files from the following public link:

[Results_&_ Models](https://www.dropbox.com/s/69j59s81d8bxaa5/RE-best-models.zip?dl=0)

(Generated Precision-Recall (PR) files have precision in 1st column and recall in the second column)
