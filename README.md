# BNET-DS


Abstract: Distant Supervision (DS) is a popular technique for developing relation extractors starting with limited supervision. Our contributions in this paper are threefold. Firstly, we propose three novel models for distantly-supervised relation extraction: (1) a Bi-GRU based word attention model (BGWA), (2) an entity-centric attention model (EA), and (3) and a combination model (BNET-DS) which jointly trains and combines multiple complementary models for improved relation extraction. Secondly, we introduce GDS, a new distant supervision dataset for relation extraction. GDS removes test data noise present in all previous distance supervision benchmark datasets, making credible automatic evaluation possible. Thirdly, through extensive experiments on multiple real-world datasets, we demonstrate effectiveness of the proposed methods.

## Preprocessing the files

The folder `Code/Preprocess/` has the files for preprocessing the data. 
For the **Reidel2010** dataset, just run the file `preprocess.sh` to get the output files in the same folder. There will be some intermediate files, but the final processed files will have the following name:
a. train_final.p : The processed train files
b. test_final.p : The processed test files
c. dev_final.p : The processed dev files

For the **GIDS** dataset, just run the file `preprocess_GIDS.sh` to get the output files in the same folder. There will be some intermediate files, but the final processed files will have the following name:
a. train_final.p : The processed train files
b. test_final.p : The processed test files
c. dev_final.p : The processed dev files

## Running the Model Files

The folder `Codes/Models/` has the files for the 3 models: 
1. **BGWA.py** : Bi-GRU based word attention model
2. **EA.py** : Entity-centric attention model
3. **PCNN.py** : Piecewise convolutional neural model

Each of the files can be run in the following way:
```
python2.7 <file> <data directory> <train file name> <test file name> <dev file name> <word embedding file name>
```
The command has 5 arguments
1. <data directory> : The name of the directory containing the processed files
2. <train file name> : The name (not the path) of the processed train file
3. <test file name> : The name (not the path) of the processed test file
4. <dev file name> : The name (not the path) of the processed dev file
5. <word embedding file name> : The name (not the path) of the processed word embedding file
