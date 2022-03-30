# IDRID_hierarchical_combination
Solution for sub-challenge 2 IDRID Dataset ( https://idrid.grand-challenge.org/ ) based on a hierarchical combination 

Our solution would get third place in the final ranking of the competition if the use of extra training data ( https://pubmed.ncbi.nlm.nih.gov/31671320/ )


Based on the idea presented in the paper "Coarse-to-fine classification for diabetic retinopathy grading using convolutional neural network" (https://www.sciencedirect.com/science/article/pii/S0933365720301354)

This solution uses a hierarchical combination of deep neural networks for the classification of diabetic retinopathy in its entire spectrum, showing an improvement of 2% in relation to the paper used as a starting point.



### Coarse-to-fine classification for diabetic retinopathy grading using convolutional neural network - Results

Coarse Network (binary classification) ..................................: 80.58% \
Fine Network (multi-class classification) ...............................: 58.33% \
CF-DRNet (final classification) ..............................................: 56.19% 


### This solution - Results

Binary classification ...................................................: 83.55% \
Multi-class classification .............................................: 55.26% \
Final classification ......................................................: 58.25% 


# How to run?

1 - put the test IDRID in : idrid_datast/test \
2 - put the training IDRID in : idrid_datast/train \
3 - the .csv files must be in: idrid_datast/test.csv and train.csv idrid_datast/train.csv with the headers: id_code,level \
4 - pip install -r requirements.txt \
5 - Run 


This solution was built using PyTorch and made use of densenet121 in both classification phases
