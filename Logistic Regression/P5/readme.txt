Fine-Tuning a Pre-Trained Network

Question) Run validation of the model every few training epochs on validation or test set of the dataset
and save the model with the best validation error. 

Ans)

The datasets are segregated into Training and Validation datasets from CIFAR10. 
For every epoch,the training and the validation datasets are ran. The no of epochs set here are 5.
For every epoch ran, the training and the validation loss are calculated and shown as below:-

Training and Validation loss is as below for 5 Epochs:-


Epoch	Training Loss	Validation Loss
0	0.3491		0.194
1	0.3128		0.1622
2	0.3071		0.1755
3	0.3013		0.1678
4	0.3006		0.1584



1) Running the Code on GPU

I have checked whether the system runs on a GPU or not using device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


2) Performing Validation of the model for training epochs 

2.1) Have created two dataloaders. Training and Validation dataloaders respectively.
For the validation, the datasets are created using the datasets.CIFAR10. 
Once the data is loaded, the train function is set to False for validation.

2.2) Using the for loop for the each epoch, the training and the Validationl losses are created.

2.3) In the validation phase, the validation dataloader is loaded. If the validation loss is less than the best Validation error, the best Validation error is updated.

2.4) The Validation losses are logged into a file named 'finalized_tck_model.sav' and the Validation losses are calculated.

The best Validation loss is calculated for every epoch and the model is saved.


3) How to run the code:-

3.1) Import copy, logging, pickle. (preferrably using pip install)
3.2) Create the environment using .yml file provided
3.3) Source activate the environment
3.4) Run the command 'python3 imagenet_finetune.py' 	