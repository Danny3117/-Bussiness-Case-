# Bussiness Case action plan

1.	Preprocess the data
	1.1	Balance the dataset
	1.2	Divide the dataset into training, validation and test
	1.3	Save the data in a tensor format(.npz)
2.	Create the Machine Learning Algorithm.

Extract the data from CSV.file:

In[]: import numpy as np
        From sklearn import preprocessing
        Raw_csv_data = np.loadtxt(‘Audiobooks_data.csv’,delimiter =’,’)
        Unscaled_inputs_all = raw_csv_data[:,1:-1]
        Targets_all = raw_csv_data[:,-1]

#Balane the datasheet			
In[]: 	num_one_targets = int (np.sum(targets_all))
	Zero_targets_counter = 0
	Indices_to_remove =[]
	For i in range(targets_all_shape[0]):
		If targets_all[i] == 0:
		Zero_targets_counter +=1
	If zero_targets_counter > num_one_targets:
	Indices_to_remove.append(i)
	Unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all,indices_to_remove,axis = 0)
	Targets_equal_priors = np.delete(targets_all, indices_to_remove,axis = 0)
Scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors) #standardize the inputs:

#shuffle the data:
In[]:	shuffled_indices = np.arrange(scaled_inputs.shape[0])	
	Np.random.shuffle(shuffled_indices)
	Shuffled_inputs = scaled_inputs[shuffled_indices]
	Shuffled_targets = targets_equal_priors[shuffled_indices]

#Split the dataset into train,validation and test:
In[]: 	samples_count = shuffled_inputs.shape[0]
	Train_samples_count = int(0.8*samples_count)
	validation_samples_count = int(0.1*samples_count)
	test_samples_count = samples_count_train_samples_count_validation.samples_count
	train_inputs = shuffled_inputs[:train_samples_count]
	train_targets = shuffled_targets[:train_samples_count]
	validation_inputs =  shuffled_inputs[train_sample_count:train_samples_count + validation_samples_count]
	validation_targets=  shuffled_ targets[train_sample_count:train_samples_count + validation_samples_count]
	
	test_input = shuffled_inputs[train_samples_count + validation_samples_count]
	test_targets= shuffled_ targets[train_samples_count + validation_samples_count]
	
	print(np.sum(train_targets),train_samples_count,np.sum(train_targets)/train_samples_count)
	print(np.sum(validation_targets), validation_samples_count,np.sum(validation_targets)/ validation_samples_count)
	print(np.sum(test_targets), test_samples_count,np.sum(test_targets)/test_samples_count)

#Save the three dataset in .npz file
In[]:	np.savez(‘Audiobooks_data_train’,inputs = train_inputs,targets= train_tragets)
	np.savez(‘Audiobooks_data_validation,inputs = validation_inputs,targets= validation_tragets)
	np.savez(‘Audiobooks_data_test,inputs = test_inputs,targets= test_tragets)

# Create ML Algorithm:
in[]:	import numpy as np
	import tensorflow as tf
in[]:	npz = np.load('Audiobooks_data_train.npz')
	train_inputs = npz['inputs'].astype(np.float)
	train_targets= npz['inputs'].astype(np.int)

	npz = np.load('Audiobooks_data_validation.npz')
	validation_inputs = npz['inputs'].astype(np.float)
	validation_targets= npz['inputs'].astype(np.int)

	npz = np.load('Audiobooks_data_test.npz')
	test_inputs = npz['inputs'].astype(np.float)
	test_targets= npz['inputs'].astype(np.int)

#Model
in[]:	input_size = 10
	output_size = 2
	hidden_layer_size = 50
	model = tf.keras.sequential([
		tf.keras.layers.dense(hidden_layers_size, activation = 'relu'),
		tf.keras.layers.dense(hidden_layers_size, activation = 'relu'),
		tf.keras.layers.dense(output_size, activation = 'softmax'),
	model.compile(optimiser='adam', loss = 'sparse_categorical_crssentropy',metrics = ['accracy'])
	batch_size = 100
	max_epochs = 100
	early_stopping = tf.keras.callbacks.Earlystopping(patience=2)

	model.fit(train_inputs,train_targets, batch_size = batch_size, epochs = max_epochs,
	callbacks = [early_stopping],
	validaton_data = (validation_inputs, validation_targets), verbose = 2)

#Test the model
in[]:	test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
in[]:	print('\n test loss: {0:.2f}.test accuracy:{1:.2f}.format(test_loss, test_accuracy*100))

	




