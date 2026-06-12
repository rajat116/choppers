# Instructions:
### Step 1: Data Download
Download the "hls4ml_LHCjet_30p_train.tar.gz" from https://zenodo.org/records/3601436.

### Step 2: Process data
Run the `processing_data.ipynb` notebook to create the Jet images for each class, organized into .npy data files with 10k images each. Organize your repository as instructed at the bottom of the notebook.

### Step 3: Train VAEs
Run through the `training_VAEs.ipynb` notebook to train the VAE architectures for each preprocessing class of `log`, `truncated` and `scaled`. ROC curve data and associated plots will be automatically generated and saved. These can be further analyzed as desired. Pick well-performing models as the _teacher_ models for the next step. Suggested models are given at the end of the notebook.

### Step 4: Train the student models
Run through the `training_students.ipynb` notebook to train the NN student architectures for each teacher and each of the five anomaly scores as defined by Equations (2)-(6) in the paper. Again, ROC curve data and associated plots will be automatically generated and saved, which can be processed for further analysis as desired.

### Step 5: Train BDT students
```bash
python prepare_tmva_inputs_jetimages.py
python train_tmva_bdt_allAD_jetimages.py
python run_tmva_inference_multi_jetimages.py
```

### Step 6: Generate firmware configs (requires fwXmachina)
```bash
python export_fwX_configs_jetimages.py
```

### Step 7: Plot results
```bash
python plot_roc_all_signals_jetimages.py
python plot_roc_combined_signal.py
```

