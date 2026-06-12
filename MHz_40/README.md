### Step 1:
download the data from Zenodo: 
background data: 
```
https://zenodo.org/records/5046389
```
signals:
```
https://zenodo.org/records/5061688
http://zenodo.org/records/5061633
https://zenodo.org/records/5046446
https://zenodo.org/records/5055454
```
### Step 2:
Train the VAE (`teacher_training.ipynb`). This notebook also includes some evals and plots.

### Step 3:
Train the NN students (`nn_students.ipynb`). This notebook also includes some evals and plots.

### Step 4: Train BDT students
```bash
python train_tmva_bdt_allAD.py
python run_tmva_inference_multi.py
```

### Step 5: Generate firmware configs (requires fwXmachina)
```bash
python export_fwX_configs_40MHz.py
```

### Step 6: Plot results and compute uncertainties
```bash
python plot_roc_paper_40MHz.py
python compute_efficiency_uncertainties.py
```
