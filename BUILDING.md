
# Running and evaluating the model

In order to train and/or evaluate the model, begin by setting up the environment.

```bash
python -m venv .
source bin/activate
pip install -r requirements.txt
```

In order to point to the appropriate paths for training and testing. Set up your .env file. Below is my .env file.

```text
video_global_path=../v-Strykathon-PS2/PS2 Train/
video_test_path=../v-Strykathon-PS2/PS2 Test/
best_model_path=Trainer/weights/run_2/model21.pth
```

The .env file lives in the root directory of the repository.

Once set up, in order to start training, create the `weights\run_1` or corresponding save path corresponding to the model save path on line 225 of `train.py` or line 233 of `train_graphs.py`.
