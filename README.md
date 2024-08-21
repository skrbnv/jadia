## Jadia Diarization package

Kmeans-based fully open source (model code provided) package, good at parsing dialogs. Requires number of speakers to be provided. Fast in 'fast_fit' mode, when clusters are defined using only first slice of audio. 

### Install:
`pip install jadia`

`pip install jadia-plot` if you want to plot predictions

### Usage
```python
diarizer = Jadia(device=torch.device("cuda:0"),model="lite", batch_size=64)
segments = diarizer.process(FILENAME, num_voices=NUM_VOICES)
```
or 
```python
diarizer = Jadia(device=torch.device("cuda:0"), model="lite", batch_size=64)
diarizer.setup(num_voices=NUM_VOICES)
diarizer.load_audio(filename=FILENAME)
predictions = diarizer.predict()
segments = diarizer.predictions_to_segments(predictions)
```

Look into `eval.ipynb` notebook for plotting, metrics etc. 

### TODO:
- improved lite model
- heavier model with extra transformer layers
- fine-tuning
