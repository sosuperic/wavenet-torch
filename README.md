# Torch implementation of Wavenet

## Main dependencies
- [Torchnet](https://github.com/torchnet/torchnet)

## Data and setup
First, create the directory structure by running
```
./setup.sh
```

Next, download the [VCTK corpus](http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz), unpack it, rename the folder to `vctk`, and place it in `data/` (i.e. `data/vctk/`). 

Then, create the training, validation, and test splits by running:
```
th preprocess.lua -fn split
```

These splits are created by splitting the wavs of each of the 109 people in the VCTK corpus. The current split settings are 98% training, 1% validation, 1% test, but these can be adjusted in `preprocess.lua`.

You can also easily create your own splits (e.g. to overfit while debugging) by creating a new folder in `data/processed/vctk/` and copying lines created by `th preprocess.lua -fn split` into `train.txt`, `valid.txt`, and `test.txt` files. For instance, I created split files in `data/processed/vctk/split_overfitonewav/`, each of which contain `225,data/vctk/wav48/p225/p225_008.wav,387179` (the speaker id, path to the wav file, and number of samples in the wav).

## Training
Example (see main.lua for flags):
```
th main.lua
```
If `-dont_save` flag is false, models, stats, and flags are saved to `models/<timestamp>/` every `save_model_every_epoch`.

## Generating using trained model
Example:
```
th main.lua -mode test -load_model_dir <timestamp> -load_model_name net_e10.t7
```

## TODO
- Local conditioning model (TTS)
- Global conditioning model (TTS + speakerid)
- Multi-GPU