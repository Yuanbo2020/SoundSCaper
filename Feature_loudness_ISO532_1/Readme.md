The script running *ISO_532-1.exe* loudness calculation from ISO 532-1:2017 standard. 

## Download

(Already placed in the folder **ISO_532_bin**)

The *ISO_532-1.exe* file can be downloaded from <a href="https://standards.iso.org/iso/532/-1/ed-1/en" target="_blank">here</a> and it needs to be placed into **ISO_532_bin** directory. 
The source code used to build the *ISO_532-1.exe* can be found on the same link.

## Requirements :

- Python **2.7** 
    - *numpy*
    - *subprocess*
    - *pandas* 
  
## Run the code

- Please place the audio clip files in '.wav' format to be processed into the **Dataset_wav** folder
 
- If the audio file is not in ".wav" format, please use the "convert_flac_to_wav.py" here to convert it.


## Demo:
```python 
python ISO_loudness.py
-----------------------------------------------------------------------------------------------------------
Processing the audio clip: fold_0_participant_10001_stimulus_02
 |- running ISO_532-1 loudness calculation
 |- getting loudness features
 |- saving loudness features
 |- removing temporary csv files
    |- removing Loudness.csv
    |- removing SpecLoudness.csv
Loudness extraction is done in 15.087 seconds.
```

 