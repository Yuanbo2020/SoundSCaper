# Generate soundscape captions using generic LLM

## 1. Data preparation

- Place the matrix file of audio event probabilities predicted by the SoundAQnet into the `SoundAQnet_event_probability` directory

- Place the SoundAQnet prediction file, including the predicted acoustic scene label, ISOP value, ISOE value, and the PAQ 8D AR values, into the `SoundAQnet_scene_ISOPl_ISOEv_PAQ8DARs` directory

## 2. Generate soundscape caption

- Replace the "YOUR_API_KEY_HERE" in line 26 of the `LLM_GPT_soundscape_caption.py` file with your OpenAI API key

- Run `LLM_GPT_soundscape_caption.py`
   

## 3. Demonstration

```python
python LLM_GPT_soundscape_caption.py
-----------------------------------------------------------------------------------------------------------
Soundscape audio clip: fold_1_participant_00056_stimulus_13
Soundscape caption: In this park, you will hear Bird and Animal sounds, with occasional Human sounds and Speech in the background. The atmosphere feels pleasant and calm, perfect for relaxation and enjoying nature.
```

Please note that the OpenAI temperature parameter can control the randomness of the output. Temperature is a number between 0 and 2, with a default value of 1.
When you set it higher, you'll get more random outputs. When you set it lower, towards 0, the values are more deterministic.
 
The default temperature value is used in the script so that the output may vary slightly.
  
```python
python LLM_GPT_soundscape_caption.py
-----------------------------------------------------------------------------------------------------------
Soundscape audio clip: fold_1_participant_00056_stimulus_13
Soundscape caption: In this park, you will hear Bird and Animal sounds, with occasional Human sounds and Speech in the background. The atmosphere feels pleasant and calm, perfect for relaxation and enjoying nature.
```



