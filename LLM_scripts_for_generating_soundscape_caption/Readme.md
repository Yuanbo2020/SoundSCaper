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
Soundscape caption: In this park, you may hear bird songs, occasional animal sounds, human activities, and snippets of speech. The overall atmosphere feels pleasant and calm, with a touch of natural tranquility despite some human presence.
```

```python
python LLM_GPT_soundscape_caption.py
-----------------------------------------------------------------------------------------------------------
Soundscape audio clip: fold_1_participant_00056_stimulus_13
Soundscape caption: In this park, you will hear Bird and Animal sounds, along with Human sounds and Speech. The atmosphere feels pleasant and calm, with a touch of liveliness from the diverse sounds. 
```

```python
python LLM_GPT_soundscape_caption.py
-----------------------------------------------------------------------------------------------------------
Soundscape audio clip: fold_1_participant_00056_stimulus_13
Soundscape caption: In this park, you will hear animal sounds, human sounds, speech, and birds, creating a lively environment. The background is filled with natural sounds, and occasionally you may hear the sounds of things and vehicles. The atmosphere is pleasant and calm, not chaotic, annoying, or monotonous. 
```

```python
python LLM_GPT_soundscape_caption.py
-----------------------------------------------------------------------------------------------------------
Soundscape audio clip: fold_1_participant_00056_stimulus_13
Soundscape caption: In this park, you hear a variety of sounds such as birds chirping, animals moving around, and distant human voices chatting. The atmosphere feels pleasant and calm, making it a serene and peaceful environment to be in. 
```

```python
python LLM_GPT_soundscape_caption.py
-----------------------------------------------------------------------------------------------------------
Soundscape audio clip: fold_1_participant_00056_stimulus_13
Soundscape caption: In this park, you will hear animal sounds, human sounds, and speech, creating a lively atmosphere. Birds chirping and the natural environment provide a soothing background. Some noise can be heard as well. Overall, the soundscape is pleasant and calm, with a touch of liveliness and nature's presence. 
```


