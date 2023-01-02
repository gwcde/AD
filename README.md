# AD
the detection of AD

Steps:
1. install pytorch and s3prl
2. download MADReSS-23-train.tgz from https://media.talkbank.org/dementia/English/0extra/
3. decompress MADReSS-23-train.tgz to ./MADReSS-23-train.tgz, `gzip -dc MADReSS-23-train.tgz | tar xf -` (other commands could also work)
4. run the notebook, and extract features. features are saved as npz. Note: it is very important to set the audio backend `torchaudio.set_audio_backend("soundfile")'
