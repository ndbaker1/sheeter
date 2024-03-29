# Sheeter
🎵 Music Transcriber

> Work has halted since solutions like Spotify's [Basic Pitch](https://github.com/spotify/basic-pitch) have already addressed this same topic

Process WAV format audio files into sheet music.
1. Send PCM (Pulse Code Modulation) data through a Fourier Transform to get frequency domain
2. Iterate through time slices in order to get a matrix of frequency to strength. `F: X * Y -> Z`
3. Normalize & amplify signal strengths in order to get a more clear activation threshhold for notes
4. Feed blocks into Neural Network Model to get MIDI notes
5. Export MIDI data & convert into Sheet Music

*several steps above can sought to be parallelized*

This is based on a [previous project of mine](https://github.com/ndbaker1/WAV-analyzer), extended to higher dimensionality and based on other projects/papers I have read in the subject before.

[![gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/from-referrer)

## Goals
- [x] be able to parse .wav data into fft and read pitches
- [x] visualize notes through colored images of the fft data
- [ ] convert fft data into MIDI format
- [ ] test MIDI to sheet music converter

## Resources
- https://github.com/philjonas/c-midi-writer
- https://github.com/LUMII-Syslab/RSE
- https://github.com/9552nZ/SmartSheetMusic
- https://colinraffel.com/projects/lmd/#get
