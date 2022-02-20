# Sheeter
🎵 Music Transcriber

Process WAV format audio files into sheet music.
1. Send PCM (Pulse Code Modulation) data through a Fourier Transform to get frequency domain
2. Iterate through time slices in order to get a matrix of frequency to strength. `F: X * Y -> Z`
3. Normalize & amplify signal strengths in order to get a more clear activation threshhold for notes
4. Feed blocks into Neural Network Model to get MIDI notes
5. Export MIDI data & convert into Sheet Music

*several steps above can sought to be parallelized*

This is based on a [previous project of mine](https://github.com/ndbaker1/WAV-analyzer), extended to higher dimensionality.

[![gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/from-referrer)

## Goals
- [x] be able to parse .wav data into fft and read pitches
- [x] visualize notes through colored images of the fft data
- [ ] convert fft data into MIDI format
- [ ] test MIDI to sheet music converter

## References
- https://github.com/philjonas/c-midi-writer
- https://github.com/LUMII-Syslab/RSE