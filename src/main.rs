use std::{fs::File, io::Error, path::Path, process::Command, vec};

use clap::Parser;
use midly::{Format, Header, Timing, TrackEvent, TrackEventKind};
use pix::{hwb::SHwb8, Raster};
use rustfft::{num_complex::Complex, FftPlanner};

/// Command Line Arguments for processing a WAV data file
#[derive(Parser, Debug)]
struct ProgramArgs {
    /// the path to the target WAV file
    audio_filepath: String,
    /// number of seconds to convert into frames
    /// based on the sampling_rate of the file
    #[clap(long, short, default_value_t = 0.1)]
    time_step: f64,
    /// amount of overlap to inlude with other chunk frames
    #[clap(long, default_value_t = 0.1)]
    chunk_size_in_seconds: f64,
    /// amount of time in seconds to process frames
    #[clap(long, short)]
    duration: Option<f64>,
    /// start time in seconds for processing
    #[clap(long, short, default_value_t = 0.0)]
    start_time: f64,
}

fn main() -> Result<(), Error> {
    let args = ProgramArgs::parse();
    println!("{args:#?}");

    let wav_filepath = path_into_wav(&args.audio_filepath).unwrap();

    let (header, pcm_samples) = parse_wav(wav_filepath.to_str().unwrap())?;

    let (fft_map, width, height) = fft_transform(&pcm_samples, &header, &args)?;

    save_image(
        &fft_map,
        width,
        height / 10,
        wav_filepath.with_extension("png").to_str().unwrap(),
    );

    save_midi(wav_filepath.with_extension("midi").to_str().unwrap());

    Ok(())
}

fn parse_wav(input_file: &str) -> Result<(wav::Header, Vec<f64>), Error> {
    let (header, data) = wav::read(&mut File::open(input_file)?)?;

    let pcm_samples: Vec<f64> = match header.bits_per_sample {
        8 => data
            .try_into_eight()
            .unwrap()
            .into_iter()
            .map(|num| num as f64)
            .collect(),
        16 => data
            .try_into_sixteen()
            .unwrap()
            .into_iter()
            .map(|num| num as f64)
            .collect(),
        24 => data
            .try_into_twenty_four()
            .unwrap()
            .into_iter()
            .map(|num| num as f64)
            .collect(),
        32 => data
            .try_into_thirty_two_float()
            .unwrap()
            .into_iter()
            .map(|num| num as f64)
            .collect(),
        _ => panic!("was not matching bit size"),
    };

    assert!(!pcm_samples.is_empty(), "WAV file was empty");

    Ok((header, pcm_samples))
}

fn fft_transform(
    pcm_samples: &[f64],
    header: &wav::Header,
    args: &ProgramArgs,
) -> Result<(Vec<Vec<f64>>, usize, usize), Error> {
    let last_frame = pcm_samples.len() - 1;
    // Chunk Size = Number of frames (samples) to process at a time
    // Get the chunk size by multiplying the seconds by the sampling rate
    let chunk_size = (args.time_step * header.sampling_rate as f64) as usize;
    // Starting frame index
    // Do no frame to overflow past the length of the file
    let start_chunk = ((args.start_time * header.sampling_rate as f64) as usize).min(last_frame);
    // Ending frame index
    // Do no frame to overflow past the length of the file
    let end_chunk = args.duration.map_or_else(
        || last_frame,
        |duration| {
            let duration_chunk = (duration * header.sampling_rate as f64) as usize;
            (start_chunk + duration_chunk).min(last_frame)
        },
    );

    println!("{header:#?}");
    println!("sample_count:     {}", pcm_samples.len());
    println!("chunk_size:       {}", chunk_size);
    println!("start_chunk:      {}", start_chunk);
    println!("end_chunk:        {}", end_chunk);

    // dimensions of the resulting fft_map
    // (the extra + 1 is necessary for numbers that dont divide nicely?)
    let width = (end_chunk - start_chunk) / chunk_size + 1;
    // this removes the mirror frequencies on the higher range,
    // which is caused by the symmetry of the real portion of the Fast Fourier Transform
    let height = chunk_size / 2;

    // Vec to store FFT transformation after the buffer is overwritten
    let mut fft_map = vec![vec![0_f64; height]; width];

    // Setup FFT transformer
    let fft = FftPlanner::new().plan_fft_forward(chunk_size);

    for (x, chunk) in (start_chunk..end_chunk).step_by(chunk_size).enumerate() {
        // reading the data for a single channel at a time and performing FFT on it
        for channel in 0..header.channel_count as usize {
            let buffer = &mut pcm_samples[channel + chunk..]
                .iter()
                .step_by(header.channel_count as usize)
                .take(chunk_size)
                // turn the numbers into Complex form for fft library
                .map(Complex::from)
                .collect::<Vec<Complex<f64>>>();

            // make sure that there is padding on the buffer,
            // because the FFT expects a buffer of fixed length
            buffer.resize(chunk_size, Complex::from(0_f64));

            // FFT step
            fft.process(buffer);

            for (y, magnitude) in buffer
                .iter()
                .take(height)
                // ignore the DC component by skipping the 0th index (corresponding to no period)
                .skip(1)
                .map(|complex| complex.re.abs())
                .enumerate()
            {
                fft_map[x][y] += magnitude;
            }
        }
    }

    Ok((fft_map, width, height))
}

/// Renders the FFT values onto a Raster image so that you can visualize the resulting frequencies
/// This is the precursor to a MIDI translation of the data, as the image is easy to understand
/// but not concrete enough to make inferences about the notes/keys being concurrently pressed
fn save_image(fft_map: &[Vec<f64>], width: usize, height: usize, image_file: &str) {
    // Global max seen during transform for normalization purposes
    // Find the max in the pool in order to normalize the f64 values
    let signal_max = fft_map.iter().flatten().fold(f64::NAN, |acc, i| i.max(acc));

    let signal_amplifier: fn(f64) -> f64 = |n| (10_f64 * n).powi(2);

    // Generate image representing the heatmap
    let mut raster = Raster::with_clear(width as u32, height as u32);
    for (x, row) in fft_map.iter().enumerate().take_while(|&(x, _)| x < width) {
        for (y, sigal) in row.iter().enumerate().take_while(|&(y, _)| y < height) {
            // normalize the Complex FFT value and use it for the color
            //
            // NOTE! normalization is only applied/computed for the chunks covered during the tranformations,
            // this means that processing different portions withh yield different normalization ratios
            let normalized_signal = sigal / signal_max;
            let scale =
                (u8::MAX as f64 * signal_amplifier(normalized_signal)).min(u8::MAX as f64) as u8;
            // update the pixel value
            // Coordinate {
            //      x <-- current chunk
            //      y <-- current row
            // }
            *raster.pixel_mut(x as i32, y as i32) = SHwb8::new(scale / 2, scale / 2, scale);
        }
    }

    image::save_buffer(
        image_file,
        raster.as_u8_slice(),
        raster.width(),
        raster.height(),
        image::ColorType::Rgb8,
    )
    .expect("failed to parse raster to image.");
}

// TODO
fn save_midi(file_path: &str) {
    let midi_data = vec![vec![
        TrackEvent {
            delta: 0.into(),
            kind: TrackEventKind::Midi {
                channel: 0.into(),
                message: midly::MidiMessage::NoteOn {
                    vel: 200.into(),
                    key: 60.into(),
                },
            },
        },
        TrackEvent {
            delta: 5000.into(),
            kind: TrackEventKind::Midi {
                channel: 0.into(),
                message: midly::MidiMessage::NoteOn {
                    vel: 0.into(),
                    key: 60.into(),
                },
            },
        },
        TrackEvent {
            delta: 0.into(),
            kind: TrackEventKind::Meta(midly::MetaMessage::EndOfTrack),
        },
    ]];

    midly::write_std(
        &Header::new(Format::SingleTrack, Timing::Metrical(100.into())),
        &midi_data,
        &mut File::create(file_path).unwrap(),
    )
    .expect("failed to write midi.");
}

/// Convert a provided path string for an audio file into a Path struct.
///
/// If the path given is not already a WAV file then utilize ffmpeg from the User's environment in order to convert the file into WAV format.
/// This will only work for files that are still in some valid auido format, such as mp3.
///
/// ## Errors
/// None will be returned when the file does not have an extension.
fn path_into_wav(filepath: &str) -> Option<&Path> {
    let input_file = Path::new(filepath);

    match input_file.extension().unwrap().to_str()? {
        "wav" => Some(input_file),
        _ => {
            let wav_path = Path::new(Box::leak(Box::new(input_file.with_extension("wav"))));
            Command::new("ffmpeg")
            .arg("-i")
            .args([input_file, wav_path])
            .output().expect("could not use ffmpeg to convert input into .wav format. Check if ffmpeg is installed.");

            Some(wav_path)
        }
    }
}
