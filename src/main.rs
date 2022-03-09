use std::{fs::File, io::Error, path::Path, process::Command, vec};

use clap::Parser;
use midly::{Format, Header, Timing, TrackEvent, TrackEventKind};
use pix::{hwb::SHwb8, Raster};
use rustfft::{num_complex::Complex, FftPlanner};

#[derive(Parser, Debug)]
struct ProgramArgs {
    /// the path to the target audio file
    audio_filepath: String,
    /// start time in seconds for processing
    #[clap(long, short, default_value_t = 0.0)]
    start_time: f64,
    /// number of seconds to convert into frames
    /// based on the sampling_rate of the file
    #[clap(long, short, default_value_t = 0.1)]
    time_step: f64,
    /// amount of overlap to inlude with other chunk frames.
    /// If no value is provided then it will default to be equal to the timestep
    #[clap(long, short)]
    chunk_step: Option<f64>,
    /// amount of time in seconds to process frames
    #[clap(long, short)]
    duration: Option<f64>,
}

impl ProgramArgs {
    fn get_chunk_step(&self) -> f64 {
        self.chunk_step.unwrap_or_else(|| self.time_step)
    }
}

fn main() -> Result<(), Error> {
    let args = ProgramArgs::parse();
    println!("{args:#?}");

    let wav_filepath = path_into_wav(Path::new(&args.audio_filepath)).unwrap();

    let (header, pcm_samples) = parse_wav(wav_filepath.to_str().unwrap())?;

    let (mut fft_map, width, height) = fft_transform(&pcm_samples, &header, &args)?;

    amplify_and_normalize(&mut fft_map, None);

    save_image(
        &fft_map,
        width,
        height / 10,
        wav_filepath.with_extension("png").to_str().unwrap(),
    );

    save_midi(
        &fft_map,
        wav_filepath.with_extension("midi").to_str().unwrap(),
    );

    Ok(())
}

/// Optionally perform an element-wise transformation on the fft data
/// in order to modify the strength of particular frequencies.
/// Next, normalize the data based on the max of the data points in the map.
fn amplify_and_normalize(fft_map: &mut [Vec<f64>], optional_amplifier: Option<fn(&f64) -> f64>) {
    if let Some(amplifier) = optional_amplifier {
        fft_map
            .iter_mut()
            .flatten()
            .for_each(|val| *val = amplifier(val));
    }

    let signal_max = fft_map.iter().flatten().fold(f64::NAN, |p, i| i.max(p));

    fft_map
        .iter_mut()
        .flatten()
        .for_each(|val| *val /= signal_max);
}

/// Extracts the Header and PCM data from a WAV format audio file.
/// PCM data is converted into f64 such that the FFT becomes easier to handle
/// overflow and normalizing operations
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

/// Returns the 2D matrix of FFT data, along with the dimensions of the data
///
/// ## Notes
/// - half of the FFT data per-timeslice is discarded as it is a mirror of the first half.
/// - data points are additive per channel, meaning that frequencies occuring in both channels will appear stronger
fn fft_transform(
    pcm_samples: &[f64],
    header: &wav::Header,
    args: &ProgramArgs,
) -> Result<(Vec<Vec<f64>>, usize, usize), Error> {
    let last_frame = pcm_samples.len() - 1;
    // number of frames to skip for each column in the transform map
    let step_size = (args.time_step * header.sampling_rate as f64) as usize;
    // Starting frame index
    let start_chunk = ((args.start_time * header.sampling_rate as f64) as usize).min(last_frame);
    // Ending frame index
    let end_chunk = args.duration.map_or_else(
        || last_frame,
        |seconds| {
            let duration_chunk = (seconds * header.sampling_rate as f64) as usize;
            (start_chunk + duration_chunk).min(last_frame)
        },
    );
    // the amount of frames to process for each time step
    let read_chunk_size = (args.get_chunk_step() * header.sampling_rate as f64) as usize;

    println!("{header:#?}");
    println!("sample_count:     {}", pcm_samples.len());
    println!("chunk_size:       {}", step_size);
    println!("start_chunk:      {}", start_chunk);
    println!("read_chunk_size:  {}", read_chunk_size);
    println!("end_chunk:        {}", end_chunk);

    // dimensions of the resulting fft_map
    // (the extra + 1 is necessary for numbers that dont divide nicely?)
    let width = (end_chunk - start_chunk) / step_size + 1;
    // this removes the mirror frequencies on the higher range,
    // which is caused by the symmetry of the real portion of the Fast Fourier Transform
    let height = read_chunk_size / 2;

    // Vec to store FFT transformation after the buffer is overwritten
    let mut fft_map = vec![vec![0_f64; height]; width];

    // Setup FFT transformer
    let fft = FftPlanner::new().plan_fft_forward(read_chunk_size);

    for (x, chunk_start) in (start_chunk..end_chunk).step_by(step_size).enumerate() {
        // reading the data for a single channel at a time and performing FFT on it
        for channel in 0..header.channel_count as usize {
            let buffer = &mut pcm_samples[channel
                + chunk_start
                    .saturating_add(read_chunk_size / 2)
                    .max(0)
                    .min(last_frame)..]
                .iter()
                .step_by(header.channel_count as usize)
                .take(read_chunk_size)
                // turn the numbers into Complex form for fft library
                .map(Complex::from)
                .collect::<Vec<Complex<f64>>>();

            // make sure that there is padding on the buffer,
            // because the FFT expects a buffer of fixed length
            // !!! This behavior is still not right because it needs to pad left or right depending on if it is at the beginning or end of the file
            buffer.resize(read_chunk_size, Complex::from(0_f64));

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

    // Generate image representing the heatmap
    let mut raster = Raster::with_clear(width as u32, height as u32);
    for (x, row) in fft_map.iter().enumerate().take_while(|&(x, _)| x < width) {
        for (y, sigal) in row.iter().enumerate().take_while(|&(y, _)| y < height) {
            // normalize the Complex FFT value and use it for the color
            //
            // NOTE! normalization is only applied/computed for the chunks covered during the tranformations,
            // this means that processing different portions withh yield different normalization ratios
            let normalized_signal = sigal / signal_max;
            let scale = (u8::MAX as f64 * normalized_signal).min(u8::MAX as f64) as u8;
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

/// Save the fft_data into a MIDI format
///
/// ## Operations
/// 1. interpret FFT data as notes with start & stop times
/// 2. sorts objects by time to be used as delta in MIDI standard
/// 3. parses list into TrackEvents
fn save_midi(fft_data: &[Vec<f64>], file_path: &str) {
    let mut midi_data: Vec<TrackEvent> = vec![];
    // generate_notes(fft_data)
    //     .into_iter()
    //     .map(|a| TrackEvent {
    //         delta: (time as u32).into(),
    //         kind: TrackEventKind::Midi {
    //             channel: 0.into(),
    //             message: midly::MidiMessage::NoteOn {
    //                 vel: match order {
    //                     0 => 90.into(),
    //                     1 => 0.into(),
    //                     _ => 0.into(),
    //                 },
    //                 key: note.into(),
    //             },
    //         },
    //     })
    //     .collect();

    // append the end of track message
    midi_data.push(TrackEvent {
        delta: 0.into(),
        kind: TrackEventKind::Meta(midly::MetaMessage::EndOfTrack),
    });

    midly::write_std(
        &Header::new(Format::SingleTrack, Timing::Metrical(100.into())),
        &[midi_data],
        &mut File::create(file_path).unwrap(),
    )
    .expect("failed to write midi.");
}

/// Returns a set of notes that contain the start and end time
///
/// maybe this is a neural network
fn generate_notes(fft_data: &[Vec<f64>]) -> Vec<u128> {
    let mut notes = vec![];

    for time_slice in fft_data {
        // convolute time_slice of height of X into 88
        // compute note vector from previous timeslice + new data
        let note_vec = vec![1_f64; 88];

        let note_bitmap = note_vec
            .into_iter()
            .enumerate()
            .filter(|(_, note)| *note > 0.5)
            .fold(0_u128, |note_bitmap, (i, _)| note_bitmap | 1 << i);

        notes.push(note_bitmap);
    }

    notes
}

/// Convert a provided path string for an audio file into a Path struct.
///
/// If the path given is not already a WAV file then utilize `ffmpeg` from the User's environment
/// in order to convert the file into WAV format.
///
/// ## Errors
/// - This will only work for files that are still in some valid auido format, such as mp3.
/// - None will be returned when the file does not have an extension.
fn path_into_wav(filepath: &Path) -> Option<&Path> {
    let wav_extension_file = Box::leak(Box::new(filepath.with_extension("wav")));

    if !wav_extension_file.exists() {
        println!(
            "attempting to use locally installed ffmpeg to convert {:?} to {:?}",
            filepath.file_name()?,
            wav_extension_file.file_name()?,
        );
        Command::new("ffmpeg")
            .arg("-i")
            .args([filepath, wav_extension_file])
            .output().expect("could not use ffmpeg to convert input into .wav format. Check if ffmpeg is installed.");
    }

    return Some(wav_extension_file);
}
