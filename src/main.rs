use std::{fs::File, io::Error, path::Path, vec};

use clap::Parser;
use pix::{hwb::SHwb8, Raster};
use rustfft::{num_complex::Complex, FftPlanner};

/// Command Line Arguments for processing a WAV data file
#[derive(Parser, Debug)]
struct ProgramArgs {
    /// the path to the target WAV file
    wav_filepath: String,
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

    let signal_amplifier: fn(f64) -> f64 = match true {
        true => |n| (10_f64 * n).powi(2),
        false => |n| n,
    };

    let wav_file_path = Path::new(&args.wav_filepath);

    let (header, data) = {
        let mut wav_file = File::open(wav_file_path)?;
        wav::read(&mut wav_file)?
    };

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
        _ => {
            eprintln!("was not matching bit size");
            return Ok(());
        }
    };

    assert!(!pcm_samples.is_empty(), "WAV file was empty");

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

    // dimensions of the Raster
    let width = (end_chunk - start_chunk) / chunk_size + 1;
    let height = {
        // this removes the mirror frequencies on the higher range,
        // which is caused by the symmetry of the Real portion
        // of the Fast Fourier Transform
        let halved = chunk_size / 2;
        // we are gonna cut out some of the higher ranges that we dont want
        halved / 10
    };

    println!("output dimensions: {:?}", (width, height));

    // Vec to store FFT transformation after the buffer is overwritten
    let mut fft_map = vec![vec![0_f64; height]; width];
    // Global max seen during transform for normalization purposes
    let mut global_max = f64::NAN;

    // Setup FFT transformer
    let fft = FftPlanner::new().plan_fft_forward(chunk_size);

    for (x, chunk) in (start_chunk..end_chunk).step_by(chunk_size).enumerate() {
        // reading the data for a single channel at a time and performing FFT on it
        // TODO use all channels eventually
        for channel in 0..header.channel_count as usize {
            let mut buffer = pcm_samples[channel + chunk..]
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
            fft.process(&mut buffer);

            for (y, magnitude) in buffer
                .iter()
                .take(height)
                // ignore the DC component by skipping the 0th index (corresponding to no period)
                .skip(1)
                .map(|complex| complex.re.abs())
                .enumerate()
            {
                fft_map[x][y] += magnitude;
                // Find the max in the pool in order to normalize the f64 values
                global_max = global_max.max(fft_map[x][y]);
            }
        }
    }

    // we dont need the FFT transformer anymore
    drop(fft);

    let mut raster = Raster::with_clear(width as u32, height as u32);
    for (x, row) in fft_map.iter().enumerate() {
        for (y, value) in row.iter().enumerate() {
            // normalize the Complex FFT value and use it for the color
            //
            // NOTE! normalization is only applied/computed for the chunks covered during the tranformations,
            // this means that processing different portions withh yield different normalization ratios
            let normalized = value / global_max;
            let scale = (u8::MAX as f64 * signal_amplifier(normalized)).min(u8::MAX as f64) as u8;
            // update the pixel value
            // Coordinate {
            //      x <-- current chunk
            //      y <-- current row
            // }
            *raster.pixel_mut(x as i32, y as i32) = SHwb8::new(scale / 2, scale / 2, scale);
        }
    }

    image::save_buffer(
        &wav_file_path.with_extension("png"),
        raster.as_u8_slice(),
        raster.width(),
        raster.height(),
        image::ColorType::Rgb8,
    )
    .expect("failed to parse raster to image.");

    Ok(())
}
