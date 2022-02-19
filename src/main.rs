use std::{env, fs::File, io::Error, path::Path, vec};

use pix::{hwb::SHwb8, Raster};
use rustfft::{num_complex::Complex, FftPlanner};

fn main() -> Result<(), Error> {
    let wav_filepath = env::args()
        .nth(1)
        .expect("first argument should be path to a .wav file");

    let chunk_size_in_seconds: f64 = env::args()
        .nth(2)
        .expect("second argument should be a duration")
        .parse()
        .expect("expected a number");

    // TODO - max the normalization global, maybe by doing a second pass algorithm
    // allow user to specify chunk overlap & a target interval
    let chunk_overlap: f64 = env::args()
        .nth(3)
        .expect("third argument should be a duration")
        .parse()
        .expect("expected a number");

    let mut wav_file = File::open(Path::new(&wav_filepath))?;

    let (header, data) = wav::read(&mut wav_file)?;

    // reusable iterator that will be for consecutive reads
    let data_reader: Vec<f64> = match header.bits_per_sample {
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
        _ => return Ok(eprintln!("was not matching bit size")),
    };

    eprintln!("wav_data_length: {}", data_reader.len());

    let chunk_size = (chunk_size_in_seconds * header.sampling_rate as f64) as usize;
    let chunk_count = data_reader.len() / chunk_size;
    let channel_count = header.channel_count as usize;

    eprintln!("chunk_size: {chunk_size}");
    eprintln!("{header:#?}");

    let fft = FftPlanner::new().plan_fft_forward(chunk_size);

    let width = chunk_count;
    let height = {
        // this removes the mirror frequencies on the higher range,
        // which is caused by the symmetry of the Real portion
        // of the Fast Fourier Transform
        let halved = chunk_size / 2;
        // we are gonna cut out some of the higher ranges that we dont want
        halved / 25
    };

    let mut map_data = vec![vec![0_f64; height]; width];
    let mut global_max = 1_f64;

    for chunk in 0..chunk_count {
        // reading the data for a single channel at a time and performing FFT on it
        let mut channel_buffers: Vec<Vec<_>> = (0..channel_count)
            .map(|channel| {
                let mut mapped: Vec<Complex<f64>> = data_reader[channel + chunk * chunk_size..]
                    .iter()
                    .step_by(channel_count)
                    .take(chunk_size)
                    // turn the numbers into Complex form for fft library
                    .map(Complex::from)
                    .collect();
                mapped.resize(chunk_size, Complex::from(0_f64));
                mapped
            })
            .collect();

        // use the first channel for now
        let buffer = &mut channel_buffers[0];
        fft.process(buffer);

        // Find the max in the pool in order to normalize the f64 values
        // ignore the DC component by skipping the 0th index (corresponding to no period)
        global_max = buffer[1..height]
            .iter()
            .map(|c| c.re.abs())
            .fold(1f64, f64::max);

        for y in 1..height {
            map_data[chunk][y] = buffer[y].re.abs();
        }
    }

    let mut raster = Raster::with_clear(width as u32, height as u32);
    // update the pixel value
    // Coordinate {
    //      x <-- current chunk
    //      y <-- current row
    // }
    for (x, row) in map_data.iter().enumerate() {
        for (y, value) in row.iter().enumerate() {
            // normalize the Complex FFT value and use it for the color
            let normalized = (u8::MAX as f64 * value / global_max) as u8;
            *raster.pixel_mut(x as i32, y as i32) =
                SHwb8::new(normalized / 2, normalized / 2, normalized);
        }
    }

    Ok(image::save_buffer(
        &Path::new("graph.png"),
        raster.as_u8_slice(),
        raster.width(),
        raster.height(),
        image::ColorType::Rgb8,
    )
    .expect("failed to parse raster to image."))
}
