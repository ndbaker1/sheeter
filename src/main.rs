use std::{env, fs::File, io::Error, path::Path};

use pix::{hwb::SHwb8, Raster};
use rustfft::{num_complex::Complex, FftPlanner};

fn main() -> Result<(), Error> {
    let wav_filepath = env::args()
        .nth(1)
        .expect("first argument should be path to a .wav file");

    let duration: f64 = env::args()
        .nth(2)
        .expect("third argument should be a duration")
        .parse()
        .expect("expected a number");

    let mut wav_file = File::open(Path::new(&wav_filepath))?;

    let (header, data) = wav::read(&mut wav_file)?;

    let chunk_size = (duration * header.sampling_rate as f64) as usize;
    let chunk_count = header.sampling_rate as usize / chunk_size;
    let channel_count = header.channel_count as usize;

    eprintln!("chunk_size: {chunk_size}");
    eprintln!("{header:#?}");

    let fft = FftPlanner::new().plan_fft_forward(chunk_size);

    let width = chunk_count;
    let height = chunk_size;

    let mut raster = Raster::with_clear(width as u32, height as u32);

    if header.bits_per_sample == 8 {
        // reusable iterator that will be for consecutive reads
        let data_reader = data.try_into_eight().unwrap();
        eprintln!("wav_data_length: {}", data_reader.len());

        for chunk in 0..chunk_size {
            // reading the data for a single channel at a time and performing FFT on it
            let mut channel_buffers: Vec<Vec<_>> = (0..channel_count)
                .map(|channel| {
                    data_reader[channel..]
                        .iter()
                        .step_by(channel_count)
                        .take(chunk_size)
                        // Map smaller range integers to i64 to avoid overflows
                        .map(|num| *num as f64)
                        // turn the numbers into Complex form for fft library
                        .map(Complex::from)
                        .collect::<Vec<Complex<f64>>>()
                })
                .collect();

            // use the first channel for now
            let buffer = &mut channel_buffers[0];
            fft.process(buffer);
            // Find the max in the pool in order to normalize the f64 values
            // ignore the DC component by skipping the 0th index (corresponding to no period)
            let max_val = buffer[1..].iter().map(|c| c.re.abs()).fold(1f64, f64::max);

            for (y, value) in buffer[1..].iter().enumerate() {
                // normalize the Complex FFT value and use it for the color
                let color_value = (u8::MAX as f64 * value.re.abs() / max_val) as u8;
                // update the pixel value
                // Coordinate {
                //      x <-- current chunk
                //      y <-- current row
                // }
                *raster.pixel_mut(chunk as i32, y as i32) =
                    SHwb8::new(color_value / 2, color_value / 2, color_value);
            }
        }
    } else {
        eprintln!("was not matching bit size");
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
