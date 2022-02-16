use std::{env, fs::File, io::Error, path::Path};

use poloto::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};

fn main() -> Result<(), Error> {
    let wav_filepath = env::args()
        .nth(1)
        .expect("first argument should be path to a .wav file");

    let start_time: f64 = env::args()
        .nth(2)
        .expect("second argument should be start time")
        .parse()
        .expect("expected a number");

    let duration: f64 = env::args()
        .nth(3)
        .expect("third argument should be a duration")
        .parse()
        .expect("expected a number");

    let mut wav_file = File::open(Path::new(&wav_filepath))?;

    let (header, data) = wav::read(&mut wav_file)?;

    let start_time = (start_time * header.sampling_rate as f64) as usize;
    let chunk_size = (duration * header.sampling_rate as f64) as usize;
    let channel_count = header.channel_count as usize;

    eprintln!("{header:#?}");

    let fft = FftPlanner::new().plan_fft_forward(chunk_size);

    if header.bits_per_sample == 16 {
        // reusable iterator that will be for consecutive reads
        let data_reader = data.try_into_sixteen().unwrap().into_iter();
        eprintln!("wav_data_length: {}", data_reader.len());

        // reading the data for a single channel at a time and performing FFT on it
        let mut channel_buffers: Vec<Vec<_>> = (0..channel_count)
            .map(|channel| {
                data_reader
                    .clone()
                    .skip(start_time + channel)
                    .step_by(channel_count)
                    .take(chunk_size)
                    // Map smaller range integers to i64 to avoid overflows
                    .map(|num| num as i64)
                    // turn the numbers into Complex form for fft library
                    .map(Complex::from)
                    .collect::<Vec<Complex<i64>>>()
            })
            .collect();

        eprintln!("buffer collection");

        for (channel, buffer) in &mut channel_buffers.iter_mut().enumerate() {
            fft.process(buffer);

            for (i, a) in buffer
                .iter()
                // ignore the symmetric portion on the left
                .skip(buffer.len() / 2)
                // ignore the 0Hz DC signal
                .skip(1)
                .enumerate()
            {
                if a.re.abs() > chunk_size as i64 / 4 {
                    eprintln!("{:#?}", i);
                }
            }

            eprintln!("graph rendering");

            // Graph Generation Portion
            let mut data = poloto::data();
            data.line(
                "Hz",
                buffer
                    .iter()
                    // ignore the symmetric portion on the left
                    .skip(buffer.len() / 2)
                    // ignore the 0Hz DC signal
                    .skip(1)
                    .enumerate()
                    .map(|(i, n)| [i as f64, n.re.abs() as f64]),
            );

            let mut plotter = data.build().plot(format!("chanel {}", channel), "x", "y");
            println!("{}", poloto::disp(|a| plotter.simple_theme(a)));
        }
    } else {
        eprintln!("was not matching bit size");
    }

    Ok(())
}
