use std::{env, fs::File, io::Error, path::Path};

use poloto::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};

fn main() -> Result<(), Error> {
    let wav_filepath = env::args().nth(1).unwrap();
    let mut wav_file = File::open(Path::new(&wav_filepath))?;
    let (header, data) = wav::read(&mut wav_file)?;

    println!("{header:#?}");

    let fft = FftPlanner::new().plan_fft_forward(header.sampling_rate as usize);

    if header.bits_per_sample == 16 {
        let data_reader = data.try_into_sixteen().unwrap();

        // reading the data for a single channel and performing FFT on it
        let mut buffer: Vec<Complex<i16>> = data_reader[..header.sampling_rate as usize]
            .into_iter()
            .map(Complex::from)
            .collect();

        fft.process(&mut buffer);

        println!("process graph");
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

        let mut plotter = data.build().plot("pitches", "x", "y");
        println!("{}", poloto::disp(|a| plotter.simple_theme(a)));
    } else {
        println!("was not matching bit size");
    }

    Ok(())
}
