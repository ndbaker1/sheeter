use std::{env, fs::File, io::Error, path::Path};

use rustfft::{num_complex::Complex, FftPlanner};

const FFT_LENGTH: usize = 2 ^ 12;

fn main() -> Result<(), Error> {
    let wav_filepath = env::args().nth(1).unwrap();
    let mut wav_file = File::open(Path::new(&wav_filepath))?;
    let (header, data) = wav::read(&mut wav_file)?;

    let fft = FftPlanner::new().plan_fft_forward(FFT_LENGTH);

    let mut buffer = vec![
        Complex {
            re: 0.0f32,
            im: 0.0f32
        };
        FFT_LENGTH
    ];
    fft.process(&mut buffer);

    Ok(println!("{buffer:?}"))
}
