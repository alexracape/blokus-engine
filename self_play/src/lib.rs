mod node;
mod simulation;

use pyo3::prelude::*;
use simulation::Config;
use simulation::{test_game, training_game, test_against_random};

/// Works with Pytorch model to generate self-play data
#[pyfunction]
fn play_training_game(
    id: i32,
    config: PyObject,
    inference_queue: PyObject,
    pipe: PyObject,
) -> PyResult<(Vec<(i32, i32)>, Vec<Vec<(i32, f32)>>, Vec<f32>)> {
    Python::with_gil(|py| {
        let config: Config = config.extract::<Config>(py).unwrap();
        let i_queue = inference_queue.bind(py);
        let r_queue = pipe.bind(py);

        match training_game(&config, i_queue, r_queue, id) {
            Ok(data) => Ok(data),
            Err(e) => {
                return Err(PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "{:?}",
                    e
                )))
            }
        }

    })
}

#[pyfunction]
fn play_test_against_random(
    id: i32,
    config: PyObject,
    inference_queue: PyObject,
    pipe: PyObject,
) -> PyResult<f32> {
    Python::with_gil(|py| {
        let config: Config = config.extract::<Config>(py).unwrap();
        let inference_queue = inference_queue.bind(py);
        let response_pipe = pipe.bind(py);

        match test_against_random(&config, id, inference_queue, response_pipe) {
            Ok(score) => Ok(score),
            Err(e) => {
                return Err(PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "{:?}",
                    e
                )))
            }
        }
    })
}

#[pyfunction]
fn play_test_game(
    id: i32,
    model_queue: PyObject,
    baseline_queue: PyObject,
    pipe: PyObject,
) -> PyResult<f32> {
    Python::with_gil(|py| {
        let model_queue = model_queue.bind(py);
        let baseline_queue = baseline_queue.bind(py);
        let response_pipe = pipe.bind(py);

        match test_game(id, model_queue, baseline_queue, response_pipe) {
            Ok(score) => Ok(score),
            Err(e) => {
                return Err(PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "{:?}",
                    e
                )))
            }
        }
    })
}

#[pymodule]
fn blokus_self_play(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m.add_function(wrap_pyfunction!(play_training_game, m)?);
    _ = m.add_function(wrap_pyfunction!(play_test_game, m)?);
    _ = m.add_function(wrap_pyfunction!(play_test_against_random, m)?);
    Ok(())
}
