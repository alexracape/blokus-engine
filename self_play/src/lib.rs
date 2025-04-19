mod node;
mod simulation;

use pyo3::prelude::*;
use simulation::Config;
use simulation::Runtime;

/// Works with Pytorch model to generate self-play data
#[pyfunction]
fn play_training_game(
    id: i32,
    config: PyObject,
    result_queue: PyObject,
    inference_queue: PyObject,
    pipe: PyObject,
) -> u8 {
    Python::with_gil(|py| {
        let config: Config = config.extract::<Config>(py).unwrap();
        let runtime = Runtime {
            config,
            id,
            result_queue: result_queue.bind(py),
            queue:  inference_queue.bind(py),
            pipe:   pipe.bind(py)
        };

        runtime.training_game()
    })
}

#[pyfunction]
fn play_test_against_random(
    id: i32,
    config: PyObject,
    result_queue: PyObject,
    inference_queue: PyObject,
    pipe: PyObject,
) -> u8 {
    Python::with_gil(|py| {
        let config: Config = config.extract::<Config>(py).unwrap();
        let runtime = Runtime {
            config,
            id,
            result_queue: result_queue.bind(py),
            queue:  inference_queue.bind(py),
            pipe:   pipe.bind(py)
        };

        runtime.test_against_random() 
    })
}

#[pyfunction]
fn play_test_game(
    id: i32,
    config: PyObject,
    result_queue: PyObject,
    model_queue: PyObject,
    baseline_queue: PyObject,
    pipe: PyObject,
) -> PyResult<f32> {
    Python::with_gil(|py| {
        let config: Config = config.extract::<Config>(py).unwrap();
        let model_queue = model_queue.bind(py);
        let baseline_queue = baseline_queue.bind(py);
        let mut runtime = Runtime {
            config,
            id,
            result_queue: result_queue.bind(py),
            queue:  model_queue,
            pipe:   pipe.bind(py)
        };

        match runtime.test_game(model_queue, baseline_queue) {
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
