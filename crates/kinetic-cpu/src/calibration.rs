//! Bounded, thread-local input capture for offline experiments.
//!
//! Only compiled with the `calibration` feature. Targets match matrix byte
//! addresses and shapes; addresses are compared, never dereferenced. Keep
//! the corresponding weight storage alive throughout [`capture`]. Calls on
//! other threads are deliberately outside this capture's scope.

use std::cell::RefCell;

use kinetic_quant::Matrix;

#[derive(Clone, Copy)]
pub struct Target {
    address: usize,
    bytes: usize,
    rows: usize,
    cols: usize,
}

impl Target {
    pub fn new(matrix: &Matrix<'_>) -> Self {
        Self {
            address: matrix.data.as_ptr() as usize,
            bytes: matrix.data.len(),
            rows: matrix.rows,
            cols: matrix.cols,
        }
    }

    fn matches(&self, matrix: &Matrix<'_>) -> bool {
        self.address == matrix.data.as_ptr() as usize
            && self.bytes == matrix.data.len()
            && self.rows == matrix.rows
            && self.cols == matrix.cols
    }
}

pub struct Inputs {
    pub cols: usize,
    pub calls: usize,
    pub seen_rows: usize,
    /// A prefix of real input rows, bounded by `max_rows`.
    pub values: Vec<f32>,
}

struct Trace {
    targets: Vec<Target>,
    inputs: Vec<Inputs>,
    max_rows: usize,
}

thread_local! {
    static TRACE: RefCell<Option<Trace>> = const { RefCell::new(None) };
}

struct Reset;
impl Drop for Reset {
    fn drop(&mut self) {
        TRACE.with(|slot| *slot.borrow_mut() = None);
    }
}

/// Runs `run` normally and returns the selected matrices' real inputs.
/// A panic clears capture state. Nested captures are rejected. There is
/// no file I/O, environment switch, callback, or change to arithmetic.
pub fn capture<R>(
    targets: Vec<Target>,
    max_rows: usize,
    run: impl FnOnce() -> R,
) -> (R, Vec<Inputs>) {
    assert!(max_rows > 0, "capture row bound must be positive");
    let inputs = targets
        .iter()
        .map(|t| Inputs {
            cols: t.cols,
            calls: 0,
            seen_rows: 0,
            values: Vec::new(),
        })
        .collect();
    TRACE.with(|slot| {
        let mut slot = slot.borrow_mut();
        assert!(slot.is_none(), "nested calibration capture");
        *slot = Some(Trace {
            targets,
            inputs,
            max_rows,
        });
    });
    let reset = Reset;
    let result = run();
    let inputs = TRACE.with(|slot| slot.borrow_mut().take().expect("active capture").inputs);
    drop(reset);
    (result, inputs)
}

pub(crate) fn observe(matrices: &[&Matrix<'_>], x: &[f32]) {
    TRACE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(trace) = slot.as_mut() else { return };
        for (target, input) in trace.targets.iter().zip(&mut trace.inputs) {
            if matrices.iter().any(|matrix| target.matches(matrix)) {
                input.calls += 1;
                input.seen_rows += x.len() / input.cols;
                let room = trace.max_rows * input.cols - input.values.len();
                input.values.extend_from_slice(&x[..x.len().min(room)]);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinetic_quant::TensorType;

    #[test]
    fn captures_shared_input_once_without_changing_output_and_obeys_limit() {
        let bytes: Vec<u8> = [1f32, 2., 3., 4.]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let other = bytes.clone();
        let a = Matrix::new(TensorType::F32, &bytes, 2, 2);
        let b = Matrix::new(TensorType::F32, &other, 2, 2);
        let x = [2., 3., 4., 5.];
        let mut expected = [0.; 4];
        crate::matmul(&a, &x, &mut expected);
        let (actual, captured) = capture(vec![Target::new(&a)], 1, || {
            let (mut ya, mut yb) = ([0.; 4], [0.; 4]);
            crate::matmul_many(&[&a, &b], &x, &mut [&mut ya, &mut yb]);
            assert_eq!(ya, yb);
            ya
        });
        assert_eq!(expected, actual);
        assert_eq!(captured[0].values, x[..2]);
        assert_eq!((captured[0].calls, captured[0].seen_rows), (1, 2));
        let (_, empty) = capture(vec![Target::new(&b)], 10, || {
            crate::matmul(&a, &x, &mut [0.; 4])
        });
        assert!(
            empty[0].values.is_empty(),
            "different storage must not match"
        );
    }

    #[test]
    fn panic_clears_capture_and_other_threads_are_excluded() {
        let bytes = 1f32.to_le_bytes();
        let a = Matrix::new(TensorType::F32, &bytes, 1, 1);
        assert!(
            std::panic::catch_unwind(|| capture(vec![Target::new(&a)], 1, || panic!("fixture")))
                .is_err()
        );
        let (_, inputs) = capture(vec![Target::new(&a)], 1, || {
            std::thread::scope(|s| {
                s.spawn(|| crate::matmul(&a, &[2.], &mut [0.]))
                    .join()
                    .unwrap()
            });
            crate::matmul(&a, &[3.], &mut [0.]);
        });
        assert_eq!(inputs[0].values, [3.]);
    }
}
