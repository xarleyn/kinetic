//! Building the state a recurrent layer runs on must be possible while
//! another thread records a CUDA graph, as it is when two models share a
//! device and one records a step while the other opens a sequence.

// The shared harness holds more than this test uses; the others use it all.
#[allow(dead_code)]
mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use common::gpu;
use kinetic_cuda::{DeltaShape, DeltaState, DeviceBuffer, Stream, MAX_BATCH};

/// A reference a thread can carry. The driver's context is safe from
/// several threads and each stream below belongs to one thread, as the
/// seats of two models do — the same justification the server's backends
/// carry for sending theirs.
struct OnThread<T>(T);
// SAFETY: the wrapped reference is used by one thread at a time, and the
// context it reaches is the driver's own thread-safe one.
unsafe impl<T> Send for OnThread<T> {}

/// Takes what a wrapper carried. A call uses the whole wrapper, so a
/// thread's closure captures the wrapper, which is Send, and not the
/// reference inside it, which is not.
fn carried<T>(wrapped: OnThread<T>) -> T {
    wrapped.0
}

/// Waits for `what` for as long as a slow device may take.
fn await_flag(what: &str, flag: &AtomicBool) {
    for _ in 0..6_000 {
        if flag.load(Ordering::SeqCst) {
            return;
        }
        thread::sleep(Duration::from_millis(5));
    }
    panic!("{what}");
}

#[test]
fn state_builds_while_another_thread_records() {
    let Some(gpu) = gpu() else { return };
    let shape = DeltaShape {
        n_k_heads: 2,
        n_v_heads: 4,
        head_k_dim: 128,
        head_v_dim: 128,
        conv_taps: 4,
        eps: 1e-6,
    };
    // A second stream, as a second model on the device has.
    let other = Stream::new(&gpu.ctx).unwrap();
    // Allocated before any recording: a graph must not allocate.
    let mut keeper = DeviceBuffer::<f32>::new(&gpu.ctx, 8).unwrap();
    let recording = AtomicBool::new(false);
    let built = AtomicBool::new(false);

    let ctx = OnThread(&gpu.ctx);
    let record_stream = OnThread(&gpu.stream);
    let build_stream = OnThread(&other);
    let keeper_ref = OnThread(&mut keeper);
    let shape_ref = OnThread(&shape);
    let recording_ref = OnThread(&recording);
    let recording_gate = OnThread(&recording);
    let built_ref = OnThread(&built);
    let built_gate = OnThread(&built);

    thread::scope(move |scope| {
        let recorder = scope.spawn(move || {
            let stream = carried(record_stream);
            let keeper = carried(keeper_ref);
            let recording = carried(recording_ref);
            let built = carried(built_gate);
            let graph = stream
                .capture(|| {
                    // The recording is open from here on.
                    recording.store(true, Ordering::SeqCst);
                    // Keeps the graph from being empty; a recorded memset.
                    keeper.zero_on(stream)?;
                    await_flag("the state was never built", built);
                    Ok(())
                })
                .unwrap();
            drop(graph);
        });
        let builder = scope.spawn(move || {
            let ctx = carried(ctx);
            let stream = carried(build_stream);
            let shape = carried(shape_ref);
            let recording = carried(recording_gate);
            await_flag("the recording never opened", recording);
            // With the zeros once queued on the legacy default stream,
            // this failed with CUDA_ERROR_STREAM_CAPTURE_IMPLICIT.
            let state = DeltaState::new(ctx, stream, shape, MAX_BATCH);
            let built = carried(built_ref);
            built.store(true, Ordering::SeqCst);
            state.unwrap();
        });
        builder.join().unwrap();
        recorder.join().unwrap();
    });
}
