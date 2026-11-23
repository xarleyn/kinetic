//! `kinetic serve` as a container's PID 1, where the kernel drops SIGTERM
//! and SIGINT unless the process handles them, so `docker stop` would wait
//! out its timeout and kill it.

/// Ends the process on SIGTERM and SIGINT, as the default action does for
/// any process that is not PID 1. Warm Start writes each entry to a
/// temporary file and renames it into place, so a stop leaves no entry
/// half written.
#[cfg(unix)]
pub fn exit_on_stop() {
    extern "C" fn stop(signal: libc::c_int) {
        // SAFETY: _exit is async-signal-safe.
        unsafe { libc::_exit(128 + signal) }
    }
    let handler = stop as extern "C" fn(libc::c_int);
    for signal in [libc::SIGTERM, libc::SIGINT] {
        // SAFETY: the handler only calls _exit.
        unsafe { libc::signal(signal, handler as libc::sighandler_t) };
    }
}

/// Windows has no PID 1, and Ctrl+C already ends a console program.
#[cfg(not(unix))]
pub fn exit_on_stop() {}
