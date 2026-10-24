//! Devices are created while other threads make Helio instances and exit, as
//! parallel tests do. With a GL backend in those instances, NVIDIA's Windows
//! driver deadlocks: an exiting thread that had a WGL context current holds
//! the loader lock and waits on the driver, while `vkCreateDevice` holds the
//! driver's lock and calls `LoadLibrary` (see
//! `helio_core::wgpu_instance_descriptor`). A deadlock can't be unwound, so
//! the watchdog aborts the process.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[test]
fn devices_are_created_while_instance_threads_exit() {
    let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };

    static CREATED: AtomicUsize = AtomicUsize::new(0);
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        std::thread::spawn(move || {
            let mut last = (0, Instant::now());
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(250));
                let created = CREATED.load(Ordering::Relaxed);
                if created != last.0 {
                    last = (created, Instant::now());
                } else if last.1.elapsed() > Duration::from_secs(30) {
                    // Past the test's output capture, and out without `exit`:
                    // it waits for the loader lock the deadlock holds.
                    let message = format!("device creation deadlocked after {created} devices\n");
                    let _ = std::io::Write::write_all(&mut std::io::stderr(), message.as_bytes());
                    std::process::abort();
                }
            }
        });
    }

    // Short-lived threads, each making an instance and finding an adapter.
    let exits: Vec<_> = (0..2)
        .map(|_| {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut exited = 0;
                while !stop.load(Ordering::Relaxed) {
                    std::thread::spawn(|| {
                        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
                        drop(pollster::block_on(instance.request_adapter(&Default::default())));
                    })
                    .join()
                    .unwrap();
                    exited += 1;
                }
                exited
            })
        })
        .collect();

    let start = Instant::now();
    while CREATED.load(Ordering::Relaxed) < 40 && start.elapsed() < Duration::from_secs(10) {
        let gpu = pollster::block_on(adapter.request_device(&Default::default()))
            .expect("device");
        drop(gpu);
        CREATED.fetch_add(1, Ordering::Relaxed);
    }
    stop.store(true, Ordering::Relaxed);
    let exited: usize = exits.into_iter().map(|t| t.join().unwrap()).sum();
    assert!(exited > 0, "no instance thread exited while devices were created");
}
