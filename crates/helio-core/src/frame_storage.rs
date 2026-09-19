//! Bounded, executor-owned storage for transient render descriptors.
//!
//! Render pass descriptors contain slices borrowed for the duration of command
//! recording.  Passes must not leak those slices to manufacture a `'static`
//! lifetime.  The executor clears this arena at the start of every frame and
//! keeps it alive until all command recording for that frame has completed.

use std::any::Any;

struct ErasedAllocation {
    ptr: *mut (),
    len: usize,
    drop_fn: unsafe fn(*mut (), usize),
}

impl ErasedAllocation {
    unsafe fn drop(self) {
        (self.drop_fn)(self.ptr, self.len);
    }
}

/// Transient allocations owned by a render executor for one frame.
///
/// Values are retained until [`Self::reset`] (normally the next frame). This
/// intentionally supports arbitrary descriptor element types while keeping
/// ownership in the executor rather than in a process-wide leak.
#[derive(Default)]
pub struct RenderFrameStorage {
    values: Vec<Box<dyn Any>>,
    allocations: Vec<ErasedAllocation>,
}

impl Drop for RenderFrameStorage {
    fn drop(&mut self) {
        self.reset();
    }
}

impl RenderFrameStorage {
    pub fn new() -> Self { Self::default() }

    /// Release all transient descriptor data and retain the allocation
    /// capacity for the next frame.
    pub fn reset(&mut self) {
        self.values.clear();
        for allocation in self.allocations.drain(..) {
            // SAFETY: each pointer was produced by Box::into_raw for the
            // matching monomorphized drop function below, and is dropped once.
            unsafe { allocation.drop() };
        }
    }

    /// Store an owned value and return a reference valid until the next reset.
    ///
    /// The erased value remains pinned in its box. The lifetime is tied to the
    /// storage borrow by the caller, so it cannot outlive the executor-owned
    /// frame arena without an explicit unsafe escape by the caller.
    pub fn retain<T: 'static>(&mut self, value: T) -> &T {
        self.values.push(Box::new(value));
        self.values
            .last()
            .and_then(|value| value.downcast_ref::<T>())
            .expect("just-retained frame storage value has the expected type")
    }

    pub fn len(&self) -> usize { self.values.len() }
    pub fn is_empty(&self) -> bool { self.values.is_empty() }

    /// Retain a boxed slice until the next frame reset and borrow it for the
    /// duration of this storage borrow. This is used for wgpu descriptors whose
    /// elements borrow frame-local texture views and therefore cannot satisfy
    /// the `'static` bound of [`Self::retain`].
    pub fn retain_boxed_slice<'a, T: 'a>(&'a mut self, value: Box<[T]>) -> &'a [T] {
        let len = value.len();
        let ptr = Box::into_raw(value) as *mut T;
        unsafe fn drop_allocation<T>(ptr: *mut (), len: usize) {
            // SAFETY: ptr/len came from Box<[T]>::into_raw and are consumed once.
            unsafe { drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr as *mut T, len))) };
        }
        self.allocations.push(ErasedAllocation { ptr: ptr.cast(), len, drop_fn: drop_allocation::<T> });
        // SAFETY: the allocation remains owned by self until reset, and the
        // returned borrow is tied to the mutable storage borrow.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}
