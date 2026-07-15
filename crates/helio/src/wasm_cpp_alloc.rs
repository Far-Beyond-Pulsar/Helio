//! C++ allocation ABI bridge for `meshopt` on `wasm32-unknown-unknown`.
//!
//! `meshopt` compiles meshoptimizer's C++ sources without a C++ standard
//! library for this target. Its default allocator still references the C++
//! scalar new/delete symbols, so route those calls through Rust's global
//! allocator. The small header makes unsized delete recover the allocation
//! layout without relying on allocator-specific usable-size APIs.

use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};

const ALIGN: usize = 16;
const HEADER_SIZE: usize = ALIGN;

#[inline]
unsafe fn cpp_alloc(size: usize) -> *mut u8 {
    let allocation_size = size
        .max(1)
        .checked_add(HEADER_SIZE)
        .unwrap_or_else(|| handle_alloc_error(Layout::new::<usize>()));
    let layout = Layout::from_size_align_unchecked(allocation_size, ALIGN);
    let base = alloc(layout);
    if base.is_null() {
        handle_alloc_error(layout);
    }
    base.cast::<usize>().write(allocation_size);
    base.add(HEADER_SIZE)
}

#[inline]
unsafe fn cpp_dealloc(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    let base = ptr.sub(HEADER_SIZE);
    let allocation_size = base.cast::<usize>().read();
    let layout = Layout::from_size_align_unchecked(allocation_size, ALIGN);
    dealloc(base, layout);
}

// Itanium C++ ABI symbols used by clang for wasm32. Keep scalar, array, and
// sized delete variants so compiler optimization does not alter linkability.

#[no_mangle]
unsafe extern "C" fn _Znwm(size: usize) -> *mut u8 {
    cpp_alloc(size)
}

#[no_mangle]
unsafe extern "C" fn _ZdlPv(ptr: *mut u8) {
    cpp_dealloc(ptr);
}

#[no_mangle]
unsafe extern "C" fn _Znam(size: usize) -> *mut u8 {
    cpp_alloc(size)
}

#[no_mangle]
unsafe extern "C" fn _ZdaPv(ptr: *mut u8) {
    cpp_dealloc(ptr);
}

#[no_mangle]
unsafe extern "C" fn _ZdlPvm(ptr: *mut u8, _size: usize) {
    cpp_dealloc(ptr);
}

#[no_mangle]
unsafe extern "C" fn _ZdaPvm(ptr: *mut u8, _size: usize) {
    cpp_dealloc(ptr);
}
