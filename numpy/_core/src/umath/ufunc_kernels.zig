const std = @import("std");

const npy_intp = isize;
const npy_double = f64;

fn loadPtr(comptime T: type, p: [*]u8) *align(1) T {
    return @ptrCast(p);
}

fn ptrAdd(p: [*]u8, offset: npy_intp) [*]u8 {
    // strides can be negative, so do wrapping address arithmetic.
    const addr = @intFromPtr(p);
    const off: usize = @bitCast(offset);
    return @ptrFromInt(addr +% off);
}

fn asSteps(comptime N: usize, steps: [*c]const npy_intp) *const [N]npy_intp {
    return @ptrCast(steps);
}

fn asDims2(dims: [*c]const npy_intp) [2]usize {
    std.debug.assert(dims[0] >= 0);
    std.debug.assert(dims[1] >= 0);
    return .{ @intCast(dims[0]), @intCast(dims[1]) };
}

/// addition with overflow wrapping for integer types.
inline fn add(comptime T: type, a: T, b: T) T {
    if (comptime @typeInfo(T) == .int) return a +% b;
    return a + b;
}

/// multiplication with overflow wrapping for integer types.
inline fn mul(comptime T: type, a: T, b: T) T {
    if (comptime @typeInfo(T) == .int) return a *% b;
    return a * b;
}

inline fn mulPtr(comptime T: type, p1: [*]u8, p2: [*]u8) T {
    return mul(T, loadPtr(T, p1).*, loadPtr(T, p2).*);
}

// gufunc kernels

/// signature: (i),(i)->()
fn gufunc_inner1d(
    comptime T: type,
    args: [*c][*c]u8,
    dims: [*c]const npy_intp,
    steps: [*c]const npy_intp,
) void {
    const dN, const di = asDims2(dims);
    const s0, const s1, const s2, const is1, const is2 = asSteps(5, steps).*;
    var a0, var a1, var a2 = .{ args[0], args[1], args[2] };

    for (0..dN) |_| {
        var ip1, var ip2 = .{ a0, a1 };
        var sum: T = 0;

        for (0..di) |_| {
            sum = add(T, sum, mulPtr(T, ip1, ip2));
            ip1, ip2 = .{ ptrAdd(ip1, is1), ptrAdd(ip2, is2) };
        }

        loadPtr(T, a2).* = sum;
        a0, a1, a2 = .{ ptrAdd(a0, s0), ptrAdd(a1, s1), ptrAdd(a2, s2) };
    }
}

/// signature: (i),(i),(i)->()
fn gufunc_innerwt(
    comptime T: type,
    args: [*c][*c]u8,
    dims: [*c]const npy_intp,
    steps: [*c]const npy_intp,
) void {
    const dN, const di = asDims2(dims);
    const s0, const s1, const s2, const s3, const is1, const is2, const is3 = asSteps(7, steps).*;
    var a0, var a1, var a2, var a3 = .{ args[0], args[1], args[2], args[3] };

    for (0..dN) |_| {
        var ip1, var ip2, var ip3 = .{ a0, a1, a2 };
        var sum: T = 0;

        for (0..di) |_| {
            const v12 = mulPtr(T, ip1, ip2);
            const v3 = loadPtr(T, ip3).*;
            sum = add(T, sum, mul(T, v12, v3));
            ip1, ip2, ip3 = .{ ptrAdd(ip1, is1), ptrAdd(ip2, is2), ptrAdd(ip3, is3) };
        }

        loadPtr(T, a3).* = sum;
        a0, a1, a2, a3 = .{ ptrAdd(a0, s0), ptrAdd(a1, s1), ptrAdd(a2, s2), ptrAdd(a3, s3) };
    }
}

/// signature: (i)->(i)
fn gufunc_cumsum(
    comptime T: type,
    args: [*c][*c]u8,
    dims: [*c]const npy_intp,
    steps: [*c]const npy_intp,
) void {
    const dN, const di = asDims2(dims);
    const s0, const s1, const is_, const os_ = asSteps(4, steps).*;
    var a0, var a1 = .{ args[0], args[1] };

    for (0..dN) |_| {
        var ip, var op = .{ a0, a1 };
        var acc: T = 0;

        for (0..di) |_| {
            acc = add(T, acc, loadPtr(T, ip).*);
            loadPtr(T, op).* = acc;
            ip, op = .{ ptrAdd(ip, is_), ptrAdd(op, os_) };
        }

        a0, a1 = .{ ptrAdd(a0, s0), ptrAdd(a1, s1) };
    }
}

fn exportKernel(
    comptime kernel_tag: []const u8,
    comptime dtype_tag: []const u8,
    comptime T: type,
    comptime kernel: fn (
        comptime T: type,
        args: [*c][*c]u8,
        dims: [*c]const npy_intp,
        steps: [*c]const npy_intp,
    ) void,
) void {
    const name = std.fmt.comptimePrint("npy_zig_{s}_{s}", .{ kernel_tag, dtype_tag });
    // @export requires a null-terminated (sentinel) name.
    const namez: [:0]const u8 = (name ++ "\x00")[0..name.len :0];

    const Wrapper = struct {
        fn f(
            args: [*c][*c]u8,
            dims: [*c]const npy_intp,
            steps: [*c]const npy_intp,
            data: ?*anyopaque,
        ) callconv(.c) void {
            _ = data;
            kernel(T, args, dims, steps);
        }
    };

    @export(&Wrapper.f, .{ .name = namez });
}

comptime {
    exportKernel("inner1d", "intp", npy_intp, gufunc_inner1d);
    exportKernel("inner1d", "double", npy_double, gufunc_inner1d);
    exportKernel("innerwt", "intp", npy_intp, gufunc_innerwt);
    exportKernel("innerwt", "double", npy_double, gufunc_innerwt);
    exportKernel("cumsum", "intp", npy_intp, gufunc_cumsum);
    exportKernel("cumsum", "double", npy_double, gufunc_cumsum);
}
