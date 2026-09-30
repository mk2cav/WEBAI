/**
 * Public API of `@huggingface/kernels`.
 *
 * This preview exposes two functions: `getKernel`, which loads a kernel from a Hugging
 * Face Hub kernel repository and prepares it to run, and `disposeSharedKernelRuntime`,
 * which releases the GPU runtime it creates.
 */

/** Logical element type of a tensor. */
export type KernelDtype =
  | "float16"
  | "float32"
  | "int8"
  | "int16"
  | "int32"
  | "uint8"
  | "uint32"
  | "bool"
  | "int64";

/**
 * Executable WebGPU storage type.
 *
 * WebGPU storage buffers have no sub-32-bit integer element type, so int8, int16 and
 * bool travel as 32-bit words. WGSL has no 64-bit integer either, so int64 storage is an
 * opaque eight-byte element (two 32-bit words): kernels that accept it move it
 * (gather, concatenate, reshape, ...) but never compute on it. On the host it is a
 * `BigInt64Array`.
 */
export type KernelStorageDtype = KernelDtype;

/**
 * A tensor supplied from host memory.
 *
 * `data` may be a `Float32Array`, `Float16Array`, `Int8Array`, `Int16Array`,
 * `Int32Array`, `Uint8Array`, `Uint8ClampedArray`, `Uint16Array`, `Uint32Array` or
 * `BigInt64Array`. Narrow types are widened to their WebGPU storage form for you, so
 * image bytes can be handed over as a `Uint8Array` unchanged. A bare `Uint16Array`
 * means float16 — raw IEEE binary16 words — because WebGPU has no uint16 element
 * type; prefer a `Float16Array` where the environment provides one. A `Float64Array`
 * is refused rather than silently rounded — convert it yourself.
 */
export interface KernelCpuTensor {
  readonly data: ArrayBufferView & { readonly length: number };
  readonly shape: readonly number[];
  /**
   * Element type, when the array class does not already say it. The class decides by
   * its own width — a `Uint8Array` is uint8, an `Int32Array` is int32 — so this is
   * needed only to say something else: `bool` for a `Uint8Array` of 0/1, or `int64`
   * for a `BigInt64Array` where the kernel declares that projection.
   */
  readonly dtype?: KernelDtype;
}

/**
 * A GPU-resident tensor: what `output: "gpu"` returns, and what a later call accepts
 * back, so kernels chain without a round trip through host memory.
 *
 * You own it once a call returns it — release it with `destroy()`. Obtain one only
 * from a kernel result; a hand-built object is refused at the call.
 */
export interface KernelGpuTensor {
  readonly buffer: GPUBuffer;
  readonly shape: readonly number[];
  readonly dtype: KernelStorageDtype;
  readonly byteOffset: number;
  readonly byteLength: number;
  /** Element count. */
  readonly size: number;
  readonly destroyed: boolean;
  destroy(): void;
}

/** A destination to allocate for an output the kernel cannot infer. */
export interface KernelOutputSpec {
  readonly shape: readonly number[];
  readonly dtype?: KernelDtype;
}

/**
 * A tensor returned in host memory.
 *
 * `data` arrives in the natural class for its `dtype`: float16 results are a
 * `Float16Array` where the environment provides one, and otherwise a `Uint16Array`
 * of raw IEEE binary16 words.
 */
export interface KernelCpuTensorOutput {
  readonly data: ArrayBufferView & { readonly length: number };
  readonly shape: readonly number[];
  readonly dtype: KernelDtype;
}

/** A GPU-resident result, carrying both its public and its executable type. */
export type KernelGpuTensorOutput = KernelGpuTensor & {
  readonly logicalDtype: KernelDtype;
  readonly storageDtype: KernelStorageDtype;
};

/** Named inputs: tensors, plus any scalar arguments the kernel declares. */
export type KernelInputs = Record<string, KernelCpuTensor | KernelGpuTensor | number | boolean | string>;

export type KernelCpuResult = Record<string, KernelCpuTensorOutput>;
export type KernelGpuResult = Record<string, KernelGpuTensorOutput>;
export type KernelResult = Record<string, KernelCpuTensorOutput | KernelGpuTensorOutput>;

export interface KernelCallOptions {
  /** Kernel attributes (ONNX-style), where the kernel declares them. */
  readonly attrs?: Record<string, unknown>;
  /** Required scalar arguments, where the kernel declares them. */
  readonly args?: Record<string, unknown>;
  /**
   * Destinations for outputs whose shape the kernel cannot infer — a resident tensor
   * to write into, or a shape to allocate. Calling without one where it is needed
   * fails before anything reaches the GPU, and the kernel's own README shows the
   * correct call.
   */
  readonly outputs?: Record<string, KernelGpuTensor | KernelOutputSpec>;
  /** Where results are returned. Defaults to `"cpu"`. */
  readonly output?: "cpu" | "gpu";
}

/**
 * A loaded kernel: call it with named inputs.
 *
 * The result type follows the mode the call asks for — a default call yields host
 * tensors, `output: "gpu"` yields resident ones you own.
 */
export interface Kernel {
  (inputs: KernelInputs, options?: KernelCallOptions & { output?: "cpu" }): Promise<KernelCpuResult>;
  (inputs: KernelInputs, options: KernelCallOptions & { output: "gpu" }): Promise<KernelGpuResult>;
  (inputs: KernelInputs, options?: KernelCallOptions): Promise<KernelResult>;
  /** The op this kernel implements, e.g. `ai.onnx.Relu`. */
  readonly opId: string;
}

interface KernelSourceOptions {
  /** Bearer token for private repositories. Sent only to Hugging Face origins. */
  readonly token?: string;
  /** Verify every fetched file against the artifact's digest. Default `true`. */
  readonly verifyHashes?: boolean;
  /** Require the loaded artifact to be this op. */
  readonly expectedOpId?: string;
  /**
   * Load a kernel published outside the `webgpu-kernels` organization.
   *
   * A kernel is code that runs on your GPU, so this release loads only the official
   * organization by default. Set this to `true` for another publisher you trust.
   */
  readonly trustRemoteCode?: boolean;
}

/**
 * Which build of a kernel to load. One of the two is required — there is no default.
 *
 * `version: N` follows the repository's `vN` branch, which moves as fixes land. Pass
 * a 40-character commit `revision` instead to pin bytes that can never change.
 */
export type GetKernelOptions = KernelSourceOptions & (
  | { readonly version: number; readonly revision?: string }
  | { readonly version?: number; readonly revision: string }
);

/**
 * Load a kernel from a Hugging Face Hub kernel repository and prepare it to run.
 *
 * ```js
 * const relu = await getKernel("webgpu-kernels/ai.onnx.Relu", { version: 1 });
 * const { y } = await relu({ x: { data: new Float32Array(100), shape: [10, 10] } });
 * ```
 */
export declare function getKernel(repoId: string, options: GetKernelOptions): Promise<Kernel>;

/**
 * Release the shared GPU runtime.
 *
 * It is created lazily on the first kernel call, not by `getKernel`. Kernels stay
 * usable afterwards — the next call simply creates a new runtime — so this is for page
 * teardown and tests. Resolves once the device is actually gone.
 */
export declare function disposeSharedKernelRuntime(): Promise<void>;
