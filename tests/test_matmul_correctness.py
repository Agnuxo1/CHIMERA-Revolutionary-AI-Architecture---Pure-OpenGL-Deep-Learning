#!/usr/bin/env python
"""
Test exhaustivo de correctness para matmul GPU.
Verifica que GPU da resultados identicos a NumPy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import moderngl


def test_matmul_correctness(size: int, tolerance: float = 1e-4) -> bool:
    """
    Test matmul para un tamaño especifico.

    Args:
        size: Tamaño de matrices (size × size)
        tolerance: Tolerancia para error

    Returns:
        True si pasa el test
    """

    # Crear matrices random
    np.random.seed(42 + size)  # Seed deterministica
    a_data = np.random.randn(size, size).astype(np.float32) * 0.1
    b_data = np.random.randn(size, size).astype(np.float32) * 0.1

    # Crear contexto OpenGL
    ctx = moderngl.create_standalone_context()

    # Crear texturas para matrices
    texture_a = ctx.texture((size, size), 4, dtype='f4')
    texture_b = ctx.texture((size, size), 4, dtype='f4')
    texture_result = ctx.texture((size, size), 4, dtype='f4')

    # Subir datos a texturas
    texture_a.write(a_data.tobytes())
    texture_b.write(b_data.tobytes())

    # Crear shader simple para matrix multiplication
    shader_source = """
    #version 430 core

    layout (local_size_x = 16, local_size_y = 16) in;

    layout (rgba32f, binding = 0) uniform image2D matrix_a;
    layout (rgba32f, binding = 1) uniform image2D matrix_b;
    layout (rgba32f, binding = 2) uniform image2D result;

    void main() {
        ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

        if (pos.x >= """ + str(size) + """ || pos.y >= """ + str(size) + """) return;

        vec4 a_row = imageLoad(matrix_a, ivec2(pos.x, pos.y));
        vec4 b_col0 = imageLoad(matrix_b, ivec2(0, pos.y));
        vec4 b_col1 = imageLoad(matrix_b, ivec2(1, pos.y));

        vec4 result_val = vec4(
            dot(a_row, vec4(b_col0.x, b_col1.x, 0.0, 0.0)),
            dot(a_row, vec4(b_col0.y, b_col1.y, 0.0, 0.0)),
            dot(a_row, vec4(b_col0.z, b_col1.z, 0.0, 0.0)),
            dot(a_row, vec4(b_col0.w, b_col1.w, 0.0, 0.0))
        );

        imageStore(result, pos, result_val);
    }
    """

    # Crear compute program
    compute_shader = ctx.compute_shader(shader_source)

    # Bind textures
    texture_a.bind_to_image(0)
    texture_b.bind_to_image(1)
    texture_result.bind_to_image(2)

    # Dispatch compute shader
    group_size = 16
    groups_x = (size + group_size - 1) // group_size
    groups_y = (size + group_size - 1) // group_size

    compute_shader.run(groups_x, groups_y)
    ctx.finish()  # Wait for completion

    # Read result
    result_data = texture_result.read()
    gpu_output = np.frombuffer(result_data, dtype=np.float32).reshape(size, size, 4)[:,:,0]

    # NumPy reference
    expected = a_data @ b_data

    # Comparar
    max_error = np.abs(gpu_output - expected).max()
    mean_error = np.abs(gpu_output - expected).mean()
    rel_error = max_error / (np.abs(expected).max() + 1e-8)

    passed = max_error < tolerance

    # Cleanup
    texture_a.release()
    texture_b.release()
    texture_result.release()
    ctx.release()

    return passed, max_error, mean_error, rel_error


def run_all_tests():
    """Ejecuta tests para todos los tamaños."""
    print("\n" + "=" * 70)
    print(" " * 15 + "MATMUL CORRECTNESS TEST")
    print("=" * 70)

    sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    tolerance = 1e-4

    print(f"\nTolerance: {tolerance}")
    print("\n| Size | Max Error | Mean Error | Rel Error | Status |")
    print("|------|-----------|------------|-----------|--------|")

    all_passed = True

    for size in sizes:
        try:
            passed, max_err, mean_err, rel_err = test_matmul_correctness(size, tolerance)

            status = "OK" if passed else "FAIL"
            if not passed:
                all_passed = False

            print(f"| {size:4d} | {max_err:9.2e} | {mean_err:10.2e} | {rel_err:9.2e} | {status:6s} |")

        except Exception as e:
            print(f"| {size:4d} | ERROR: {str(e)[:30]:30s} | FAIL   |")
            all_passed = False

    print()

    if all_passed:
        print("=" * 70)
        print(" " * 20 + "TODOS LOS TESTS PASARON")
        print("=" * 70)
        print("\n✅ Matmul GPU es CORRECTO")
        print("GPU puede hacer algebra lineal confiablemente")
        return True
    else:
        print("=" * 70)
        print(" " * 20 + "ALGUNOS TESTS FALLARON")
        print("=" * 70)
        print("\n❌ Matmul GPU tiene problemas de correctness")
        return False


def test_non_square_matrices():
    """Test con matrices no cuadradas."""
    print("\n" + "=" * 70)
    print(" " * 15 + "NON-SQUARE MATRICES TEST")
    print("=" * 70)

    test_cases = [
        (10, 20, 30),  # (M, K, N)
        (32, 64, 48),
        (100, 50, 75),
    ]

    print("\n| Shape (M×K)×(K×N) | Max Error | Status |")
    print("|-------------------|-----------|--------|")

    for M, K, N in test_cases:
        # Crear matrices
        a_data = np.random.randn(M, K).astype(np.float32) * 0.1
        b_data = np.random.randn(K, N).astype(np.float32) * 0.1

        # Crear contexto OpenGL
        ctx = moderngl.create_standalone_context()

        # Crear texturas
        texture_a = ctx.texture((M, K), 4, dtype='f4')
        texture_b = ctx.texture((K, N), 4, dtype='f4')
        texture_result = ctx.texture((M, N), 4, dtype='f4')

        # Subir datos
        texture_a.write(a_data.tobytes())
        texture_b.write(b_data.tobytes())

        # Crear shader para non-square matrices
        shader_source = f"""
        #version 430 core

        layout (local_size_x = 16, local_size_y = 16) in;

        layout (rgba32f, binding = 0) uniform image2D matrix_a;
        layout (rgba32f, binding = 1) uniform image2D matrix_b;
        layout (rgba32f, binding = 2) uniform image2D result;

        void main() {{
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

            if (pos.x >= {M} || pos.y >= {N}) return;

            vec4 sum = vec4(0.0);

            for (int k = 0; k < {K}; k++) {{
                vec4 a_val = imageLoad(matrix_a, ivec2(k, pos.x));
                vec4 b_val = imageLoad(matrix_b, ivec2(pos.y, k));
                sum += a_val * b_val.x;
            }}

            imageStore(result, pos, sum);
        }}
        """

        # Crear compute program
        compute_shader = ctx.compute_shader(shader_source)

        # Bind textures
        texture_a.bind_to_image(0)
        texture_b.bind_to_image(1)
        texture_result.bind_to_image(2)

        # Dispatch compute shader
        group_size = 16
        groups_x = (M + group_size - 1) // group_size
        groups_y = (N + group_size - 1) // group_size

        compute_shader.run(groups_x, groups_y)
        ctx.finish()

        # Read result
        result_data = texture_result.read()
        gpu_output = np.frombuffer(result_data, dtype=np.float32).reshape(M, N, 4)[:,:,0]

        # NumPy reference
        expected = a_data @ b_data

        # Comparar
        max_error = np.abs(gpu_output - expected).max()
        passed = max_error < 1e-4

        status = "OK" if passed else "FAIL"
        shape_str = f"({M}×{K})×({K}×{N})"

        print(f"| {shape_str:17s} | {max_error:9.2e} | {status:6s} |")

        # Cleanup
        texture_a.release()
        texture_b.release()
        texture_result.release()
        ctx.release()

    print()


def main():
    """Main test function."""
    print("\n" + "=" * 70)
    print(" " * 10 + "CHIMERA v3.0 - MATMUL CORRECTNESS SUITE")
    print("=" * 70)

    # Test matrices cuadradas
    all_passed = run_all_tests()

    # Test matrices no cuadradas
    test_non_square_matrices()

    if all_passed:
        print("\n" + "=" * 70)
        print("✅ MATMUL GPU VERIFICADO")
        print("=" * 70)
        print("\nMatmul GPU shader es correcto y confiable.")
        print("Listo para uso en produccion.")
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ Revisar implementacion")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
