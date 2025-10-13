#!/usr/bin/env python3
"""
Benchmark: Speed Tests
Mide velocidad de cada componente del pipeline.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import moderngl


def benchmark_rasterization():
    """Benchmark de operaciones básicas."""
    print("\n" + "=" * 70)
    print(" " * 20 + "BENCHMARK: BASIC OPERATIONS")
    print("=" * 70)

    # Crear contexto OpenGL
    ctx = moderngl.create_standalone_context()

    # Test diferentes tamaños de matriz
    sizes = [256, 512, 1024]

    print("\n⏱️ Midiendo velocidad de operaciones básicas...")

    for size in sizes:
        # Crear matrices de prueba
        a_data = np.random.randn(size, size).astype(np.float32) * 0.1
        b_data = np.random.randn(size, size).astype(np.float32) * 0.1

        # Crear texturas
        texture_a = ctx.texture((size, size), 4, dtype='f4')
        texture_b = ctx.texture((size, size), 4, dtype='f4')
        texture_result = ctx.texture((size, size), 4, dtype='f4')

        # Subir datos
        texture_a.write(a_data.tobytes())
        texture_b.write(b_data.tobytes())

        # Crear shader simple para suma
        shader_source = """
        #version 330 core

        in vec2 v_texcoord;
        out vec4 f_color;

        uniform sampler2D texture_a;
        uniform sampler2D texture_b;

        void main() {
            vec4 a = texture(texture_a, v_texcoord);
            vec4 b = texture(texture_b, v_texcoord);
            f_color = a + b;
        }
        """

        # Crear programa
        program = ctx.program(
            vertex_shader="""
            #version 330 core
            in vec2 in_vert;
            out vec2 v_texcoord;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
                v_texcoord = in_vert * 0.5 + 0.5;
            }
            """,
            fragment_shader=shader_source
        )

        # Crear framebuffer y VAO
        fbo = ctx.framebuffer(color_attachments=[texture_result])

        vertices = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        vbo = ctx.buffer(vertices.tobytes())
        vao = ctx.vertex_array(program, [(vbo, '2f', 'in_vert')])

        # Medir tiempo
        num_iterations = 10
        start_time = time.time()

        for _ in range(num_iterations):
            fbo.use()
            texture_a.use(0)
            texture_b.use(1)
            vao.render()

        ctx.finish()
        total_time = time.time() - start_time

        avg_time = (total_time / num_iterations) * 1000  # ms

        print(f"  {size}x{size} matrices: {avg_time".2f"} ms avg")

        # Cleanup
        texture_a.release()
        texture_b.release()
        texture_result.release()
        fbo.release()
        ctx.release()

    print()


def benchmark_matrix_multiplication():
    """Benchmark de multiplicación de matrices."""
    print("\n" + "=" * 70)
    print(" " * 15 + "BENCHMARK: MATRIX MULTIPLICATION")
    print("=" * 70)

    # Crear contexto OpenGL
    ctx = moderngl.create_standalone_context()

    sizes = [128, 256, 512]

    for size in sizes:
        # Crear matrices de prueba
        a_data = np.random.randn(size, size).astype(np.float32) * 0.1
        b_data = np.random.randn(size, size).astype(np.float32) * 0.1

        # Crear texturas
        texture_a = ctx.texture((size, size), 4, dtype='f4')
        texture_b = ctx.texture((size, size), 4, dtype='f4')
        texture_result = ctx.texture((size, size), 4, dtype='f4')

        # Subir datos
        texture_a.write(a_data.tobytes())
        texture_b.write(b_data.tobytes())

        # Crear compute shader para matrix multiplication
        shader_source = f"""
        #version 430 core

        layout (local_size_x = 16, local_size_y = 16) in;

        layout (rgba32f, binding = 0) uniform image2D matrix_a;
        layout (rgba32f, binding = 1) uniform image2D matrix_b;
        layout (rgba32f, binding = 2) uniform image2D result;

        void main() {{
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

            if (pos.x >= {size} || pos.y >= {size}) return;

            vec4 a_row = imageLoad(matrix_a, ivec2(pos.x, pos.y));
            vec4 sum = vec4(0.0);

            for (int k = 0; k < {size}; k++) {{
                vec4 b_col = imageLoad(matrix_b, ivec2(k, pos.y));
                sum += a_row * b_col.x;
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
        groups_x = (size + group_size - 1) // group_size
        groups_y = (size + group_size - 1) // group_size

        # Medir tiempo
        num_iterations = 5
        start_time = time.time()

        for _ in range(num_iterations):
            compute_shader.run(groups_x, groups_y)

        ctx.finish()
        total_time = time.time() - start_time

        avg_time = (total_time / num_iterations) * 1000  # ms

        print(f"  {size}x{size} matmul: {avg_time".2f"} ms avg")

        # Cleanup
        texture_a.release()
        texture_b.release()
        texture_result.release()

    ctx.release()
    print()


def benchmark_attention_mechanism():
    """Benchmark de mecanismo de atención."""
    print("\n" + "=" * 70)
    print(" " * 20 + "BENCHMARK: ATTENTION")
    print("=" * 70)

    # Crear contexto OpenGL
    ctx = moderngl.create_standalone_context()

    seq_len = 32
    d_model = 64

    # Crear datos de atención
    Q = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1
    K = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1
    V = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1

    # Crear texturas
    texture_q = ctx.texture((seq_len, d_model), 4, dtype='f4')
    texture_k = ctx.texture((seq_len, d_model), 4, dtype='f4')
    texture_v = ctx.texture((seq_len, d_model), 4, dtype='f4')
    texture_result = ctx.texture((seq_len, d_model), 4, dtype='f4')

    # Subir datos
    texture_q.write(Q.tobytes())
    texture_k.write(K.tobytes())
    texture_v.write(V.tobytes())

    # Crear compute shader para atención simplificada
    shader_source = """
    #version 430 core

    layout (local_size_x = 16, local_size_y = 16) in;

    layout (rgba32f, binding = 0) uniform image2D queries;
    layout (rgba32f, binding = 1) uniform image2D keys;
    layout (rgba32f, binding = 2) uniform image2D values;
    layout (rgba32f, binding = 3) uniform image2D result;

    void main() {
        ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

        if (pos.x >= 32 || pos.y >= 64) return;

        // Simplified attention computation
        vec4 q = imageLoad(queries, ivec2(pos.x, pos.y));
        vec4 sum = vec4(0.0);
        float total_weight = 0.0;

        for (int i = 0; i < 32; i++) {
            vec4 k = imageLoad(keys, ivec2(i, pos.y));
            float score = dot(q, k);
            float weight = exp(score);
            total_weight += weight;

            vec4 v = imageLoad(values, ivec2(i, pos.y));
            sum += weight * v;
        }

        if (total_weight > 0.0) {
            sum /= total_weight;
        }

        imageStore(result, pos, sum);
    }
    """

    # Crear compute program
    compute_shader = ctx.compute_shader(shader_source)

    # Bind textures
    texture_q.bind_to_image(0)
    texture_k.bind_to_image(1)
    texture_v.bind_to_image(2)
    texture_result.bind_to_image(3)

    # Dispatch compute shader
    group_size = 16
    groups_x = (seq_len + group_size - 1) // group_size
    groups_y = (d_model + group_size - 1) // group_size

    # Medir tiempo
    num_iterations = 5
    start_time = time.time()

    for _ in range(num_iterations):
        compute_shader.run(groups_x, groups_y)

    ctx.finish()
    total_time = time.time() - start_time

    avg_time = (total_time / num_iterations) * 1000  # ms

    print(f"  Attention ({seq_len}x{d_model}): {avg_time".2f"} ms avg")

    # Cleanup
    texture_q.release()
    texture_k.release()
    texture_v.release()
    texture_result.release()
    ctx.release()

    print()


def main():
    """Main benchmark function."""
    print("\n" + "=" * 70)
    print(" " * 25 + "CHIMERA v3.0 BENCHMARKS")
    print("=" * 70)

    try:
        # Run benchmarks
        benchmark_rasterization()
        benchmark_matrix_multiplication()
        benchmark_attention_mechanism()

        print("\n✅ Benchmarks completados")

    except Exception as e:
        print(f"\n❌ Error durante benchmarks: {e}")
        print("Esto puede deberse a:")
        print("- GPU no compatible con OpenGL 4.3+")
        print("- Memoria GPU insuficiente")
        print("- Drivers GPU desactualizados")


if __name__ == '__main__':
    main()
