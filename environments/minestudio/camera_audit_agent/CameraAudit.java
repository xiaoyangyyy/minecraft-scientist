import java.io.BufferedWriter;
import java.lang.instrument.ClassFileTransformer;
import java.lang.instrument.Instrumentation;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.ProtectionDomain;
import jdk.internal.org.objectweb.asm.ClassReader;
import jdk.internal.org.objectweb.asm.ClassVisitor;
import jdk.internal.org.objectweb.asm.ClassWriter;
import jdk.internal.org.objectweb.asm.MethodVisitor;
import jdk.internal.org.objectweb.asm.Opcodes;

/** Evaluator-only instrumentation of the installed JDK8 recorder, not a policy. */
public final class CameraAudit {
    private static BufferedWriter output;
    private static int sequence;
    private static int renderSequence;
    private static String lastRender = "null";

    public static void premain(String path, Instrumentation instrumentation) throws Exception {
        output = Files.newBufferedWriter(Paths.get(path), StandardCharsets.UTF_8,
                StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
        instrumentation.addTransformer(new ClassFileTransformer() {
            public byte[] transform(ClassLoader loader, String name, Class<?> type,
                    ProtectionDomain domain, byte[] bytes) {
                final boolean recorder = name.equals("com/minerl/multiagent/recorder/PlayRecorder");
                final boolean world = name.equals("net/minecraft/client/renderer/WorldRenderer");
                if (!recorder && !world) return null;
                ClassReader reader = new ClassReader(bytes);
                ClassWriter writer = new ClassWriter(reader, ClassWriter.COMPUTE_MAXS);
                final int[] hooks = {0};
                reader.accept(new ClassVisitor(Opcodes.ASM5, writer) {
                    public MethodVisitor visitMethod(int access, String method, String descriptor,
                            String signature, String[] exceptions) {
                        MethodVisitor original = super.visitMethod(access, method, descriptor,
                                signature, exceptions);
                        if (world && method.equals("updateCameraAndRender") && descriptor.equals(
                                "(Lcom/mojang/blaze3d/matrix/MatrixStack;FJZLnet/minecraft/client/renderer/ActiveRenderInfo;Lnet/minecraft/client/renderer/GameRenderer;Lnet/minecraft/client/renderer/LightTexture;Lnet/minecraft/util/math/vector/Matrix4f;)V")) {
                            hooks[0]++;
                            return new MethodVisitor(Opcodes.ASM5, original) {
                                public void visitCode() {
                                    super.visitCode();
                                    super.visitVarInsn(Opcodes.ALOAD, 1);
                                    super.visitVarInsn(Opcodes.ALOAD, 6);
                                    super.visitVarInsn(Opcodes.ALOAD, 9);
                                    super.visitVarInsn(Opcodes.FLOAD, 2);
                                    super.visitMethodInsn(Opcodes.INVOKESTATIC, "CameraAudit", "render",
                                            "(Ljava/lang/Object;Ljava/lang/Object;Ljava/lang/Object;F)V", false);
                                }
                            };
                        }
                        if (!recorder || !method.equals("resolveImageRequests") || !descriptor.equals("()V"))
                            return original;
                        hooks[0]++;
                        return new MethodVisitor(Opcodes.ASM5, original) {
                            public void visitInsn(int opcode) {
                                if (opcode == Opcodes.RETURN) {
                                    super.visitVarInsn(Opcodes.ALOAD, 0);
                                    super.visitMethodInsn(Opcodes.INVOKESTATIC, "CameraAudit",
                                            "capture", "(Ljava/lang/Object;)V", false);
                                }
                                super.visitInsn(opcode);
                            }
                        };
                    }
                }, 0);
                if (hooks[0] != 1) throw new IllegalStateException("unexpected recorder hook count");
                line("{\"kind\":\"instrumented\",\"target\":\""
                        + (recorder ? "recorder" : "world_renderer") + "\",\"hooks\":1}");
                return writer.toByteArray();
            }
        });
    }

    private static synchronized void line(String text) {
        try {
            output.write(text);
            output.newLine();
            output.flush();
        } catch (Exception error) {
            throw new IllegalStateException("camera audit write failed", error);
        }
    }

    private static Object field(Object object, String name) throws Exception {
        Class<?> type = object.getClass();
        while (type != null) {
            try {
                Field value = type.getDeclaredField(name);
                value.setAccessible(true);
                return value.get(object);
            } catch (NoSuchFieldException ignored) {
                type = type.getSuperclass();
            }
        }
        throw new NoSuchFieldException(name);
    }

    private static Object call(Object object, String name) throws Exception {
        return object.getClass().getMethod(name).invoke(object);
    }

    private static String sha256(byte[] data) throws Exception {
        StringBuilder result = new StringBuilder();
        for (byte value : MessageDigest.getInstance("SHA-256").digest(data))
            result.append(String.format("%02x", value & 255));
        return result.toString();
    }

    private static String player(Object value, boolean previous) throws Exception {
        return "{\"xpos\":" + (previous ? field(value, "prevPosX") : call(value, "getPosX"))
                + ",\"ypos\":" + (previous ? field(value, "prevPosY") : call(value, "getPosY"))
                + ",\"zpos\":" + (previous ? field(value, "prevPosZ") : call(value, "getPosZ"))
                + ",\"yaw\":" + field(value, previous ? "prevRotationYaw" : "rotationYaw")
                + ",\"pitch\":" + field(value, previous ? "prevRotationPitch" : "rotationPitch") + "}";
    }

    private static String matrix(Object value) throws Exception {
        StringBuilder text = new StringBuilder("[");
        for (int row = 0; row < 4; row++) {
            if (row > 0) text.append(',');
            text.append('[');
            for (int col = 0; col < 4; col++) {
                if (col > 0) text.append(',');
                text.append(field(value, "m" + row + col));
            }
            text.append(']');
        }
        return text.append(']').toString();
    }

    public static void render(Object stack, Object camera, Object projection, float partial) {
        try {
            if (renderSequence >= 10000) throw new IllegalStateException("render audit budget exceeded");
            Object position = call(camera, "getProjectedView");
            Class<?> shaders = Class.forName("net.optifine.shaders.Shaders", false,
                    camera.getClass().getClassLoader());
            lastRender = "{\"sequence\":" + ++renderSequence + ",\"partial_ticks\":" + partial
                    + ",\"model_view\":" + matrix(call(call(stack, "getLast"), "getMatrix"))
                    + ",\"projection\":" + matrix(projection)
                    + ",\"base_position\":[" + field(position, "x") + ","
                    + field(position, "y") + "," + field(position, "z") + "]"
                    + ",\"shader_pack_loaded\":" + shaders.getField("shaderPackLoaded").get(null)
                    + ",\"shadow_pass\":" + shaders.getField("isShadowPass").get(null) + "}";
        } catch (Exception error) {
            line("{\"kind\":\"render_error\"}");
            throw new IllegalStateException("render matrix audit failed", error);
        }
    }

    public static void capture(Object recorder) {
        try {
            if (sequence >= 10000) throw new IllegalStateException("camera audit sample budget exceeded");
            Object minecraft = field(recorder, "mc");
            Object entity = field(minecraft, "player");
            if (entity == null) return;
            Object settings = field(minecraft, "gameSettings");
            Object renderer = field(minecraft, "gameRenderer");
            Object info = call(renderer, "getActiveRenderInfo");
            Object position = call(info, "getProjectedView");
            byte[] pixels = ((byte[]) field(recorder, "lastImageBytes")).clone();
            line("{\"kind\":\"capture\",\"sequence\":" + sequence++
                    + ",\"tick_counter\":" + call(recorder, "getTickCounter")
                    + ",\"width\":" + field(recorder, "width")
                    + ",\"height\":" + field(recorder, "height")
                    + ",\"raw_rgb_sha256\":\"" + sha256(pixels) + "\""
                    + ",\"partial_ticks\":" + call(minecraft, "getRenderPartialTicks")
                    + ",\"sync_integrated_server\":" + field(settings, "syncIntegratedServer")
                    + ",\"disable_recorder\":" + field(settings, "disableRecorder")
                    + ",\"bob_view\":" + field(settings, "viewBobbing")
                    + ",\"fov_hand\":" + field(renderer, "fovModifierHand")
                    + ",\"previous_fov_hand\":" + field(renderer, "fovModifierHandPrev")
                    + ",\"player_fov_modifier\":" + call(entity, "getFovModifier")
                    + ",\"third_person\":" + call(info, "isThirdPerson")
                    + ",\"camera_valid\":" + call(info, "isValid")
                    + ",\"eye_height\":" + call(entity, "getEyeHeight")
                    + ",\"render\":" + lastRender
                    + ",\"player\":" + player(entity, false)
                    + ",\"previous_player\":" + player(entity, true)
                    + ",\"camera_base\":{\"xpos\":" + field(position, "x")
                    + ",\"ypos\":" + field(position, "y")
                    + ",\"zpos\":" + field(position, "z")
                    + ",\"yaw\":" + call(info, "getYaw")
                    + ",\"pitch\":" + call(info, "getPitch") + "}}");
        } catch (Exception error) {
            line("{\"kind\":\"error\",\"sequence\":" + sequence + "}");
            throw new IllegalStateException("camera audit capture failed", error);
        }
    }
}
