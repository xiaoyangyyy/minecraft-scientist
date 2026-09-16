package dev.minecraftscientist.worldrules;

import com.mojang.brigadier.Command;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.arguments.LongArgumentType;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.fabricmc.fabric.api.event.player.PlayerBlockBreakEvents;
import net.fabricmc.fabric.api.event.player.UseBlockCallback;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.commands.arguments.EntityArgument;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.permissions.Permissions;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Deterministic, server-authoritative mechanisms for causal discovery experiments. */
public final class ScientistWorldRulesMod implements ModInitializer {
    public static final String MOD_ID = "scientist_world_rules";
    private static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);
    private static final Map<String, ScheduledMachine> MACHINES = new HashMap<>();
    private static final Map<String, ActiveTransform> TRANSFORMS = new HashMap<>();
    private static volatile RuleVariant activeVariant = RuleVariant.STANDARD;
    private static volatile long activeSeed = 0L;

    @Override
    public void onInitialize() {
        registerCommands();
        registerMiningIntervention();
        registerMachineAndTransforms();
        registerMachineClock();
        LOGGER.info("Minecraft Scientist rule controller initialized");
    }

    private static void registerCommands() {
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
            dispatcher.register(Commands.literal("scientist")
                .requires(source -> source.permissions().hasPermission(Permissions.COMMANDS_ADMIN))
                .then(Commands.literal("reset")
                    .then(Commands.argument("seed", LongArgumentType.longArg())
                        .executes(context -> {
                            resetMachines();
                            resetPlayers(context.getSource());
                            activeSeed = LongArgumentType.getLong(context, "seed");
                            activeVariant = RuleVariant.STANDARD;
                            return sendState(context.getSource(), "reset");
                        })))
                .then(Commands.literal("variant")
                    .then(Commands.literal("standard").executes(context -> {
                        activeVariant = RuleVariant.STANDARD;
                        return sendState(context.getSource(), "variant");
                    }))
                    .then(Commands.literal("shifted").executes(context -> {
                        activeVariant = RuleVariant.SHIFTED;
                        return sendState(context.getSource(), "variant");
                    })))
                .then(Commands.literal("status")
                    .executes(context -> sendState(context.getSource(), "status")))
                .then(Commands.literal("fall")
                    .then(Commands.argument("player", EntityArgument.player())
                        .then(Commands.argument("armor", IntegerArgumentType.integer(0, 2))
                            .then(Commands.argument("height", IntegerArgumentType.integer(1, 10))
                                .executes(context -> runFallIntervention(
                                    context.getSource(),
                                    EntityArgument.getPlayer(context, "player"),
                                    IntegerArgumentType.getInteger(context, "armor"),
                                    IntegerArgumentType.getInteger(context, "height")
                                ))))))
                .then(Commands.literal("truth").executes(context -> {
                    String truth = truthJson();
                    LOGGER.info("SCIENTIST_GROUND_TRUTH {}", truth);
                    context.getSource().sendSuccess(() -> Component.literal(truth), false);
                    return Command.SINGLE_SUCCESS;
                }))));
    }

    private static int runFallIntervention(
        CommandSourceStack source,
        ServerPlayer player,
        int armor,
        int height
    ) {
        float damage = fallSlope(armor) * Math.max(0, height - fallSafeHeight());
        // The intervention must remain stable until the controller observes it.
        // At normal difficulty an empty food bar disables vanilla regeneration;
        // otherwise the observation transport latency would become a hidden
        // confounder in the measured equation.
        player.getFoodData().setFoodLevel(0);
        player.getFoodData().setSaturation(0.0F);
        player.setHealth(Math.max(0.5F, 20.0F - damage));
        String payload = "{\"kind\":\"fall\",\"seed\":" + activeSeed
            + ",\"variant\":\"" + activeVariant.id + "\",\"armor\":" + armor
            + ",\"height\":" + height + ",\"damage\":" + damage + "}";
        LOGGER.info("SCIENTIST_EVENT {}", payload);
        source.sendSuccess(() -> Component.literal(payload), false);
        return Command.SINGLE_SUCCESS;
    }

    private static void registerMiningIntervention() {
        PlayerBlockBreakEvents.BEFORE.register((world, player, pos, state, blockEntity) -> {
            if (!isMiningTarget(state)) {
                return true;
            }
            boolean allowed = toolRank(player) >= miningRequirement(state);
            LOGGER.info(
                "SCIENTIST_EVENT {{\"kind\":\"break_attempt\",\"seed\":{},"
                    + "\"variant\":\"{}\",\"allowed\":{},\"player\":\"{}\"}}",
                activeSeed,
                activeVariant.id,
                allowed,
                player.getUUID()
            );
            return allowed;
        });

        PlayerBlockBreakEvents.AFTER.register((world, player, pos, state, blockEntity) -> {
            if (activeVariant != RuleVariant.SHIFTED || !isMiningTarget(state)) {
                return;
            }
            if (!player.getMainHandItem().isCorrectToolForDrops(state)) {
                grant(player, new ItemStack(dropFor(state)));
            }
        });
    }

    private static void registerMachineAndTransforms() {
        UseBlockCallback.EVENT.register((player, world, hand, hitResult) -> {
            if (world.isClientSide() || !(world instanceof ServerLevel serverLevel)) {
                return InteractionResult.PASS;
            }
            BlockPos pos = hitResult.getBlockPos();
            Block block = serverLevel.getBlockState(pos).getBlock();
            ItemStack held = player.getItemInHand(hand);
            if (isMachineBlock(block)) {
                return interactMachine(serverLevel, pos, player, held, block);
            }
            return interactTransform(serverLevel, pos, player, held, block);
        });
    }

    private static InteractionResult interactMachine(
        ServerLevel level,
        BlockPos pos,
        Player player,
        ItemStack held,
        Block block
    ) {
        boolean success = false;
        if (
            block == Blocks.LODESTONE
                && !player.isShiftKeyDown()
                && held.is(machineInput())
        ) {
            held.shrink(1);
            level.setBlockAndUpdate(pos, Blocks.COPPER_BLOCK.defaultBlockState());
            success = true;
        } else if (block == Blocks.COPPER_BLOCK && held.isEmpty()) {
            level.setBlockAndUpdate(pos, Blocks.ORANGE_GLAZED_TERRACOTTA.defaultBlockState());
            long readyTick = level.getServer().getTickCount() + machineDelay();
            MACHINES.put(machineKey(level, pos), new ScheduledMachine(level, pos.immutable(), readyTick));
            success = true;
        } else if (block == Blocks.EMERALD_BLOCK && held.isEmpty()) {
            grant(player, new ItemStack(machineOutput()));
            level.setBlockAndUpdate(pos, Blocks.LODESTONE.defaultBlockState());
            MACHINES.remove(machineKey(level, pos));
            success = true;
        }
        logMechanism("machine_interaction", pos, player, success);
        return InteractionResult.SUCCESS;
    }

    private static InteractionResult interactTransform(
        ServerLevel level,
        BlockPos pos,
        Player player,
        ItemStack held,
        Block block
    ) {
        Transform specification = transformFor(block);
        if (specification == null) {
            return InteractionResult.PASS;
        }
        String key = machineKey(level, pos);
        ActiveTransform active = TRANSFORMS.get(key);
        boolean success = false;
        if (active == null && !player.isShiftKeyDown() && held.is(specification.input)) {
            held.shrink(1);
            TRANSFORMS.put(
                key,
                new ActiveTransform(level, pos.immutable(), specification.output, false, -1L)
            );
            success = true;
        } else if (active != null && !active.activated && held.isEmpty()) {
            TRANSFORMS.put(
                key,
                new ActiveTransform(
                    active.level,
                    active.pos,
                    active.output,
                    true,
                    level.getServer().getTickCount() + machineDelay()
                )
            );
            success = true;
        } else if (
            active != null
                && active.activated
                && held.isEmpty()
                && level.getServer().getTickCount() >= active.readyTick
        ) {
            grant(player, new ItemStack(active.output));
            TRANSFORMS.remove(key);
            success = true;
        }
        logMechanism("transform", pos, player, success);
        return InteractionResult.SUCCESS;
    }

    private static void registerMachineClock() {
        ServerTickEvents.END_SERVER_TICK.register(server -> {
            Iterator<Map.Entry<String, ScheduledMachine>> iterator =
                MACHINES.entrySet().iterator();
            while (iterator.hasNext()) {
                ScheduledMachine machine = iterator.next().getValue();
                if (server.getTickCount() < machine.readyTick) {
                    continue;
                }
                Block current = machine.level.getBlockState(machine.pos).getBlock();
                if (current == Blocks.ORANGE_GLAZED_TERRACOTTA) {
                    machine.level.setBlockAndUpdate(
                        machine.pos,
                        Blocks.EMERALD_BLOCK.defaultBlockState()
                    );
                    LOGGER.info(
                        "SCIENTIST_EVENT {{\"kind\":\"machine_ready\",\"seed\":{},"
                            + "\"variant\":\"{}\",\"position\":\"{}\"}}",
                        activeSeed,
                        activeVariant.id,
                        machine.pos.toShortString()
                    );
                }
                iterator.remove();
            }
        });
    }

    private static Transform transformFor(Block block) {
        if (activeVariant == RuleVariant.STANDARD) {
            if (block == Blocks.CHISELED_COPPER) {
                return new Transform(Items.OAK_LOG, Items.OAK_PLANKS);
            }
            if (block == Blocks.CHISELED_TUFF) {
                return new Transform(Items.OAK_PLANKS, Items.STICK);
            }
            if (block == Blocks.POLISHED_BLACKSTONE) {
                return new Transform(Items.STICK, Items.TORCH);
            }
        } else {
            if (block == Blocks.CHISELED_COPPER) {
                return new Transform(Items.STICK, Items.TORCH);
            }
            if (block == Blocks.CHISELED_TUFF) {
                return new Transform(Items.OAK_LOG, Items.OAK_PLANKS);
            }
            if (block == Blocks.POLISHED_BLACKSTONE) {
                return new Transform(Items.OAK_PLANKS, Items.STICK);
            }
        }
        return null;
    }

    private static Item machineInput() {
        return activeVariant == RuleVariant.STANDARD ? Items.RAW_COPPER : Items.OAK_LOG;
    }

    private static Item machineOutput() {
        return activeVariant == RuleVariant.STANDARD ? Items.COPPER_INGOT : Items.GOLD_INGOT;
    }

    private static int machineDelay() {
        // Keep the mechanism clock well beyond the bridge's 250 ms packet-
        // acknowledgement window.  A protocol without an explicit symbolic
        // wait must therefore fail, while one wait quantum can reliably cross
        // this ten-tick boundary on a normally ticking local server.
        return 10;
    }

    private static int fallSafeHeight() {
        int standard = 2 + Math.floorMod(activeSeed, 3);
        if (activeVariant == RuleVariant.STANDARD) {
            return standard;
        }
        return standard == 2 ? 3 : standard == 3 ? 4 : 2;
    }

    private static float fallSlope(int armor) {
        float[] values = {0.4F, 0.7F, 1.0F};
        int rotation = Math.floorMod((int) (activeSeed / 3), values.length);
        int shifted = activeVariant == RuleVariant.SHIFTED ? 1 : 0;
        return values[Math.floorMod(armor + rotation + shifted, values.length)];
    }

    private static void resetMachines() {
        for (ScheduledMachine machine : MACHINES.values()) {
            machine.level.setBlockAndUpdate(machine.pos, Blocks.LODESTONE.defaultBlockState());
        }
        MACHINES.clear();
        TRANSFORMS.clear();
    }

    private static void resetPlayers(CommandSourceStack source) {
        for (ServerPlayer player : source.getServer().getPlayerList().getPlayers()) {
            player.setHealth(20.0F);
            player.getFoodData().setFoodLevel(20);
            player.getFoodData().setSaturation(5.0F);
        }
    }

    private static int sendState(CommandSourceStack source, String action) {
        String payload = "{\"action\":\"" + action + "\",\"seed\":" + activeSeed
            + ",\"variant\":\"" + activeVariant.id + "\"}";
        LOGGER.info("SCIENTIST_CONTROL {}", payload);
        source.sendSuccess(() -> Component.literal(payload), false);
        return Command.SINGLE_SUCCESS;
    }

    private static void logMechanism(String kind, BlockPos pos, Player player, boolean success) {
        LOGGER.info(
            "SCIENTIST_EVENT {{\"kind\":\"{}\",\"seed\":{},\"variant\":\"{}\","
                + "\"success\":{},\"position\":\"{}\",\"player\":\"{}\"}}",
            kind,
            activeSeed,
            activeVariant.id,
            success,
            pos.toShortString(),
            player.getUUID()
        );
    }

    private static boolean isMachineBlock(Block block) {
        return block == Blocks.LODESTONE
            || block == Blocks.COPPER_BLOCK
            || block == Blocks.ORANGE_GLAZED_TERRACOTTA
            || block == Blocks.EMERALD_BLOCK;
    }

    private static boolean isMiningTarget(BlockState state) {
        Block block = state.getBlock();
        return block == Blocks.STONE || block == Blocks.IRON_ORE || block == Blocks.DIAMOND_ORE;
    }

    private static int toolRank(Player player) {
        Item item = player.getMainHandItem().getItem();
        if (item == Items.WOODEN_PICKAXE) {
            return 0;
        }
        if (item == Items.STONE_PICKAXE) {
            return 1;
        }
        if (item == Items.IRON_PICKAXE) {
            return 2;
        }
        return -1;
    }

    private static int miningRequirement(BlockState state) {
        Block block = state.getBlock();
        if (activeVariant == RuleVariant.STANDARD) {
            if (block == Blocks.STONE) {
                return 0;
            }
            if (block == Blocks.IRON_ORE) {
                return 1;
            }
            if (block == Blocks.DIAMOND_ORE) {
                return 2;
            }
        } else {
            if (block == Blocks.STONE) {
                return 2;
            }
            if (block == Blocks.IRON_ORE) {
                return 0;
            }
            if (block == Blocks.DIAMOND_ORE) {
                return 1;
            }
        }
        return Integer.MAX_VALUE;
    }

    private static Item dropFor(BlockState state) {
        Block block = state.getBlock();
        if (block == Blocks.STONE) {
            return Items.COBBLESTONE;
        }
        if (block == Blocks.IRON_ORE) {
            return Items.RAW_IRON;
        }
        if (block == Blocks.DIAMOND_ORE) {
            return Items.DIAMOND;
        }
        return Items.AIR;
    }

    private static void grant(Player player, ItemStack reward) {
        if (!player.getInventory().add(reward)) {
            player.drop(reward, false);
        }
    }

    private static String machineKey(ServerLevel level, BlockPos pos) {
        return level.dimension().identifier() + ":" + pos.asLong();
    }

    private static String truthJson() {
        if (activeVariant == RuleVariant.STANDARD) {
            return "{\"seed\":" + activeSeed + ",\"variant\":\"standard\","
                + "\"mining\":{\"stone\":0,\"iron_ore\":1,\"diamond_ore\":2},"
                + "\"fall\":{\"safe_height\":" + fallSafeHeight()
                + ",\"slopes\":[" + fallSlope(0) + "," + fallSlope(1) + ","
                + fallSlope(2) + "]},"
                + "\"machine\":{\"input\":\"raw_copper\",\"protocol\":"
                + "[\"place\",\"activate\",\"wait:1\",\"interact\"],"
                + "\"delay\":2,\"output\":\"copper_ingot\"},"
                + "\"transforms\":[[\"chiseled_copper\",\"oak_log\",\"oak_planks\"],"
                + "[\"chiseled_tuff\",\"oak_planks\",\"stick\"],"
                + "[\"polished_blackstone\",\"stick\",\"torch\"]]}";
        }
        return "{\"seed\":" + activeSeed + ",\"variant\":\"shifted\","
            + "\"mining\":{\"stone\":2,\"iron_ore\":0,\"diamond_ore\":1},"
            + "\"fall\":{\"safe_height\":" + fallSafeHeight()
            + ",\"slopes\":[" + fallSlope(0) + "," + fallSlope(1) + ","
            + fallSlope(2) + "]},"
            + "\"machine\":{\"input\":\"oak_log\",\"protocol\":"
            + "[\"place\",\"activate\",\"wait:1\",\"interact\"],"
            + "\"delay\":2,\"output\":\"gold_ingot\"},"
            + "\"transforms\":[[\"chiseled_copper\",\"stick\",\"torch\"],"
            + "[\"chiseled_tuff\",\"oak_log\",\"oak_planks\"],"
            + "[\"polished_blackstone\",\"oak_planks\",\"stick\"]]}";
    }

    private record Transform(Item input, Item output) { }

    private record ScheduledMachine(ServerLevel level, BlockPos pos, long readyTick) { }

    private record ActiveTransform(
        ServerLevel level,
        BlockPos pos,
        Item output,
        boolean activated,
        long readyTick
    ) { }

    private enum RuleVariant {
        STANDARD("standard"),
        SHIFTED("shifted");

        private final String id;

        RuleVariant(String id) {
            this.id = id;
        }
    }
}
