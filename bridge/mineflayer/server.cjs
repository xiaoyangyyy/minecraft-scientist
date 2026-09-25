"use strict";

const readline = require("node:readline");
const mineflayer = require("mineflayer");
const { pathfinder, Movements, goals } = require("mineflayer-pathfinder");
const { Vec3 } = require("vec3");

let bot = null;
let step = 0;

function reply(id, result) {
  process.stdout.write(`${JSON.stringify({ id, ok: true, result })}\n`);
}

function reject(id, error) {
  process.stdout.write(
    `${JSON.stringify({ id, ok: false, error: String(error?.message || error) })}\n`,
  );
}

function requireBot() {
  if (!bot || !bot.entity) throw new Error("bot is not connected");
  return bot;
}

function waitForSpawn(instance, timeoutMs) {
  return new Promise((resolve, rejectPromise) => {
    const timer = setTimeout(() => {
      cleanup();
      rejectPromise(new Error(`spawn timeout after ${timeoutMs}ms`));
    }, timeoutMs);
    const onSpawn = () => {
      cleanup();
      resolve();
    };
    const onError = (error) => {
      cleanup();
      rejectPromise(error);
    };
    const onEnd = (reason) => {
      cleanup();
      rejectPromise(new Error(`connection ended before spawn: ${reason}`));
    };
    const cleanup = () => {
      clearTimeout(timer);
      instance.off("spawn", onSpawn);
      instance.off("error", onError);
      instance.off("end", onEnd);
    };
    instance.once("spawn", onSpawn);
    instance.once("error", onError);
    instance.once("end", onEnd);
  });
}

async function connect(params) {
  if (bot) throw new Error("bot is already connected");
  const options = {
    host: params.host || "127.0.0.1",
    port: Number(params.port || 25565),
    username: params.username || "ScientistBot",
    auth: params.auth || "offline",
    hideErrors: true,
  };
  if (params.version) options.version = params.version;
  bot = mineflayer.createBot(options);
  bot.loadPlugin(pathfinder);
  bot.on("kicked", (reason) => console.error("kicked", reason));
  bot.on("error", (error) => console.error("bot error", error.message));
  await waitForSpawn(bot, Number(params.timeout_ms || 30000));
  bot.pathfinder.thinkTimeout = Math.max(
    5000,
    Number(params.pathfinder_timeout_ms || 15000),
  );
  bot.pathfinder.setMovements(new Movements(bot));
  step = 0;
  return {
    username: bot.username,
    version: bot.version,
    entity_id: bot.entity.id,
  };
}

function observe(params = {}) {
  const instance = requireBot();
  const radius = Math.max(1, Math.min(32, Number(params.radius || 8)));
  const maxBlocks = Math.max(1, Math.min(2048, Number(params.max_blocks || 128)));
  const positions = instance.findBlocks({
    matching: (block) => block && block.name !== "air" && block.boundingBox !== "empty",
    maxDistance: radius,
    count: maxBlocks,
  });
  const blocks = positions.map((position) => {
    const block = instance.blockAt(position);
    return {
      id: block.type,
      name: block.name,
      position: { x: position.x, y: position.y, z: position.z },
      diggable: block.diggable,
      hardness: block.hardness,
    };
  });
  const entities = Object.values(instance.entities)
    .filter((entity) => entity.id !== instance.entity.id)
    .filter((entity) => entity.position.distanceTo(instance.entity.position) <= radius)
    .map((entity) => ({
      id: entity.id,
      name: entity.name || entity.username || entity.displayName || "unknown",
      type: entity.type,
      position: {
        x: entity.position.x,
        y: entity.position.y,
        z: entity.position.z,
      },
    }));
  return {
    step,
    position: {
      x: instance.entity.position.x,
      y: instance.entity.position.y,
      z: instance.entity.position.z,
    },
    yaw: instance.entity.yaw,
    pitch: instance.entity.pitch,
    health: instance.health,
    food: instance.food,
    time: instance.time.timeOfDay,
    inventory: instance.inventory.items().map((item) => ({
      id: item.type,
      name: item.name,
      count: item.count,
      slot: item.slot,
      max_durability: Number(item.maxDurability || 0),
    })),
    held_item: instance.heldItem
      ? {
          id: instance.heldItem.type,
          name: instance.heldItem.name,
          count: instance.heldItem.count,
          slot: instance.heldItem.slot,
        }
      : null,
    blocks,
    entities,
  };
}

async function act(params) {
  const instance = requireBot();
  const action = params.action;
  if (!action || typeof action.kind !== "string") throw new Error("action.kind is required");
  let actionSuccess = true;
  switch (action.kind) {
    case "look":
      await instance.look(Number(action.yaw), Number(action.pitch), true);
      break;
    case "move_to":
      await instance.pathfinder.goto(
        new goals.GoalNear(
          Number(action.x),
          Number(action.y),
          Number(action.z),
          Math.max(1, Number(action.range || 1)),
        ),
      );
      break;
    case "select_slot": {
      const item = instance.inventory.slots[Number(action.slot)];
      if (!item) throw new Error("selected inventory slot is empty");
      await instance.equip(item, "hand");
      break;
    }
    case "dig": {
      const position = new Vec3(Number(action.x), Number(action.y), Number(action.z));
      const block = instance.blockAt(position);
      if (!block || block.name === "air") throw new Error("target block is absent");
      try {
        await instance.dig(block, true);
      } catch (error) {
        // A cancelled server-authoritative break may first look successful in the
        // client's predicted block cache. Give the correction packet time to land
        // before deciding whether the block really changed.
        await new Promise((resolve) => setTimeout(resolve, 250));
        const remaining = instance.blockAt(position);
        if (!remaining || remaining.type !== block.type) throw error;
        actionSuccess = false;
      }
      await new Promise((resolve) => setTimeout(resolve, 250));
      const remaining = instance.blockAt(position);
      actionSuccess = actionSuccess && (!remaining || remaining.type !== block.type);
      break;
    }
    case "attack": {
      const entity = instance.entities[Number(action.entity_id)];
      if (!entity) throw new Error("target entity is absent");
      instance.attack(entity);
      break;
    }
    case "use_block": {
      const position = new Vec3(Number(action.x), Number(action.y), Number(action.z));
      const block = instance.blockAt(position);
      if (!block || block.name === "air") throw new Error("target block is absent");
      const inventoryBefore = instance.inventory.items()
        .map((item) => `${item.type}:${item.count}`)
        .sort()
        .join("|");
      const useSneak = action.intent === "activate";
      if (useSneak) {
        // Use an empty main hand as an explicit, server-authoritative activation
        // token. This avoids racing the independent player-input and block-use
        // packets used by modern protocol versions.
        await instance.unequip("hand");
      }
      await instance.activateBlock(block);
      await new Promise((resolve) => setTimeout(resolve, 250));
      const after = instance.blockAt(position);
      const inventoryAfter = instance.inventory.items()
        .map((item) => `${item.type}:${item.count}`)
        .sort()
        .join("|");
      actionSuccess =
        !after || after.type !== block.type || inventoryAfter !== inventoryBefore;
      break;
    }
    case "use_item":
      instance.activateItem(Boolean(action.offhand));
      await new Promise((resolve) => setTimeout(resolve, Number(action.duration_ms || 100)));
      instance.deactivateItem();
      break;
    case "place_block": {
      const reference = instance.blockAt(
        new Vec3(Number(action.x), Number(action.y), Number(action.z)),
      );
      if (!reference) throw new Error("reference block is absent");
      const face = action.face || { x: 0, y: 1, z: 0 };
      await instance.placeBlock(
        reference,
        new Vec3(Number(face.x), Number(face.y), Number(face.z)),
      );
      break;
    }
    case "wait":
      // One symbolic wait quantum deliberately exceeds the mod's ten-tick
      // mechanism delay.  Ordinary packet acknowledgement waits are only
      // 250 ms, so they cannot accidentally satisfy a missing wait step.
      await new Promise((resolve) =>
        setTimeout(resolve, Math.max(1, Number(action.ticks || 1)) * 750),
      );
      break;
    default:
      throw new Error(`unsupported action kind: ${action.kind}`);
  }
  step += 1;
  return {
    action: action.kind,
    success: actionSuccess,
    observation: observe(params.observe || {}),
  };
}

async function disconnect() {
  if (bot) {
    const instance = bot;
    bot = null;
    instance.quit("bridge disconnect");
  }
  return { disconnected: true };
}

const handlers = {
  ping: async () => ({ protocol: 1, bridge: "mineflayer", connected: Boolean(bot) }),
  connect,
  observe,
  act,
  disconnect,
};

const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
input.on("line", async (line) => {
  let request;
  try {
    request = JSON.parse(line);
    if (!request || typeof request.id !== "number" || typeof request.method !== "string") {
      throw new Error("request requires numeric id and string method");
    }
    const handler = handlers[request.method];
    if (!handler) throw new Error(`unsupported method: ${request.method}`);
    reply(request.id, await handler(request.params || {}));
  } catch (error) {
    reject(request?.id ?? -1, error);
  }
});

input.on("close", async () => {
  await disconnect();
  process.exit(0);
});
