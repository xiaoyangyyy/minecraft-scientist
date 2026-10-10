"""Epoch-local clocks with retained, reversible global-world receipts.

Stationary learners expect local episode zero. This adapter never resets the
world's episode counter or its hidden schedule: it only rebases returned clocks,
while retaining original receipts for a separate global chronology audit.
"""

from weakref import ref

from minecraft_scientist.protocols import AgentWorldView


class RecordedEpoch:
    def __init__(self, world):
        self.world = world
        self.base = None
        self.controls = []
        self._view_ref = None

    @property
    def view(self):
        # The capability registry holds callbacks strongly, and those callbacks
        # keep this recorder alive while a caller holds its view. Do not retain
        # that registry's weak key here: owner -> key -> callback -> owner would
        # otherwise keep every completed epoch and all its receipts alive.
        view = None if self._view_ref is None else self._view_ref()
        if view is None:
            view = AgentWorldView(
                observe=self.observe,
                available=self.world.available_actions,
                step=self.step,
                begin_trial=self.begin_trial,
            )
            self._view_ref = ref(view)
        return view

    def local(self, observation):
        if self.base is None:
            raise ValueError("begin_trial is required before local observation")
        return observation.model_copy(
            update={"episode_id": observation.episode_id - self.base}, deep=True
        )

    def begin_trial(self):
        observation = self.world.begin_trial()
        if self.base is None:
            self.base = observation.episode_id
        self.controls.append(
            {"kind": "reset", "observation": observation.model_dump(mode="json")}
        )
        return self.local(observation)

    def observe(self):
        return self.local(self.world.observe())

    def step(self, action):
        record = self.world.step(action)
        self.controls.append(
            {"kind": "step", "transition": record.model_dump(mode="json")}
        )
        return record.model_copy(
            update={
                "state_before": self.local(record.state_before),
                "state_after": self.local(record.state_after),
            },
            deep=True,
        )
