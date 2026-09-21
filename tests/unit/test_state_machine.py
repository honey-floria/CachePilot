import unittest

from cachepilot.runtime.state_machine import (
    EventConflictError,
    InvalidTransitionError,
    RequestState,
    RequestStateMachine,
    TERMINAL_STATES,
    TokenEmissionError,
)


MAIN_PATH = (
    RequestState.TOKENIZED,
    RequestState.QUEUED,
    RequestState.ADMITTED,
    RequestState.ROUTED,
    RequestState.EXECUTING,
    RequestState.FINISHED,
)


class RequestStateMachineTests(unittest.TestCase):
    def create_machine(self) -> RequestStateMachine:
        return RequestStateMachine("req_1", "event_received")

    def advance_to(
        self, machine: RequestStateMachine, target: RequestState
    ) -> None:
        for index, state in enumerate(MAIN_PATH, 1):
            machine.transition(state, "event_{0}".format(index))
            if state is target:
                return

    def test_main_path_reaches_finished(self):
        machine = self.create_machine()

        for index, state in enumerate(MAIN_PATH, 1):
            result = machine.transition(state, "event_{0}".format(index))
            self.assertTrue(result.applied)
            self.assertEqual(result.current_state, state)

        self.assertEqual(machine.state, RequestState.FINISHED)
        self.assertTrue(machine.is_terminal)
        self.assertEqual(len(machine.transitions), 7)

    def test_illegal_transition_is_rejected_without_changing_state(self):
        machine = self.create_machine()

        with self.assertRaises(InvalidTransitionError):
            machine.transition(RequestState.ADMITTED, "event_skip")

        self.assertEqual(machine.state, RequestState.RECEIVED)
        self.assertEqual(len(machine.transitions), 1)

    def test_duplicate_transition_event_is_idempotent(self):
        machine = self.create_machine()
        first = machine.transition(RequestState.TOKENIZED, "event_tokenized")
        machine.transition(RequestState.QUEUED, "event_queued")

        duplicate = machine.transition(RequestState.TOKENIZED, "event_tokenized")

        self.assertFalse(duplicate.applied)
        self.assertEqual(duplicate.transition, first.transition)
        self.assertEqual(duplicate.current_state, RequestState.QUEUED)
        self.assertEqual(len(machine.transitions), 3)

    def test_reusing_event_id_for_another_event_is_rejected(self):
        machine = self.create_machine()
        machine.transition(RequestState.TOKENIZED, "event_shared")

        with self.assertRaises(EventConflictError):
            machine.transition(RequestState.QUEUED, "event_shared")

    def test_every_nonterminal_state_can_enter_each_error_terminal(self):
        nonterminal_states = (
            RequestState.RECEIVED,
            RequestState.TOKENIZED,
            RequestState.QUEUED,
            RequestState.ADMITTED,
            RequestState.ROUTED,
            RequestState.EXECUTING,
        )
        error_terminals = TERMINAL_STATES - {RequestState.FINISHED}

        for source in nonterminal_states:
            for terminal in error_terminals:
                with self.subTest(source=source, terminal=terminal):
                    machine = self.create_machine()
                    if source is not RequestState.RECEIVED:
                        self.advance_to(machine, source)
                    result = machine.transition(
                        terminal,
                        "terminal_{0}_{1}".format(source.value, terminal.value),
                    )
                    self.assertTrue(result.applied)
                    self.assertEqual(machine.state, terminal)

    def test_terminal_state_is_irreversible(self):
        for terminal in TERMINAL_STATES:
            with self.subTest(terminal=terminal):
                machine = self.create_machine()
                if terminal is RequestState.FINISHED:
                    self.advance_to(machine, RequestState.FINISHED)
                else:
                    machine.transition(terminal, "event_terminal")

                with self.assertRaises(InvalidTransitionError):
                    machine.transition(RequestState.TOKENIZED, "event_after_terminal")

                self.assertEqual(machine.state, terminal)

    def test_token_emission_is_allowed_only_while_executing(self):
        machine = self.create_machine()

        with self.assertRaises(TokenEmissionError):
            machine.record_token_emission("token_before_execution")

        self.advance_to(machine, RequestState.EXECUTING)
        self.assertTrue(machine.record_token_emission("token_1"))
        self.assertFalse(machine.record_token_emission("token_1"))
        self.assertEqual(machine.emitted_token_count, 1)

        machine.transition(RequestState.FINISHED, "event_finished")
        with self.assertRaises(TokenEmissionError):
            machine.record_token_emission("token_after_terminal")
        self.assertEqual(machine.emitted_token_count, 1)

    def test_transition_and_token_events_share_the_same_id_namespace(self):
        machine = self.create_machine()
        self.advance_to(machine, RequestState.EXECUTING)
        machine.record_token_emission("shared_event")

        with self.assertRaises(EventConflictError):
            machine.transition(RequestState.FINISHED, "shared_event")


if __name__ == "__main__":
    unittest.main()
